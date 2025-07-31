"""PDFReceiver (Docling + SmolDocling)
=====================================
URL → List[PageElement]

• Docling의 Markdown 결과를 읽어 페이지 흐름 그대로
  PageElement(kind="text" | "figure") 리스트로 변환
• data-URI 이미지는 base64 디코딩, 원격 URL은 병렬 다운로드(동시 8개)
• OCR/PyMuPDF 제거로 속도 단축
"""

from __future__ import annotations

import asyncio, base64, re
from typing import Final, List, Tuple
import httpx
import torch

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs

from app.domain.page_element import PageElement

# ──────────────── 설정 ────────────────
_TIMEOUT = httpx.Timeout(30.0)
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

# GPU 최적화 설정
if torch.cuda.is_available():
    # GPU 메모리 할당 최적화
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ──────────────── Docling 설정 ────────────────
# GPU 가속을 위한 VLM 파이프라인 옵션 설정
_vlm_options = VlmPipelineOptions(
    model_spec=vlm_model_specs.SMOLDOCLING_VLM_SPEC,
    device="cuda" if torch.cuda.is_available() else "cpu",  # GPU 우선 사용
    batch_size=4,  # GPU 메모리에 맞게 배치 크기 조정
    max_new_tokens=512,
    temperature=0.1,
    use_flash_attention=True,  # GPU 메모리 효율성 향상
    use_cache=True,  # KV 캐시 사용으로 속도 향상
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # GPU에서 half precision 사용
)

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=_vlm_options
        )
    }
)

device_info = "GPU" if torch.cuda.is_available() else "CPU"
print(f"[PDFReceiver] Docling 초기화 완료 - {device_info} 모드", flush=True)
if torch.cuda.is_available():
    print(f"[PDFReceiver] GPU: {torch.cuda.get_device_name(0)}", flush=True)

class PDFReceiver:
    """
    PDF URL → PageElement 리스트로 변환.
    SmolDocling + Docling 기반으로 완전히 재작성.
    """

    async def fetch_and_extract_elements(self, url: str) -> List[PageElement]:
        """
        PDF URL에서 텍스트와 이미지를 추출하여 PageElement 리스트로 반환.
        GPU 가속을 사용하여 처리 속도를 향상시킵니다.
        
        Returns
        -------
        List[PageElement]
            추출된 페이지 요소들 (text, figure, table, graph)
        """
        try:
            print(f"[PDFReceiver] PDF 변환 시작: {url}", flush=True)
            
            # GPU 메모리 최적화 설정
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # GPU 메모리 정리
                start_memory = torch.cuda.memory_allocated(0)
                print(f"[PDFReceiver] GPU 메모리 사용량: {start_memory / 1024**3:.2f}GB", flush=True)
            
            # Docling으로 PDF → Markdown 변환 (GPU 가속)
            doc = _converter.convert(source=url).document
            markdown_content = doc.export_to_markdown()
            print(f"[PDFReceiver] PDF 변환 완료: {len(markdown_content)}자", flush=True)
            
            # GPU 메모리 사용량 모니터링
            if torch.cuda.is_available():
                end_memory = torch.cuda.memory_allocated(0)
                memory_used = (end_memory - start_memory) / 1024**3
                print(f"[PDFReceiver] GPU 메모리 사용량 변화: {memory_used:.2f}GB", flush=True)
            
            # SmolDocling 페이지 구분자로 분할
            # SmolDocling은 <page_break> 태그를 사용하거나 페이지 번호를 포함할 수 있음
            if "<page_break>" in markdown_content:
                pages = markdown_content.split("<page_break>")
            elif "\f" in markdown_content:  # form feed character
                pages = markdown_content.split("\f")
            else:
                # 페이지 구분자가 없으면 전체를 하나의 페이지로
                pages = [markdown_content]
            
        except Exception as e:
            raise ValueError(f"Docling PDF 변환 실패: {e}")

        elements: List[PageElement] = []
        remote_imgs: List[Tuple[int, str, str, str]] = []  # (page_idx, alt, url, img_id)

        for idx, pg_md in enumerate(pages):
            image_counter = 1  # 페이지별로 이미지 ID 카운터 초기화
            if not pg_md.strip():
                continue

            # (1) 텍스트 처리 먼저 - [IMG_{page}_{id}] 플레이스홀더 유지
            def _placeholder(m: re.Match) -> str:
                nonlocal image_counter
                img_id = f"IMG_{idx}_{image_counter}"
                image_counter += 1
                return f"[{img_id}]"

            text_with_fig = _IMG_RE.sub(_placeholder, pg_md)
            for para in re.split(r"\n{2,}", text_with_fig):
                if para.strip():
                    elements.append(PageElement("text", idx, para.strip()))

            # (2) 이미지 처리 - data-URI는 즉시 bytes로, remote는 수집
            for alt, src in _IMG_RE.findall(pg_md):
                img_id = f"IMG_{idx}_{image_counter}"
                image_counter += 1
                
                if src.startswith("data:image"):
                    # data-URI → bytes 변환
                    _, b64 = src.split(",", 1)
                    try:
                        img_bytes = base64.b64decode(b64)
                        elements.append(PageElement("figure", idx, img_bytes, caption=alt, id=img_id))
                    except Exception:
                        continue
                else:
                    # remote URL은 나중에 다운로드
                    remote_imgs.append((idx, alt, src, img_id))

        # (3) 원격 이미지 다운로드 (동시 8개 제한)
        if remote_imgs:
            sem = asyncio.Semaphore(8)
            
            async def _fetch(i: int, url: str):
                async with sem:
                    try:
                        r = await cli.get(url, follow_redirects=True)
                        return i, r
                    except Exception as e:
                        return i, e

            async with httpx.AsyncClient(timeout=_TIMEOUT) as cli:
                resps = await asyncio.gather(*(_fetch(i, u) for i, _, u, _ in remote_imgs))

            for (pg_idx, alt, _, img_id), (i, r) in zip(remote_imgs, resps):
                if isinstance(r, Exception) or r.status_code != 200:
                    continue
                elements.append(PageElement("figure", pg_idx, r.content, caption=alt, id=img_id))

        if not elements:
            raise ValueError("Docling PDF 파싱 결과가 없습니다")
        
        print(f"[PDFReceiver] 요소 추출 완료: {len(elements)}개 (텍스트: {len([e for e in elements if e.kind == 'text'])}, 이미지: {len([e for e in elements if e.kind in ('figure', 'table', 'graph')])})", flush=True)
        return elements
