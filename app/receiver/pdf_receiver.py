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
# 성능 최적화된 SmolDocling 설정 (인터넷 검색 결과 기반)
try:
    # 성능 최적화 옵션 설정
    pipeline_options = VlmPipelineOptions(
        # 배치 크기 최적화 (GPU 메모리에 따라 조정)
        batch_size=8 if torch.cuda.is_available() else 4,
        # 토큰 수 제한으로 처리 속도 향상
        max_new_tokens=256,  # 기본값보다 줄임
        # 온도 낮춰서 일관성 향상 및 속도 개선
        temperature=0.1,
        # Flash Attention 사용 (GPU 메모리 효율성)
        use_flash_attention=True if torch.cuda.is_available() else False,
        # KV 캐시 사용으로 반복 계산 방지
        use_cache=True,
        # GPU에서 half precision 사용
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        # 병렬 처리 활성화
        num_workers=2 if torch.cuda.is_available() else 1
    )
    
    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options
            )
        }
    )
    print("[PDFReceiver] Docling 성능 최적화 설정으로 초기화 완료", flush=True)
    
    # GPU 가속이 가능한지 확인
    if torch.cuda.is_available():
        print(f"[PDFReceiver] GPU 사용 가능: {torch.cuda.get_device_name(0)}", flush=True)
        # GPU 메모리 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[PDFReceiver] GPU 최적화 설정 완료", flush=True)
    
except Exception as e:
    print(f"[PDFReceiver] Docling 초기화 실패: {e}", flush=True)
    raise

device_info = "GPU" if torch.cuda.is_available() else "CPU"
print(f"[PDFReceiver] Docling 초기화 완료 - {device_info} 모드", flush=True)
if torch.cuda.is_available():
    print(f"[PDFReceiver] GPU: {torch.cuda.get_device_name(0)}", flush=True)

class PDFReceiver:
    """
    PDF URL → PageElement 리스트로 변환.
    SmolDocling + Docling 기반으로 완전히 재작성.
    성능 최적화: 캐싱, 배치 처리, GPU 가속 적용.
    """
    
    def __init__(self):
        # 간단한 메모리 캐시 (URL → 처리 결과)
        self._cache = {}
        self._cache_size_limit = 10  # 최대 캐시 크기

    async def fetch_and_extract_elements(self, url: str) -> List[PageElement]:
        """
        PDF URL에서 텍스트와 이미지를 추출하여 PageElement 리스트로 반환.
        성능 최적화: 캐싱, GPU 가속, 배치 처리 적용.
        
        Returns
        -------
        List[PageElement]
            추출된 페이지 요소들 (text, figure, table, graph)
        """
        # 캐시 확인
        if url in self._cache:
            print(f"[PDFReceiver] 캐시 히트: {url}", flush=True)
            return self._cache[url]
        
        try:
            print(f"[PDFReceiver] PDF 변환 시작: {url}", flush=True)
            
            # GPU 메모리 최적화 설정
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # GPU 메모리 정리
                start_memory = torch.cuda.memory_allocated(0)
                print(f"[PDFReceiver] GPU 메모리 사용량: {start_memory / 1024**3:.2f}GB", flush=True)
                
                # 추가 GPU 최적화 설정
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            # 성능 최적화된 PDF 변환
            import time
            start_time = time.perf_counter()
            
            # Docling으로 PDF → Markdown 변환 (성능 최적화)
            doc = _converter.convert(source=url).document
            markdown_content = doc.export_to_markdown()
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            print(f"[PDFReceiver] PDF 변환 완료: {len(markdown_content)}자 ({processing_time:.2f}초)", flush=True)
            
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
        
        # 결과를 캐시에 저장
        self._cache[url] = elements
        
        # 캐시 크기 제한 관리
        if len(self._cache) > self._cache_size_limit:
            # 가장 오래된 항목 제거 (FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        print(f"[PDFReceiver] 요소 추출 완료: {len(elements)}개 (텍스트: {len([e for e in elements if e.kind == 'text'])}, 이미지: {len([e for e in elements if e.kind in ('figure', 'table', 'graph')])})", flush=True)
        return elements
