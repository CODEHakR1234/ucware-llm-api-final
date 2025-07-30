from __future__ import annotations
import base64
from typing import List, Union, Dict

from app.domain.interfaces import PdfLoaderIF
from app.domain.page_chunk import PageChunk
from app.receiver.pdf_receiver import PDFReceiver
from app.chunker.semantic_chunker import SemanticChunker
from app.domain.page_element import PageElement
from app.vision.captioner import Captioner

# ──────────────── 싱글턴 ────────────────
_receiver = PDFReceiver()
_captioner = Captioner()
_chunker = SemanticChunker()

class PdfLoader(PdfLoaderIF):
    """
    PDF URL → TextChunk 또는 PageChunk 리스트로 변환.
    """

    async def load(
        self, 
        url: str, 
        *, 
        with_figures: bool = False
    ) -> List[Union[str, PageChunk]]:
        """
        PDF를 로드하여 청크 리스트로 반환.

        Parameters
        ----------
        url : str
            PDF URL
        with_figures : bool, default=False
            True: 자습서 모드 (PageChunk with figures)
            False: 요약/Q&A 모드 (plain text chunks)

        Returns
        -------
        List[Union[str, PageChunk]]
            with_figures=True: PageChunk 리스트
            with_figures=False: 문자열 리스트
        """
        # (1) PDF → PageElement 추출
        elements: List[PageElement] = await _receiver.fetch_and_extract_elements(url)

        # (2) 자습서 모드: 캡션 생성 + data-URI 변환
        if with_figures:
            # 이미지 요소들만 캡션 생성
            vis_elements = [e for e in elements if e.kind in ("figure", "table", "graph")]
            if vis_elements:
                # bytes → 캡션 생성
                captions = await _captioner.caption([e.content for e in vis_elements])
                
                # 캡션 적용 + bytes → data-URI 변환
                for element, caption in zip(vis_elements, captions):
                    element.caption = caption or "No caption."
                    # bytes → base64 data-URI
                    if isinstance(element.content, (bytes, bytearray)):
                        b64 = base64.b64encode(element.content).decode()
                        data_uri = f"data:image/png;base64,{b64}"
                        element.content = data_uri

        # (3) 청크 분할
        chunks = _chunker.group(elements, return_pagechunk=with_figures)
        if not chunks:
            raise ValueError("PDF 텍스트 추출 실패")

        # (4) 자습서 모드: 이미지 ID를 URI로 매핑
        if with_figures:
            # 이미지 ID → URI 매핑 생성
            image_mapping = self._create_image_mapping(elements)
            
            # PageChunk의 텍스트에서 플레이스홀더 교체 및 이미지 매핑
            for chunk in chunks:
                if isinstance(chunk, PageChunk):
                    # 텍스트에서 [IMG_id] → [IMG_id:caption] 교체
                    chunk.text = self._replace_placeholders_with_captions(chunk.text, elements)
                    
                    # figs에서 (image_id, uri) 튜플로 변환
                    chunk.figs = [(img_id, image_mapping.get(img_id, content)) for img_id, content in chunk.figs]
                    
                    # 매핑 실패한 이미지들 로깅
                    missing_images = [img_id for img_id, _ in chunk.figs if img_id not in image_mapping]
                    if missing_images:
                        print(f"[PdfLoader] 매핑 실패한 이미지들: {missing_images}", flush=True)

        # (5) 반환 형식 결정
        if with_figures:
            return chunks  # List[PageChunk]
        else:
            return [chunk if isinstance(chunk, str) else chunk.text for chunk in chunks]  # List[str]

    def _create_image_mapping(self, elements: List[PageElement]) -> Dict[str, str]:
        """PageElement에서 이미지 ID를 URI로 매핑하는 딕셔너리 생성."""
        image_mapping = {}
        
        for element in elements:
            if element.kind in ("figure", "table", "graph") and element.id:
                image_mapping[element.id] = element.content
        
        return image_mapping

    # 개선 제안: 한 번에 모든 교체
    def _replace_placeholders_with_captions(self, text: str, elements: List[PageElement]) -> str:
        replacements = {}
        for element in elements:
            if element.kind in ("figure", "table", "graph") and element.id:
                placeholder = f"[{element.id}]"
                caption = element.caption or "No caption."
                replacements[placeholder] = f"[{element.id}:{caption}]"
        
        for placeholder, replacement in replacements.items():
            text = text.replace(placeholder, replacement)
        return text

