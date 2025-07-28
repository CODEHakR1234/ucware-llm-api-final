from __future__ import annotations
from typing import List, Union

from app.domain.interfaces   import PdfLoaderIF, TextChunk
from app.domain.page_chunk   import PageChunk
from app.domain.page_element import PageElement
from app.receiver.pdf_receiver import PDFReceiver
from app.vision.captioner      import Captioner
from app.chunker.semantic_chunker import SemanticChunker
from app.infra.figure_store     import FigureStore

_receiver  = PDFReceiver()
_captioner = Captioner()
_chunker   = SemanticChunker()
_store     = FigureStore()

class PdfLoader(PdfLoaderIF):
    async def load(
        self,
        url: str,
        *,
        with_figures: bool = False,
    ) -> List[Union[TextChunk, PageChunk]]:
        """PDF → 청크

        Parameters
        ----------
        url          : PDF URL
        with_figures : True  → PageChunk 리스트 (GuideGraph 용)
                       False → plain str 리스트 (SummaryGraph 용)
        """
        elements: List[PageElement] = await _receiver.fetch_and_extract_elements(url)

        # ── (1) 자습서 모드: 캡션 + 디스크 저장 ───────────────────
        if with_figures:
            vis = [e for e in elements if e.kind in ("figure", "table", "graph")]
            if vis:
                caps = await _captioner.caption([e.content for e in vis])
                urls = _store.save_many(url.split("/")[-1], [e.content for e in vis])
                for e, cap, u in zip(vis, caps, urls):
                    e.caption, e.content = cap or "No caption.", u

        # ── (2) 청크 분할 ────────────────────────────────────────
        chunks = _chunker.group(elements, return_pagechunk=with_figures)
        if not chunks:
            raise ValueError("PDF 텍스트 추출 실패")

        # GuideGraph → PageChunk / SummaryGraph → str
        if with_figures:
            return chunks                       # List[PageChunk]
        return [ck if isinstance(ck, str) else ck.text for ck in chunks]  # List[str]

