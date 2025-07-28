"""
PDFReceiver  ★Partial-OCR + 멀티모달
==================================
URL → List[PageElement]

• fitz.open(stream=bytes)  – 임시파일 없이 메모리에서 바로 로드
• 페이지 블록 분석
    ├─ 긴 텍스트 블록          → PageElement(kind="text", content=str)
    ├─ 짧은 텍스트·이미지 블록  → OCR / 이미지 bytes
• asyncio + to_thread 로 완전 논블로킹 처리
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Final, List

import httpx
import fitz                                       # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image

from app.domain.page_element import PageElement

_TIMEOUT: Final[int] = 30
_MAX_PDF_SIZE = 50 * 1024 * 1024   # 50 MB

# ────────────────────────── PaddleOCR 싱글턴 ──────────────────────────
@lru_cache(maxsize=1)
def get_paddle_ocr() -> PaddleOCR:
    try:
        return PaddleOCR(use_gpu=True, gpu_mem=8_000, lang="korean", show_log=False)
    except Exception:
        # GPU 없을 때 자동 CPU 모드
        return PaddleOCR(use_gpu=False, lang="korean", show_log=False)
# ────────────────────────── 튜닝 상수 ──────────────────────────
_MIN_PLAIN_CHARS = 50     # 이보다 짧은 텍스트 블록이면 OCR 후보
_MIN_PLAIN_WORDS = 10
_MIN_IMG_AREA     = 8_000 # px² 이상인 이미지만 그림 후보

# ────────────────────────── 메인 클래스 ──────────────────────────
class PDFReceiver:
    """PDF URL → PageElement 리스트(텍스트 + 그림·표)."""

    async def fetch_and_extract_elements(self, url: str) -> List[PageElement]:
        pdf_bytes = await self._download(url)
        return await self._extract_elements(pdf_bytes)

    # --------------------------------------------------------------
    async def _download(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as cli:
            resp = await cli.get(url)
            resp.raise_for_status()
            if len(resp.content) > _MAX_PDF_SIZE:
                raise ValueError("PDF too large (> 50 MB)")
            return resp.content

    # --------------------------------------------------------------
    async def _extract_elements(self, pdf_bytes: bytes) -> List[PageElement]:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            tasks = [self._page_to_elements(doc, idx) for idx in range(doc.page_count)]
            pages: List[List[PageElement]] = await asyncio.gather(*tasks)
        # 2-D 리스트 → 1-D
        return [el for pg in pages for el in pg]

    # --------------------------------------------------------------
    async def _page_to_elements(self, doc: fitz.Document, idx: int) -> List[PageElement]:
        page   = doc.load_page(idx)
        blocks = page.get_text("blocks")            # list of tuples

        elements: List[PageElement] = []
        ocr_tasks: List[asyncio.Future] = []

        for blk in blocks:
            x0, y0, x1, y1, btype = blk[:5]
            bbox = (x0, y0, x1, y1)

            # ── 텍스트 블록 ──────────────────────────────────────
            if btype == 0:
                text = blk[4].strip() if len(blk) >= 6 else blk[5].strip()
                if len(text) >= _MIN_PLAIN_CHARS or len(text.split()) >= _MIN_PLAIN_WORDS:
                    elements.append(PageElement("text", idx, text))
                else:  # 너무 짧으면 OCR 로 재검
                    ocr_tasks.append(
                        asyncio.to_thread(self._ocr_crop_text, page, bbox, idx)
                    )

            # ── 이미지/표/그래프 블록 ────────────────────────────
            elif btype == 1 and (x1 - x0) * (y1 - y0) >= _MIN_IMG_AREA:
                img_bytes = self._crop_bytes(page, bbox)
                elements.append(PageElement("figure", idx, img_bytes))

        # OCR 결과 추가
        if ocr_tasks:
            elements += await asyncio.gather(*ocr_tasks)

        return elements

    # --------------------------------------------------------------
    # helper: 크롭 PNG bytes
    @staticmethod
    def _crop_bytes(page: fitz.Page, bbox, dpi: int = 300) -> bytes:
        rect = fitz.Rect(bbox)
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, clip=rect)
        return pix.tobytes("png")

    # helper: OCR 후 PageElement(text)
    @staticmethod
    def _ocr_crop_text(page: fitz.Page, bbox, page_no: int, dpi: int = 300) -> PageElement:
        rect = fitz.Rect(bbox)
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, clip=rect)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = " ".join(
            seg[1][0] for seg in get_paddle_ocr().ocr(img, cls=False)[0]
            if seg[1][0].strip()
        )
        return PageElement("text", page_no, text)

