"""
PDFReceiver  ★Partial-OCR 버전
=====================================

URL → (텍스트 레이어 ∪ OCR 후보) → 문자열

• fitz.open(stream=bytes)  → 임시 파일 I/O 제거
• 페이지를 block 단위로 분석해
    ├─ 충분한 텍스트 블록  → 그대로 사용
    └─ 짧은 텍스트·이미지 블록 → PaddleOCR GPU
• asyncio + to_thread 로 병렬·논블로킹
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Final, List

import httpx
import fitz                                   # PyMuPDF
from paddleocr import PaddleOCR
from PIL import Image                          # Pillow: for crop

_TIMEOUT: Final[int] = 30
_MAX_PDF_SIZE = 50 * 1024 * 1024  # 50 MB

# ────────────────────────── PaddleOCR 싱글턴 ──────────────────────────
@lru_cache(maxsize=1)
def get_paddle_ocr() -> PaddleOCR:
    return PaddleOCR(
        use_gpu=True,
        gpu_mem=8000,          # A100 40 GB → 8 GB 선점
        lang="korean",         # ko/en 혼합
        show_log=False,
    )

# ────────────────────────── 튜닝 상수 ──────────────────────────
_MIN_PLAIN_CHARS = 50           # 블록이 이 이상이면 OCR 생략
_MIN_PLAIN_WORDS = 10
_MIN_IMG_AREA     = 8_000       # (px²) 후보 이미지 최소 면적

# ────────────────────────── 메인 클래스 ──────────────────────────
class PDFReceiver:
    """PDF URL → 전체 텍스트(부족 영역만 GPU OCR)."""

    async def fetch_and_extract_text(self, url: str) -> str:
        pdf_bytes = await self._download(url)
        return await self._extract_text(pdf_bytes)

    # ----------------------------------------------------------
    async def _download(self, url: str) -> bytes:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as cli:
            resp = await cli.get(url)
            resp.raise_for_status()
            if len(resp.content) > _MAX_PDF_SIZE:
                raise ValueError("PDF too large (> 50 MB)")
            return resp.content

    # ----------------------------------------------------------
    async def _extract_text(self, pdf_bytes: bytes) -> str:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            tasks = [self._page_to_text(doc, idx) for idx in range(doc.page_count)]
            pages: List[str] = await asyncio.gather(*tasks)
        return "\n".join(filter(None, pages))

    # ----------------------------------------------------------
    async def _page_to_text(self, doc: fitz.Document, idx: int) -> str:
        page = doc.load_page(idx)
        blocks = page.get_text("blocks")            # list[tuple]

        texts: List[str] = []
        ocr_tasks: List[asyncio.Future] = []

        for b in blocks:
            x0, y0, x1, y1, block_type = b[:5]

            # ── 텍스트 블록 ─────────────────────────────
            if block_type == 0:                     # 0 == text
                text = b[4].strip() if len(b) >= 6 else b[5].strip()
                if len(text) >= _MIN_PLAIN_CHARS or len(text.split()) >= _MIN_PLAIN_WORDS:
                    texts.append(text)
                else:
                    ocr_tasks.append(
                        asyncio.to_thread(self._ocr_crop, page, (x0, y0, x1, y1))
                    )

            # ── 이미지/그림 블록 ────────────────────────
            elif block_type == 1:
                if (x1 - x0) * (y1 - y0) >= _MIN_IMG_AREA:
                    ocr_tasks.append(
                        asyncio.to_thread(self._ocr_crop, page, (x0, y0, x1, y1))
                    )

        if ocr_tasks:
            texts += await asyncio.gather(*ocr_tasks)

        return " ".join(_dedup_lines(texts))

    # ----------------------------------------------------------
    def _ocr_crop(self, page: fitz.Page, bbox, dpi: int = 300) -> str:
        """bbox 영역만 잘라 PaddleOCR 실행 (동기 함수)."""
        rect = fitz.Rect(bbox)
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, clip=rect)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        result = get_paddle_ocr().ocr(img, cls=False)[0]
        return " ".join([ln[1][0] for ln in result if ln[1][0].strip()])

# ────────────────────────── 헬퍼: 라인 유사도 중복 제거 ──────────────────
from rapidfuzz import fuzz

def _dedup_lines(lines: List[str], thresh: int = 90) -> List[str]:
    kept: List[str] = []
    for txt in lines:
        if all(fuzz.ratio(txt, k) < thresh for k in kept):
            kept.append(txt)
    return kept

