import os
import tempfile
from typing import Final, List

import httpx                # ✅ async HTTP client
from PIL import Image, ImageOps
import pytesseract
import fitz                 # PyMuPDF

_TIMEOUT: Final[int] = 30  # seconds


class PDFReceiver:
    """PDF 링크 → 텍스트(이미지 OCR 포함). 100 % 비동기."""

    async def fetch_and_extract_text(self, url: str) -> str:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fp:
            fp.write(resp.content)
            pdf_path = fp.name

        try:
            parser = PDFParser()
            elements: List[str] = parser.read(pdf_path)
            return "\n".join(e for e in elements if e)
        finally:
            os.remove(pdf_path)


class PDFParser:
    def __init__(self, ocr_lang: str = "kor+eng"):
        self.ocr_lang = ocr_lang

    def read(self, pdf_path: str) -> List[str]:
        """텍스트 추출 + OCR fallback."""
        with fitz.open(pdf_path) as doc:
            texts = []
            for page in doc:
                text = page.get_text("text")
                if len(text.strip()) > 50:
                    texts.append(text)
                else:
                    texts.append(self._ocr_page(page))
        return texts

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _ocr_page(self, page, dpi: int = 300) -> str:
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = pix.pil_image
            gray = ImageOps.grayscale(img)
            bw = gray.point(lambda x: 0 if x < 180 else 255, "1")
            return pytesseract.image_to_string(bw, lang=self.ocr_lang, timeout=10)
        except Exception:
            return ""
