"""pdf_receiver.py
외부 PDF 링크로부터 논문을 다운로드한 뒤, 텍스트를 추출한다.
OCR 기반 보완도 포함된 비동기 파서.
"""

import os
import tempfile
from typing import Final, List

import httpx                # ✅ async HTTP client
from PIL import Image, ImageOps
import pytesseract
import fitz                 # PyMuPDF

_TIMEOUT: Final[int] = 30  # seconds


class PDFReceiver:
    """PDF 링크 처리기.

    PDF 링크를 받아 다운로드하고, 내부 텍스트와 OCR 기반 내용을 추출한다.
    """

    async def fetch_and_extract_text(self, url: str) -> str:
        """PDF 파일을 받아 텍스트로 변환한다.

        Args:
            url: PDF 파일의 URL 링크.

        Returns:
            텍스트 추출 결과 문자열 (OCR 보완 포함).
        """
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
    """PDF 텍스트 파서 + OCR 보완.

    Attributes:
        ocr_lang: pytesseract 언어 설정 문자열 (기본: kor+eng)
    """

    def __init__(self, ocr_lang: str = "kor+eng"):
        self.ocr_lang = ocr_lang

    def read(self, pdf_path: str) -> List[str]:
        """PDF 전체 페이지에서 텍스트를 추출한다.

        각 페이지에 대해 기본 텍스트 추출을 시도하고,
        텍스트가 부족한 경우 OCR을 자동 수행한다.

        Args:
            pdf_path: 로컬 PDF 파일 경로.

        Returns:
            페이지별 텍스트 목록.
        """
        with fitz.open(pdf_path) as doc:
            texts = []
            for page in doc:
                text = page.get_text("text")
                if len(text.strip()) > 50:
                    texts.append(text)
                else:
                    texts.append(self._ocr_page(page))
        return texts

    # ───────────────────── 내부 OCR 헬퍼 ─────────────────────
    def _ocr_page(self, page, dpi: int = 300) -> str:
        """OCR을 통해 텍스트를 추출한다 (페이지 렌더링 기반).

        페이지를 이미지로 렌더링한 뒤 grayscale + 이진화 처리 후 pytesseract로 텍스트 인식.

        Args:
            page: PyMuPDF 페이지 객체.
            dpi: 렌더링 해상도 (기본 300).

        Returns:
            OCR 추출 문자열. 실패 시 빈 문자열 반환.
        """
        try:
            pix = page.get_pixmap(dpi=dpi)
            img = pix.pil_image
            gray = ImageOps.grayscale(img)
            bw = gray.point(lambda x: 0 if x < 180 else 255, "1")
            return pytesseract.image_to_string(bw, lang=self.ocr_lang, timeout=10)
        except Exception:
            return ""
