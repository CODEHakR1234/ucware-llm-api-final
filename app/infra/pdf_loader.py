"""
PdfLoader
=========

PDFReceiver → 텍스트 → RecursiveCharacterTextSplitter → List[str]
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.domain.interfaces import PdfLoaderIF, TextChunk
from app.receiver.pdf_receiver import PDFReceiver

# OCR 출력은 문장 길이가 짧은 편이므로 1 500자 chunk + overlap 200
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],   # 문단→문장→단어
)

class PdfLoader(PdfLoaderIF):
    async def load(self, url: str) -> List[TextChunk]:
        text = await PDFReceiver().fetch_and_extract_text(url)
        if not text.strip():
            raise ValueError("PDF 텍스트 추출 실패")
        return _splitter.split_text(text)

