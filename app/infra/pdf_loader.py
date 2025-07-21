from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.domain.interfaces import PdfLoaderIF, TextChunk
from app.receiver.pdf_receiver import PDFReceiver


class PdfLoader(PdfLoaderIF):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    async def load(self, url: str) -> List[TextChunk]:
        text = await PDFReceiver().fetch_and_extract_text(url)   # ✅ await
        if not text.strip():
            raise ValueError("PDF 텍스트 추출 실패")
        return self.splitter.split_text(text)

