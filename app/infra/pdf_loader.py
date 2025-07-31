"""pdf_loader.py
PDF URL을 읽어 텍스트 청크(list[str])로 변환하는 로더.

- `PDFReceiver`가 PDF를 다운로드·텍스트/OCR 추출.
- `RecursiveCharacterTextSplitter`로 2 000자 청크(중첩 200) 분할.
- 인프라 계층에서 PdfLoaderIF(Port)를 구현한다.
"""

from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.domain.interfaces import PdfLoaderIF, TextChunk
from app.infra.pdf_receiver import PDFReceiver


class PdfLoader(PdfLoaderIF):
    """PdfLoaderIF 구현체.

    Attributes
    ----------
    splitter : RecursiveCharacterTextSplitter
        1500자 청크 + 200자 오버랩 설정.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],   # 문단→문장→단어
    )


    async def load(self, url: str) -> List[TextChunk]:
        """PDF를 다운로드해 텍스트 청크 리스트로 반환한다.

        Args
        ----
        url : str
            PDF 파일의 HTTP(S) URL.

        Returns
        -------
        List[str]
            잘라낸 텍스트 청크 목록.
        """
        text = await PDFReceiver().fetch_and_extract_text(url)   # ✅ await
        if not text.strip():
            raise ValueError("PDF 텍스트 추출 실패")
        return self.splitter.split_text(text)

