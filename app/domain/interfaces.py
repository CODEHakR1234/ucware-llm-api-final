# app/domain/interfaces.py

"""interfaces.py
서비스 레이어에서 사용하는 **인터페이스(Protocol)** 모음입니다.
구현체는 infra 디렉터리에서 제공하며, 여기서는 메서드 시그니처만 정의합니다.
"""

from abc import abstractmethod
from typing import List, Optional, Protocol

# 공용 타입 -------------------------------------------------------------
TextChunk = str  # PDF에서 추출한 텍스트 조각


class PdfLoaderIF(Protocol):
    """PDF URL → TextChunk 리스트."""

    @abstractmethod
    async def load(self, url: str) -> List[TextChunk]: ...


class WebSearchIF(Protocol):
    """외부 검색 결과를 TextChunk 리스트로 반환."""

    @abstractmethod
    async def search(self, query: str) -> List[TextChunk]: ...


class VectorStoreIF(Protocol):
    """문서 청크 저장·검색용 벡터 스토어."""

    @abstractmethod
    async def upsert(self, chunks: List[TextChunk], doc_id: str) -> None: ...

    @abstractmethod
    async def similarity_search(
        self, doc_id: str, query: str, k: int = 5
    ) -> List[TextChunk]: ...

    @abstractmethod
    async def get_all(self, doc_id: str) -> List[TextChunk]: ...  # 전체 청크

    @abstractmethod
    async def has_chunks(self, doc_id: str) -> bool: ...


class LlmChainIF(Protocol):
    """LLM 호출 래퍼."""

    @abstractmethod
    async def execute(self, prompt: str, think: bool = True) -> str: ...

    @abstractmethod
    async def summarize(self, docs: List[TextChunk]) -> str: ...


class CacheIF(Protocol):
    """요약 캐싱(Port)."""

    @abstractmethod
    def get_summary(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def set_summary(self, key: str, summary: str) -> None: ...

    @abstractmethod
    def exists_summary(self, key: str) -> bool: ...

