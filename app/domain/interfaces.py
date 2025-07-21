# app/domain/interfaces.py
from abc import ABC, abstractmethod
from typing import List, Optional, Protocol

TextChunk = str


class PdfLoaderIF(Protocol):
    @abstractmethod
    async def load(self, url: str) -> List[TextChunk]: ...

class WebSearchIF(Protocol):
    @abstractmethod
    async def search(self, query: str) -> List[TextChunk]: ...

class VectorStoreIF(Protocol):
    @abstractmethod
    async def upsert(self, chunks: List[TextChunk], doc_id: str) -> None: ...

    @abstractmethod
    async def similarity_search(self, doc_id: str, query: str, k: int = 5) -> List[TextChunk]: ...
    
    @abstractmethod
    async def get_all(self, doc_id: str) -> List[TextChunk]: ... #문서 전체 갖고오기

    @abstractmethod
    async def has_chunks(self, doc_id: str) -> bool: ...

class LlmChainIF(Protocol):
    @abstractmethod
    async def execute(self, prompt: str) -> str: ...

    @abstractmethod
    async def summarize(self, docs: List[TextChunk]) -> str: ...

class CacheIF(Protocol):
    """요약 결과 캐싱을 위한 최소 계약(Port)"""

    @abstractmethod
    def get_summary(self, key: str) -> Optional[str]: ...

    @abstractmethod
    def set_summary(self, key: str, summary: str) -> None: ...

    @abstractmethod
    def exists_summary(self, key: str) -> bool: ...

