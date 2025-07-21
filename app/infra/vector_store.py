# app/infrastructure/vector_store.py
from typing import List
from app.domain.interfaces import VectorStoreIF, TextChunk
from app.vectordb.vector_db import get_vector_db


class VectorStore(VectorStoreIF):
    def __init__(self):
        self.vdb = get_vector_db()

    async def upsert(self, chunks: List[TextChunk], doc_id: str) -> None:
        self.vdb.store(chunks, doc_id)

    async def similarity_search(
        self, doc_id: str, query: str, k: int = 8
    ) -> List[TextChunk]:
        docs = self.vdb.get_docs(doc_id, query, k)
        return [d.page_content for d in docs]

    async def has_chunks(self, doc_id: str) -> bool:
        """Return *True* if *doc_id* already has at least one chunk stored."""
        return self.vdb.has_chunks(doc_id)
    
    async def get_all(self, doc_id: str) -> List[TextChunk]:
        """Return **all** stored chunks for *doc_id* (plain strings)."""
        docs = self.vdb.get_all_chunks(doc_id)
        return [d.page_content for d in docs]
