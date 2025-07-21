# app/service/summary_graph_builder.py
"""LangGraph pipeline with robust error handling **+ 3-회 재시도**.

노드가 I/O 오류를 내면 최대 3번까지 재시도하고 모두 실패하면
`st.error`에 최종 예외 메시지를 기록한 뒤 `finish`로 단락(exit)한다.
"""
from __future__ import annotations

import asyncio
from functools import wraps
from typing import Awaitable, Callable, List, Optional

from langgraph.graph import StateGraph
from pydantic import BaseModel

from app.domain.interfaces import (
    CacheIF,
    LlmChainIF,
    PdfLoaderIF,
    TextChunk,
    WebSearchIF,
    VectorStoreIF,
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
class SummaryState(BaseModel):
    file_id: str
    url: str
    query: str
    lang: str

    chunks: Optional[List[TextChunk]] = None
    retrieved: Optional[List[TextChunk]] = None
    summary: Optional[str] = None
    answer:  Optional[str] = None
    
    log : Optional[List[str]] = [] # 임시 테스트용 로그

    cached: bool = False
    embedded: bool = False
    is_summary: bool = False
    error: Optional[str] = None
    
    is_web: Optional[bool] = None
    is_good: Optional[bool] = None


# ---------------------------------------------------------------------------
# Helper: safe-retry decorator
# ---------------------------------------------------------------------------
_RETRY = 3
_SLEEP = 1  # seconds between retries


def safe_retry(fn: Callable[[SummaryState], Awaitable[SummaryState]]):
    """Try ``fn`` up to `_RETRY` times; on final failure record error."""

    @wraps(fn)
    async def _wrap(st: SummaryState):  # type: ignore[override]
        for attempt in range(1, _RETRY + 1):
            try:
                st.log.append(f"{fn.__name__} attempt {attempt}")
                return await fn(st)
            except Exception as exc:  # noqa: BLE001
                if attempt == _RETRY:
                    st.error = f"{fn.__name__} failed after {_RETRY} tries: {exc}"
                    return st
                await asyncio.sleep(_SLEEP)
        return st  # nothing should reach here

    return _wrap


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
class SummaryGraphBuilder:
    def __init__(
        self,
        loader: PdfLoaderIF,
        store: VectorStoreIF,
        web_search: WebSearchIF,
        llm: LlmChainIF,
        cache: CacheIF,
    ):
        self.loader, self.store, self.web_search, self.llm, self.cache = loader, store, web_search, llm, cache

    # ------------------------------------------------------------------
    def build(self):
        g = StateGraph(SummaryState)

        # 0. Entry ------------------------------------------------------
        async def entry_router(st: SummaryState):
            st.is_summary = st.query.strip().upper() == "SUMMARY_ALL"
            st.cached = self.cache.exists_summary(st.file_id)
            if st.cached:
                st.summary = self.cache.get_summary(st.file_id)
            st.embedded = await self.store.has_chunks(st.file_id)  # type: ignore[arg-type]
            return st

        def entry_branch(st: SummaryState) -> str:
            if st.error:
                return "finish"
            if st.is_summary:
                if st.cached:
                    return "translate"
                return "RAG_router" if st.embedded else "load"
            return "RAG_router" if st.embedded else "load"

        g.add_node("entry", entry_router)

        # 1. Load PDF ---------------------------------------------------
        @safe_retry
        async def load_pdf(st: SummaryState):
            st.chunks = await self.loader.load(st.url)
            return st

        g.add_node("load", load_pdf)

        # 2. Embed ------------------------------------------------------
        @safe_retry
        async def embed(st: SummaryState):
            if st.chunks is None:
                raise ValueError("chunks is None — cannot embed")
            if not st.embedded:
                await self.store.upsert(st.chunks, st.file_id)  # type: ignore[arg-type]
                st.embedded = True
            return st

        g.add_node("embed", embed)

        # 3-S. Summarize -----------------------------------------------
        @safe_retry
        async def summarize(st: SummaryState):
            if st.chunks is None:
                st.chunks = await self.store.get_all(st.file_id)  # type: ignore[arg-type]
            st.summary = await self.llm.summarize(st.chunks)  # type: ignore[arg-type]
            return st

        g.add_node("summarize", summarize)

        # 3-Q. Retrieve -------------------------------------------------
        @safe_retry
        async def RAG_router(st: SummaryState):
            if st.cached:
                st.summary = self.cache.get_summary(st.file_id)
            else:
                st.chunks = await self.store.get_all(st.file_id)
                st.summary = await self.llm.summarize(st.chunks)
                
            prompt = """
                You are a helpful assistant that can determine if the answer of the query need extra information from the web.
                If the answer need extra information from the web, return 'true'.
                If the answer does not need extra information from the web, return 'false'.
                Query: {query}
                Summary: {summary}
                """
            prompt = prompt.format(query=st.query, summary=st.summary)
            result = await self.llm.execute(prompt)
            
            st.is_web = "true" in result.lower()
            
            return st
        
        g.add_node("RAG_router", RAG_router)
        
        def post_RAG_router(st: SummaryState) -> str:
            if st.error:
                return "finish"
            if st.is_summary:
                return "summarize"
            return "retrieve_web" if st.is_web else "retrieve_vector"
    
    
        g.add_conditional_edges("RAG_router", post_RAG_router, {
            "retrieve_web":  "retrieve_web",
            "retrieve_vector":  "retrieve_vector",
            "summarize":    "summarize",
            "finish":    "finish",
        })
        
        @safe_retry
        async def retrieve_web(st: SummaryState):
            search_task = self.web_search.search(st.query, k=5)
            vector_task = self.store.similarity_search(st.file_id, st.query, k=3)

            search_result, vector_result = await asyncio.gather(search_task, vector_task)
            
            st.retrieved = vector_result + search_result
            
            return st
        
        g.add_node("retrieve_web", retrieve_web)
        
        def post_retrieve_web(st: SummaryState) -> str:
            if st.error:
                return "finish"
            else:
                return "grade"
        
        @safe_retry
        async def retrieve_vector(st: SummaryState):
            st.retrieved = await self.store.similarity_search(st.file_id, st.query, k=8)
            return st
        
        def post_retrieve_vector(st: SummaryState) -> str:
            if st.error:
                return "finish"
            else:
                return "grade"
        
        g.add_node("retrieve_vector", retrieve_vector)
        
        g.add_conditional_edges("retrieve_vector", post_retrieve_vector, {
            "grade": "grade",
            "finish": "finish",
        })
        
        g.add_conditional_edges("retrieve_web", post_retrieve_web, {
            "grade": "grade",
            "finish": "finish",
        })
        
        @safe_retry
        async def grade(st: SummaryState):
            # 요약된 본문, Retrieved된 결과를 보고 문맥상 필요한지 확인
            # 필요하다면 추가적인 정보를 찾아야 한다면 'true'를 반환
            # 필요하지 않다면 'false'를 반환
            
            good_chunks = []
            for chunk in st.retrieved:
                prompt = """
                You are a grader assessing relevance of a retrieved document to a user question. \n 
                If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
                It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.\n
                YOU MUST RETURN ONLY 'yes' or 'no'.\n
                Query: {query}\n
                Summary: {summary}\n
                Retrieved: {chunk}\n
                """
                prompt = prompt.format(query=st.query, summary=st.summary, chunk=chunk)
                result = await self.llm.execute(prompt)
                if "yes" in result.lower():
                    good_chunks.append(chunk)
            
            st.retrieved = good_chunks
            return st
        
        g.add_node("grade", grade)
        
        def post_grade(st: SummaryState) -> str:
            if st.error:
                return "finish"
            else:
                return "generate"
        
        g.add_conditional_edges("grade", post_grade, {
            "generate": "generate",
            "finish": "finish",
        })
        
        @safe_retry
        async def generate(st: SummaryState):
            prompt = """
            You are a helpful assistant that can generate a answer of the query in English.
            Use the retrieved information to generate the answer.
            YOU MUST RETURN ONLY THE ANSWER, NOTHING ELSE.
            Query: {query}
            Retrieved: {retrieved}
            """
            prompt = prompt.format(query=st.query, retrieved=st.retrieved)
            st.answer = await self.llm.execute(prompt)
            return st
        
        g.add_node("generate", generate)
        
        def post_generate(st: SummaryState) -> str:
            return "finish" if st.error else "verify"
        
        g.add_conditional_edges("generate", post_generate, {
            "verify": "verify",
            "finish": "finish",
        })
        
        @safe_retry
        async def verify(st: SummaryState):
            # 생성된 답변을 검증하여 다음 내용들을 판별:
            # 1. 생성된 답변이 쿼리에 대한 적절한 답변인가?
            # 2. 생성된 답변이 retrieved 정보를 바탕으로 한 것인가?
            # 3. 생성된 답변이 논리적으로 일관성이 있는가?
            # 4. 생성된 답변이 완전하고 구체적인가?
            
            prompt = """
            You are a helpful assistant that can verify the quality of the generated answer.
            Please evaluate the answer based on the following criteria:
            1. Does the answer directly address the query?
            2. Is the answer based on the retrieved information?
            3. Is the answer logically consistent?
            4. Is the answer complete and specific?
            
            Query: {query}
            Summary: {summary}
            Retrieved Information: {retrieved}
            Generated Answer: {answer}
            
            Return 'good' if the answer meets all criteria, otherwise return 'bad'. Do not return anything else.
            """
            prompt = prompt.format(query=st.query, summary=st.summary, retrieved=st.retrieved, answer=st.answer)
            result = await self.llm.execute(prompt)
            st.is_good = "good" in result.lower()
            return st
        
        g.add_node("verify", verify)
        
        def post_verify(st: SummaryState) -> str:
            if st.error:
                return "finish"
            if not st.is_good:
                return "refine"
            return "save" if st.is_summary else "translate"
        
        g.add_conditional_edges("verify", post_verify, {
            "save": "save",
            "refine": "refine",
            "finish": "finish",
            "translate": "translate",
        })
        
        @safe_retry
        async def refine(st: SummaryState):
            prompt = """
            You are a helpful assistant that can do two things:
            1. If the query is not related to the document content, return ONLY this sentence: "I'm sorry, I can't find the answer to your question even though I read all the documents. Please ask a question about the document's content."
            2. If the query is related, refine the query to get more relevant and accurate information based on the document summary and retrieved information. Return ONLY the refined query, nothing else.

            Document Summary: {summary}
            Original Query: {query}
            Retrieved Information: {retrieved}
            Generated Answer: {answer}
            """
            prompt = prompt.format(
                summary=st.summary,
                query=st.query,
                retrieved=st.retrieved,
                answer=st.answer
            )
            result = await self.llm.execute(prompt)
            # 관련 없는 경우
            if "not related to the document content" in result:
                st.answer = result
                return st
            # 관련 있는 경우(리파인된 쿼리 반환)
            st.query = result
            return st
        
        g.add_node("refine", refine)
        
        def post_refine(st: SummaryState) -> str:
            if st.error:
                return "finish"
            if "not related to the document content" in st.answer:
                return "translate"
            else:
                return "RAG_router"
        g.add_conditional_edges("refine", post_refine, {
            "RAG_router": "RAG_router",
            "translate": "translate",
            "finish": "finish",
        })
        
        # 5. Save summary ----------------------------------------------
        async def save_summary(st: SummaryState):
            if st.is_summary and not st.cached and st.summary:
                self.cache.set_summary(st.file_id, st.summary)
            return st

        g.add_node("save", save_summary)

        @safe_retry
        async def translate(st: SummaryState):
            if st.is_summary:
                st.answer = self.cache.get_summary(st.file_id)
            
            prompt = """
            You are a helpful assistant that can translate the answer to User language.\n
            ONLY RETURN THE TRANSLATED SEQUENCE, NOTHING ELSE.\n
            User language: {lang}\n
            Answer: {answer}\n
            """
            prompt = prompt.format(lang=st.lang, answer=st.answer)
            st.answer = await self.llm.execute(prompt)
            return st


        # 6. Translate & finish ----------------------------------------
        g.add_node("translate", translate)
        g.add_node("finish",    lambda st: st)

        # Routing -------------------------------------------------------
        g.set_entry_point("entry")

        g.add_conditional_edges("entry", entry_branch, {
            "translate": "translate",
            "RAG_router":  "RAG_router",
            "load":      "load",
            "finish":    "finish",
        })

        def post_load(st: SummaryState) -> str:
            return "finish" if st.error else "embed"

        g.add_conditional_edges("load", post_load, {
            "embed":  "embed",
            "finish": "finish",
        })

        def post_embed(st: SummaryState) -> str:
            return "finish" if st.error else "RAG_router"

        g.add_conditional_edges("embed", post_embed, {
            "RAG_router":  "RAG_router",
            "finish":    "finish",
        })

        g.add_edge("summarize",  "save")
        g.add_edge("save",       "translate")
        g.add_edge("translate",  "finish")

        g.set_finish_point("finish")
        return g.compile()

