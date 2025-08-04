# app/service/summary_graph_builder.py
"""summary_graph_builder.py
LangGraph 기반 요약 파이프라인 빌더.

모든 노드는 최대 3회 재시도하며, 실패 시 에러 메시지를 기록하고 종료한다.
"""
from __future__ import annotations

import time
import asyncio
from functools import wraps
from typing import Awaitable, Callable, List, Optional
import os

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field 

from app.prompts import (
    PROMPT_DETERMINE_WEB,
    PROMPT_GRADE,
    PROMPT_GENERATE,
    PROMPT_VERIFY,
    PROMPT_REFINE,
    PROMPT_TRANSLATE,
    PROMPT_FILTER_QUERY,
    PROMPT_TRANSLATE_AND_REFINE_QUERY,
)

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
    
    log: List[str] = Field(default_factory=list)

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
    """LangGraph 노드에 재시도 로직을 적용하는 데코레이터.

    `_RETRY` 횟수만큼 재시도하며 마지막 실패 시 상태 객체에 에러를 기록한다.

    Args:
        fn: SummaryState를 받아 비동기로 처리하는 함수이자 노드.

    Returns:
        자동 재시도 로직이 적용된 비동기 함수.
    """
    @wraps(fn)
    async def _wrap(st: SummaryState):  # type: ignore[override]
        for attempt in range(1, _RETRY + 1):
            try:
                t0 = time.perf_counter()
                result = await fn(st)
                elapsed = int((time.perf_counter() - t0) * 1000)  # ms
                st.log.append(
                    f"{fn.__name__} attempt {attempt} [{elapsed}ms]"
                )
                return result
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
    """요약 그래프를 빌드하는 LangGraph 파이프라인 생성기.

    Attributes:
        loader: PDF 로더 (URL → TextChunk 리스트).
        store: 벡터스토어 (저장 및 검색).
        web_search: 외부 검색 엔진 (WebSearchIF).
        llm: LangChain 호환 LLM 실행기.
        cache: 캐시 저장소 (요약, 로그, 등).
    """
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
        @safe_retry
        async def entry_router(st: SummaryState):
            """요청에 따른 초기 변수를 설정하고 캐시 상태를 확인한다.
            캐시에 데이터가 있을 경우 요약 데이터 또한 설정된다.
                
            Args:
                st: 현재 요청 상태.

            Returns:
                초기 변수(is_summary, cached, embedded 등)가 설정된 상태 객체.
            """
            st.is_summary = st.query.strip().upper() == "SUMMARY_ALL"
            st.cached = self.cache.exists_summary(st.file_id)
            if st.cached:
                st.summary = self.cache.get_summary(st.file_id)
            st.embedded = await self.store.has_chunks(st.file_id)  # type: ignore[arg-type]
            
            # 로그 기록은 서브 기능이므로 실패해도 작동을 멈추지 않고 계속 진행한다.
            try:
                self.cache.set_log(
                    st.file_id, st.url, st.query, st.lang, msg="entry"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[LOG] entry set_log 실패: {e}")

            return st

        def entry_branch(st: SummaryState) -> str:
            if st.error:
                return "finish"
            if st.is_summary:
                if st.cached:
                    return "translate"
                return "filter_query" if st.embedded else "load"
            return "filter_query" if st.embedded else "load"

        g.add_node("entry", entry_router)

        # 1. Load PDF ---------------------------------------------------
        @safe_retry
        async def load_pdf(st: SummaryState):
            """PDF 파일을 로드하여 텍스트 청크로 변환한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                텍스트 청크가 추가된 상태 객체.
            """
            st.chunks = await self.loader.load(st.url)
            return st

        g.add_node("load", load_pdf)

        # 2. Embed ------------------------------------------------------
        @safe_retry
        async def embed(st: SummaryState):
            """텍스트 청크를 임베딩을 거쳐 벡터스토어에 저장한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                임베딩된 텍스트 청크가 추가된 상태 객체.
            """
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
            """텍스트 청크를 요약한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                요약된 텍스트 청크가 추가된 상태 객체.
            """
            if st.chunks is None:
                st.chunks = await self.store.get_all(st.file_id)  # type: ignore[arg-type]
            st.summary = await self.llm.summarize(st.chunks)  # type: ignore[arg-type]
            return st

        g.add_node("summarize", summarize)

        # 3-Q. Retrieve -------------------------------------------------
        @safe_retry
        async def filter_query(st: SummaryState):
            """프롬프트를 파괴할 수 있는 쿼리문을 정제하고,
            영어로 쿼리를 번역한 뒤
            문서의 구조를 묻는 쿼리의 경우 구체적인 쿼리문으로 변경한다.
            
            Args:
                st: 현재 요청 상태.
            """
            
            # 쿼리가 SUMMARY_ALL인 경우 전체 요약 모드이므로 정제 과정정 생략.
            if st.query == "SUMMARY_ALL":
                return st
            
            # 쿼리가 프롬프트를 파괴할 수 있는 쿼리인지 판단.
            prompt = PROMPT_FILTER_QUERY.render(query=st.query)
            result = await self.llm.execute(prompt, think=True)
            
            # 프롬프트를 파괴할 수 있는 쿼리인 경우 경고 메시지 반환.
            if "yes" in result.lower():
                st.answer = "Security Notice: Your query was flagged as a potential attempt to override system behavior. Please refrain from such actions. Repeated attempts may lead to access restrictions."
            else:
                
                # 프롬프트를 파괴하지 않는는 쿼리인 경우 쿼리를 번역하고 구체적인 쿼리문으로 변경.
                prompt = PROMPT_TRANSLATE_AND_REFINE_QUERY.render(query=st.query)
                result = await self.llm.execute(prompt, think=True)
                st.query = result
            return st
        
        g.add_node("filter_query", filter_query)
        
        def post_filter_query(st: SummaryState) -> str:
            if st.error:
                return "finish"
            if st.answer == "Security Notice: Your query was flagged as a potential attempt to override system behavior. Please refrain from such actions. Repeated attempts may lead to access restrictions.":
                return "translate"
            else:
                return "RAG_router"
        
        g.add_conditional_edges("filter_query", post_filter_query, {
            "RAG_router": "RAG_router",
            "translate": "translate",
            "finish": "finish",
        })
        
        
        @safe_retry
        async def RAG_router(st: SummaryState):
            """RAG를 시작하기 전 어떻게 처리할지 LLM을 통해 결정한다.
            요약된 데이터와 쿼리를 보고 외부 검색 여부를 판단한다.
            만약 쿼리가 "SUMMARY_ALL"이면 판단 없이 넘어간다.
            
            Args:
                st: 현재 요청 상태.

            Returns:
                외부 검색 여부가 설정된 상태 객체.
            """
            if st.is_summary:
                return st
            
            if st.cached:
                st.summary = self.cache.get_summary(st.file_id)
            else:
                st.chunks = await self.store.get_all(st.file_id)
                st.summary = await self.llm.summarize(st.chunks)
            
            if os.getenv("TAVILY_API_KEY") == "":
                st.is_web = False
                return st
            
            prompt = PROMPT_DETERMINE_WEB.render(query=st.query, summary=st.summary)
            result = await self.llm.execute(prompt, think=True)
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
            """외부 검색 엔진을 통해 검색 결과를 가져온다.
            벡터스토어에서 유사도 검색을 통해 검색 결과 또한 일부 가져온다.

            Args:
                st: 현재 요청 상태.

            Returns:
                검색 결과가 추가된 상태 객체.(st.retrieved)
            """
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
            """벡터스토어에서 유사도 검색을 통해 검색 결과를 가져온다.

            Args:
                st: 현재 요청 상태.

            Returns:
                검색 결과가 추가된 상태 객체.(st.retrieved)
            """
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
            """
            Retrieve된 청크가 정말 쿼리문 응답에 필요한 정보인지 llm을 통해 문맥적으로 판단한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                필요한 텍스트 청크만으로 정제된 상태 객체.(st.retrieved)
            """
            if not st.retrieved:
                st.error = "No relevant chunks"
                return st
            
            good_chunks = []
            for chunk in st.retrieved:
                prompt = PROMPT_GRADE.render(query=st.query, summary=st.summary, chunk=chunk)
                result = await self.llm.execute(prompt, think=True)
                if "yes" in result.lower():
                    good_chunks.append(chunk)
            
            if len(good_chunks) == 0:
                st.answer = "I'm sorry, I can't find the answer to your question even though I read all the documents. Please ask a question about the document's content."
                return st
            st.retrieved = good_chunks
            return st
        
        g.add_node("grade", grade)
        
        def post_grade(st: SummaryState) -> str:
            if st.answer == "I'm sorry, I can't find the answer to your question even though I read all the documents. Please ask a question about the document's content.":
                return "translate"
            if st.error:
                return "finish"
            else:
                return "generate"
        
        g.add_conditional_edges("grade", post_grade, {
            "translate": "translate",
            "generate": "generate",
            "finish": "finish",
        })
        
        @safe_retry
        async def generate(st: SummaryState):
            """Retrieved된 청크를 바탕으로 쿼리문에 대한 답변을 생성한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                생성된 답변이 추가된 상태 객체.(st.answer)
            """
            prompt = PROMPT_GENERATE.render(query=st.query, retrieved=st.retrieved)
            st.answer = await self.llm.execute(prompt)
            return st
        
        g.add_node("generate", generate)
        
        def post_generate(st: SummaryState) -> str:
            return "finish" if st.error else "translate"
        
        g.add_conditional_edges("generate", post_generate, {
            "translate": "translate",
            "finish": "finish",
        })
        
        # 5. Save summary ----------------------------------------------
        @safe_retry
        async def save_summary(st: SummaryState):
            """요약 데이터를 캐시에 저장한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                요약 데이터가 캐시에 저장된 상태 객체.
            """
            if st.is_summary and not st.cached and st.summary:
                self.cache.set_summary(st.file_id, st.summary)
            return st

        g.add_node("save", save_summary)

        def post_save_summary(st: SummaryState) -> str:
            return "finish" if st.error else "translate"

        @safe_retry
        async def translate(st: SummaryState):
            """생성된 답변을 사용자 언어로 번역한다.

            Args:
                st: 현재 요청 상태.

            Returns:
                번역된 답변이 추가된 상태 객체.(st.answer)
            """
            if st.is_summary:
                text = self.cache.get_summary(st.file_id)
            else:
                text = st.answer
            
            # 사용자 언어가 영어일 경우 번역은 필요 없으므로 생략
            if st.lang == "EN":
                st.answer = text
                return st
            
            prompt = PROMPT_TRANSLATE.render(lang=st.lang, text=text)
            st.answer = await self.llm.execute(prompt)
            return st

        
        # 6. Translate & finish ----------------------------------------
        g.add_node("translate", translate)
        async def finish_node(st: SummaryState):
            """요약 프로세스가 종료되면 실행 로그를 기록한다.
            에러가 있으면 에러 메시지를, 없으면 실행 로그를 문자열로 묶어서 기록
            Args:
                st: 현재 요청 상태.

            Returns:
                실행 로그가 캐시에 저장된 상태 객체.
            """
            msg = st.error if st.error else " | ".join(st.log or [])
            # 로그 기록은 서브 기능이므로 실패해도 작동을 멈추지 않고 계속 진행한다.
            try:
                self.cache.set_log(
                    st.file_id, st.url, st.query, st.lang, msg=msg
                )
            except Exception as e:  # noqa: BLE001
                print(f"[LOG] finish set_log 실패: {e}")
            return st

        g.add_node("finish", finish_node)

        # Routing -------------------------------------------------------
        g.set_entry_point("entry")

        g.add_conditional_edges("entry", entry_branch, {
            "translate": "translate",
            "filter_query":  "filter_query",
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
            return "finish" if st.error else "filter_query"

        g.add_conditional_edges("embed", post_embed, {
            "filter_query":  "filter_query",
            "finish":    "finish",
        })

        g.add_edge("summarize",  "save")
        
        # --- save 노드 → translate / finish 분기 --------------------
        g.add_conditional_edges(
            "save",
            post_save_summary,
            {
                "translate": "translate",
                "finish": "finish",
            },
        )

        g.add_edge("translate",  "finish")

        g.set_finish_point("finish")
        return g.compile()

