# app/service/guide_graph_builder.py
"""GuideGraphBuilder – 멀티모달 PDF → 튜토리얼 Markdown
entry·translate 노드를 제거한 경량 파이프라인.  
**캐시 히트 시 즉시 종료** 로직도 삭제했다.
"""

from __future__ import annotations

import asyncio, time
from functools import wraps
from typing import List, Optional, Awaitable, Callable

from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from app.domain.interfaces import PdfLoaderIF, VectorStoreIF, LlmChainIF, CacheIF
from app.domain.page_chunk import PageChunk
from app.utils.segment import segment_in_order
from app.infra.embedder import get_embed_fn
from app.prompts import PROMPT_TUTORIAL

# ---------------------------------------------------------------------
_RETRY, _SLEEP = 3, 1


def retry(fn: Callable[[BaseModel], Awaitable[BaseModel]]):
    @wraps(fn)
    async def _wrap(st):
        for i in range(1, _RETRY + 1):
            try:
                t0 = time.perf_counter()
                res = await fn(st)
                st.log.append(f"{fn.__name__}:{i}:{int((time.perf_counter()-t0)*1000)}ms")
                return res
            except Exception as e:
                if i == _RETRY:
                    st.error = f"{fn.__name__} failed: {e}"
                    return st
                await asyncio.sleep(_SLEEP)
    return _wrap


# ---------------------------------------------------------------------
class GuideState(BaseModel):
    file_id: str
    url: str
    lang: str

    chunks:   Optional[List[PageChunk]] = None
    tutorial: Optional[str]            = None
    cached:   bool                     = False
    error:    Optional[str]            = None
    log:      List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------
class GuideGraphBuilder:
    """PDF → Markdown 튜토리얼 파이프라인 (entry·translate·캐시-바이패스 버전)"""

    def __init__(
        self,
        loader: PdfLoaderIF,
        store : VectorStoreIF,
        llm   : LlmChainIF,
        cache : CacheIF,
    ):
        self.loader, self.store, self.llm, self.cache = loader, store, llm, cache
        self.embed_fn = get_embed_fn()

    # ------------------------------------------------------------------
    def build(self):
        g = StateGraph(GuideState)

        # 1) load  ────────────────────────────────────────────────────
        @retry
        async def load(st: GuideState):
            # 캐시 존재 여부만 기록(저장 단계에서 중복 저지를 위해 사용)
            st.cached = self.cache.exists_summary(st.file_id)

            # 항상 PDF를 새로 읽어 들인다 (캐시 히트여도 바이패스)
            st.chunks = await self.loader.load(st.url, with_figures=True)
            return st
        g.add_node("load", load)

        # 2) embed ────────────────────────────────────────────────────
        @retry
        async def embed(st: GuideState):
            if st.chunks and not await self.store.has_chunks(st.file_id):
                await self.store.upsert([c.text for c in st.chunks], st.file_id)
            return st
        g.add_node("embed", embed)

        # 3) generate ─────────────────────────────────────────────────
        @retry
        async def generate(st: GuideState):
            segments = segment_in_order(st.chunks or [], self.embed_fn)

            async def _make_md(pages, grp):
                # (1) 섹션용 프롬프트
                prompt = PROMPT_TUTORIAL.render(
                    chunks="\n".join(c.text for c in grp)[:6_000]  # 6k 토큰 이하로 컷
                )
                section = await self.llm.execute(prompt)
                # (2) 머리글·이미지 추가
                header = f"## Pages {pages[0]+1}–{pages[-1]+1}"
                figs   = "\n".join(
                    f"![figure]({u})\n**Tutor’s note:** "
                    for u in {u for c in grp for u in c.figs}
                )
                return f"{header}\n\n{section}\n\n{figs}"

            # ↙︎ 비동기 병렬 실행
            parts = await asyncio.gather(*(_make_md(p, g) for p, g in segments))
            st.tutorial = "\n\n".join(parts) + "\n\n---\n\n"
            return st
        g.add_node("generate", generate)

        # 4) save ─────────────────────────────────────────────────────
        async def save(st: GuideState):
            # 캐시에 없을 때만 저장
            if not st.cached and st.tutorial:
                self.cache.set_summary(st.file_id, st.tutorial)
            return st
        g.add_node("save", save)

        # 5) finish ───────────────────────────────────────────────────
        async def finish(st: GuideState):
            msg = st.error or " | ".join(st.log)
            try:
                self.cache.set_log(st.file_id, st.url, "TUTORIAL", st.lang, msg)
            except Exception:
                ...
            return st
        g.add_node("finish", finish)

        # ------------------------------------------------------------------
        # Routing
        # ------------------------------------------------------------------
        g.set_entry_point("load")

        def post_load(st: GuideState) -> str:
            return "finish" if st.error else "embed"
        g.add_conditional_edges("load", post_load, {
            "embed":  "embed",
            "finish": "finish",
        })

        g.add_edge("embed", "generate")

        def post_generate(st: GuideState) -> str:
            return "finish" if st.error else "save"
        g.add_conditional_edges("generate", post_generate, {
            "save":   "save",
            "finish": "finish",
        })

        g.add_edge("save", "finish")

        g.set_finish_point("finish")
        return g.compile()

