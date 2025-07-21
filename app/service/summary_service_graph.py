# app/service/summary_service_graph.py
from app.infra.pdf_loader import PdfLoader
from app.infra.vector_store import VectorStore
from app.infra.llm_engine import LlmEngine
from app.infra.cache_store import CacheStore          
from app.infra.web_search import WebSearch
from app.domain.interfaces import CacheIF
from .summary_graph_builder import SummaryGraphBuilder, SummaryState

# ──────────────────────────────────────────────
# create graph only once at compile time
# ──────────────────────────────────────────────
_builder_singleton = SummaryGraphBuilder(
    PdfLoader(),
    VectorStore(),
    WebSearch(),
    LlmEngine(),
    CacheStore(),
)
_compiled_graph = _builder_singleton.build()


class SummaryServiceGraph:
    """A very thin facade between the controller and the graph. (Uses a singleton graph)"""

    def __init__(self):
        self.graph = _compiled_graph   # compiled graph shared

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------
    async def generate(self, file_id: str, pdf_url: str, query: str, lang: str):
        """Run the graph and return a dict tailored to the caller."""
        result = await self.graph.ainvoke(
            SummaryState(file_id=file_id, url=pdf_url, query=query, lang=lang)
        )

        body = {
            "file_id": file_id,
            "cached": result.get("cached", False),
            "log" : result.get("log"),
        }
        if result.get("error"):
            body["error"] = result["error"]
            return body

        # `is_summary` is set by the EntryRouter in the graph
        if result.get("is_summary"):
            body["summary"] = result.get("answer")
        else:
            body["answer"] = result.get("answer")
        
        return body


# ---- FastAPI DI provider ----
_service_singleton = SummaryServiceGraph()  

def get_summary_service_graph() -> SummaryServiceGraph:
    """FastAPI Depends용 싱글턴 반환"""
    return _service_singleton
