from app.infra.pdf_loader   import PdfLoader
from app.infra.vector_store import VectorStore
from app.infra.llm_engine   import LlmEngine
from app.infra.cache_store  import CacheStore
from .guide_graph_builder   import GuideGraphBuilder, GuideState

_graph = GuideGraphBuilder(
    PdfLoader(), VectorStore(), LlmEngine(), CacheStore()
).build()

class GuideServiceGraph:
    def __init__(self):
        self.graph = _graph

    async def generate(self, file_id: str, pdf_url: str, lang: str):
        st: GuideState = await self.graph.ainvoke(
            GuideState(file_id=file_id, url=pdf_url, lang=lang),
            config={"recursion_limit": 80},
        )
        return {
            "file_id": file_id,
            "tutorial": st.tutorial,
            "cached": st.cached,
            "log": st.log,
            "error": st.error,
        }

_singleton = GuideServiceGraph()
def get_guide_service() -> GuideServiceGraph:
    return _singleton

