# app/service/chat_summary_graph.py
from app.infra.llm_engine import LlmEngine
from .chat_graph_builder import ChatGraphBuilder, ChatState

# ── 그래프 단 한 번만 빌드
_builder_singleton = ChatGraphBuilder(llm=LlmEngine()).build()

class ChatSummaryGraph:
    """Chat LangGraph 래퍼 (싱글톤 그래프 사용)"""

    def __init__(self):
        self.graph = _builder_singleton

    async def generate(self, messages: list[str], query: str, lang: str):
        result = await self.graph.ainvoke(ChatState(messages=messages,
                                                    query=query,
                                                    lang=lang))
        body = {"log": result.get("log", [])}

        if result.get("error"):
            body["error"] = result["error"]
            return body

        if result.get("is_summary"):
            body["summary"] = result.get("summary")
        else:
            body["answer"] = result.get("answer")
        return body

# ---- FastAPI Depends 용 provider ----
_service_singleton = ChatSummaryGraph()

def get_chat_summary_graph() -> ChatSummaryGraph:
    return _service_singleton

