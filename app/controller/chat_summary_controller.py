# app/controller/chat_summary_controller.py
from fastapi import APIRouter, Depends
from app.dto.chat_summary_dto import ChatSummaryRequestDTO
from app.service.chat_summary_graph import (
    ChatSummaryGraph,
    get_chat_summary_graph,
)

router = APIRouter(prefix="/api")

@router.post("/chat-summary")
async def summarize_chat(
    req: ChatSummaryRequestDTO,
    service: ChatSummaryGraph = Depends(get_chat_summary_graph),
):
    # 타임스탬프 정렬 + 한 줄 문자열 변환
    msgs = sorted(req.chats, key=lambda c: c.timestamp)
    lines = [f"[{c.timestamp:%Y-%m-%d %H:%M:%S}] {c.sender}: {c.plaintext}"
             for c in msgs]

    return await service.generate(lines, query=req.query, lang=req.lang)

