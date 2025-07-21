from typing import List
from pydantic import BaseModel
from datetime import datetime

class ChatMessageDTO(BaseModel):
    chat_id: str
    plaintext: str
    sender: str
    timestamp: datetime

class ChatSummaryRequestDTO(BaseModel):
    chats: List[ChatMessageDTO]
    query: str 
    lang: str = "ko"
