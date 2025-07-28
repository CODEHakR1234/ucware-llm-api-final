# app/domain/page_chunk.py
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

@dataclass
class PageChunk:
    """
    PDF 텍스트 청크 + 메타
    --------------------
    • page  : 0-based page number  
    • text  : 본문 텍스트  
    • figs  : 이 청크와 함께 보여줘야 할 figure/table URL 목록
    """
    page: int
    text: str
    figs: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex)  # optional

