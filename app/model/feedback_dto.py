from datetime import datetime
from typing import List, Optional, Literal

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    constr,
    conint,
    conlist,
    field_validator,   # v2 전용
)

# ────────────────────── 공통 타입 ──────────────────────
Str1k = constr(max_length=1_000)          # 1,000자 이하 문자열

# ─────────────── 요청 DTO ───────────────
class FeedbackCreate(BaseModel):
    file_id: constr(
        pattern=r"^fid_[0-9a-f]+_",
        min_length=3,
        max_length=100,
    )
    pdf_url: HttpUrl
    lang: Literal["KO", "EN"]
    rating: conint(ge=1, le=5)
    comment: Optional[Str1k] = None

    usage_log: conlist(
        item_type=Str1k,                  # 각 항목 ≤ 1 000자
        min_length=0,
        max_length=10,                    # 항목 수 0–10개
    ) = Field(default_factory=list, description="follow-up 로그 (최신순)")

    # 길이 초과 항목을 1 000자까지만 잘라서 수용
    @field_validator("usage_log", mode="before")
    @classmethod
    def truncate_items(cls, v: List[str]) -> List[str]:
        return [s[:1_000] for s in v]

# ─────────────── 응답 DTO ───────────────
class FeedbackOut(BaseModel):
    id: str = Field(..., description="DB ObjectId 또는 UUID")
    created_at: datetime = Field(default_factory=datetime.utcnow, frozen=True)
    ok: Literal[True] = True

    class Config:
        frozen = True        # 모델 전체 read-only

