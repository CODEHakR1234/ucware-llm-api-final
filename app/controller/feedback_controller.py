# app/controller/feedback_controller.py
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from app.dto.feedback_dto import FeedbackCreate, FeedbackOut
from app.cache.cache_db import get_cache_db        # ★ 추가

router = APIRouter(prefix="/api")


@router.post(
    "/feedback",
    response_model=FeedbackOut,
    status_code=status.HTTP_201_CREATED,
    summary="PDF 요약 피드백 등록",
)
async def create_feedback(dto: FeedbackCreate) -> FeedbackOut:
    """
    • 클라이언트 피드백을 Redis에 저장하고
    • 생성된 `id`, `created_at` 반환
    """
    try:
        feedback_id = str(uuid4())
        created_at  = datetime.utcnow()

        # ─── Redis 저장 (set_log 패턴 동일) ─────────────────
        cache = get_cache_db()
        cache.add_feedback(                         # ★
            file_id=dto.file_id,
            fb_id=feedback_id,
            payload={
                "file_id": dto.file_id,
                "pdf_url": str(dto.pdf_url),
                "lang": dto.lang,
                "rating": dto.rating,
                "comment": dto.comment,
                "usage_log": dto.usage_log,
                "created_at": created_at.isoformat(),
            },
        )

        return FeedbackOut(id=feedback_id, created_at=created_at)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"피드백 저장 실패: {exc}",
        )

