# app/controller/pdf_summary_controller.py

from fastapi import APIRouter, Depends, HTTPException
from app.model.summary_dto import SummaryRequestDTO

# 새로 만든 LangGraph 서비스 래퍼
from app.service.summary_service_graph import (
    SummaryServiceGraph,
    get_summary_service_graph,  # FastAPI Depends 용 provider
)

router = APIRouter(prefix="/api")  # 필요에 따라 prefix 조정

@router.post("/summary", summary="PDF 요약 생성")
async def summarize_pdf(
    req: SummaryRequestDTO,
    service: SummaryServiceGraph = Depends(get_summary_service_graph),
):
    """
    • file_id : 문서 고유 식별자  
    • pdf_url : PDF 위치(URL)  
    • query   : 사용자가 알고 싶은 질문/키워드
    """
    try:
        result = await service.generate(
            file_id=req.file_id,
            pdf_url=str(req.pdf_url),
            query=req.query,
            lang=req.lang,
        )
    except ValueError as e:
        # Service 층에서 검증 실패 시 400 에러로 매핑
        raise HTTPException(status_code=400, detail=str(e))

    return result  # {file_id, summary, cached} 형식
