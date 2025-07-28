from fastapi import APIRouter, Depends, HTTPException
from app.model.summary_dto import SummaryRequestDTO
from app.service.guide_service_graph import get_guide_service, GuideServiceGraph

router = APIRouter(prefix="/api")

@router.post("/tutorial", summary="멀티모달 PDF 자습서 생성")
async def build_tutorial(
    req: SummaryRequestDTO,
    svc: GuideServiceGraph = Depends(get_guide_service),
):
    try:
        return await svc.generate(req.file_id, str(req.pdf_url), req.lang)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

