from fastapi import APIRouter, File, UploadFile

from app.schemas.ocr import DocumentUploadResponse
from app.services.ocr_service import save_document

router = APIRouter(
    prefix="/api/v1/ocr",
    tags=["OCR"]
)

@router.post(
    "/upload",
    response_model=DocumentUploadResponse
)
async def upload_document(
    file: UploadFile = File(...)
):
    return await save_document(file)