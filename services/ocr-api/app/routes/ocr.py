from fastapi import APIRouter, File, UploadFile, HTTPException

from app.schemas.ocr import (DocumentUploadResponse, DocumentClassificationResponse)
from app.services.ocr_service import save_document
from app.services.classifier_service import DocumentClassifier
from app.services.document_service import find_document

router = APIRouter(
    prefix="/api/v1/ocr",
    tags=["OCR"]
)

classifier = DocumentClassifier()

@router.post(
    "/upload",
    response_model=DocumentUploadResponse
)
async def upload_document(
    file: UploadFile = File(...)
):
    return await save_document(file)


@router.post(
    "/classify",
    response_model=DocumentClassificationResponse
)
async def classify_document(
    document_id: str
):
    file_path = find_document(document_id)

    if file_path is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    document_type, confidence = classifier.classify(file_path)

    return DocumentClassificationResponse(
        document_id=document_id,
        document_type=document_type,
        confidence=confidence
    )