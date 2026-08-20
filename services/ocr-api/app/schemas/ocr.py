from pydantic import BaseModel

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    status: str

class DocumentClassificationResponse(BaseModel):
    document_id: str
    document_type: str
    confidence: float