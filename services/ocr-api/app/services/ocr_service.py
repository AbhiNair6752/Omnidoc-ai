from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile
from app.services.file_validator import validate_upload

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def save_document(file: UploadFile) -> dict:
    contents = await validate_upload(file)
    document_id = f"doc_{uuid4().hex}"

    file_extension = Path(file.filename).suffix.lower()
    file_path = UPLOAD_DIR / f"{document_id}{file_extension}"

    file_path.write_bytes(contents)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "uploaded"
    }