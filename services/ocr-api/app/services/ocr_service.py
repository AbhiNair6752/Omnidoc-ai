from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

async def save_document(file: UploadFile) -> dict:
    document_id = f"doc_{uuid4().hex}"

    file_extension = Path(file.filename).suffix.lower()
    file_path = UPLOAD_DIR / f"{document_id}{file_extension}"

    contents = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(contents)

    return {
        "document_id": document_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "uploaded"
    }