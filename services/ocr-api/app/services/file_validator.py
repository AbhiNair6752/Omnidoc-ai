from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.config import settings

async def validate_upload(file: UploadFile) -> bytes:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    extension = Path(file.filename).suffix.lower()

    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file extension: {extension}"
        )

    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type {file.content_type}"
        )

    contents = await file.read()

    max_size = settings.max_file_size_mb * 1024 * 1024

    if len(contents) > max_size:
        raise HTTPException(
            status_code=413,
            detail = f"File exceeds {settings.max_file_size_mb} MB limit"
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail = "Uploaded file is empty"
        )
    return contents