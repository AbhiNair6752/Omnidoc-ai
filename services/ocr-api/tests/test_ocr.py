from fastapi.testclient import TestClient

from app.main import app
from io import BytesIO

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

def test_upload_valid_pdf():
    pdf_content = b"%PDF-1.4 test pdf content"

    response = client.post(
        "/api/v1/ocr/upload",
        files = {
            "file": (
                "test.pdf",
                BytesIO(pdf_content),
                "application/pdf"
            )
        },

    )

    assert response.status_code == 200

    data = response.json()

    assert "document_id" in data
    assert data["filename"] == "test.pdf"
    assert data["content_type"] == "application/pdf"
    assert data["status"] == "uploaded"

def test_upload_valid_image():
    image_content = b"fake-image-content"

    response = client.post(
        "/api/v1/ocr/upload",
        files = {
            "file": (
                "test.jpg",
                BytesIO(image_content),
                "image/jpeg"
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "document_id" in data
    assert data["filename"] == "test.jpg"
    assert data["content_type"] == "image/jpeg"
    assert data["status"] == "uploaded"

def test_upload_invalid_extension():
    response = client.post(
        "/api/v1/ocr/upload",
        files = {
            "file": (
                "test.txt",
                BytesIO(b"hello"),
                "text/plain"
            )
        },
    )

    assert response.status_code == 415

def test_upload_invalid_mime_type():
    response = client.post(
         "/api/v1/ocr/upload",
        files={
            "file": (
                "test.pdf",
                BytesIO(b"hello"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 415

def test_upload_empty_file():
    response = client.post(
        "/api/v1/ocr/upload",
        files={
            "file": (
                "empty.pdf",
                BytesIO(b""),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400