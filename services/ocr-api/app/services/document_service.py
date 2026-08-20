from pathlib import Path

UPLOAD_DIR = Path("uploads")

def find_document(document_id: str) -> Path | None:
    matches = list(UPLOAD_DIR.glob(f"{document_id}.*"))

    if not matches:
        return None

    return matches[0]