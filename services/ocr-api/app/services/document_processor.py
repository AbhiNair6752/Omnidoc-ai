from pathlib import Path

import fitz
from PIL import Image

SUPPORTED_IMAGE_TYPES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

def get_document_image(file_path: Path) -> Image.Image:

    extension = file_path.suffix.lower()

    if extension in SUPPORTED_IMAGE_TYPES:
        return Image.open(file_path).convert("RGB")

    if extension == ".pdf":
        return _pdf_to_image(file_path)

    raise ValueError(
        f"Unsupported document format: {extension}"
    )

def _pdf_to_image(file_path: Path) -> Image.Image:

    document = fitz.open(file_path)

    if len(document) == 0:
        raise ValueError("PDF contains no pages")

    page = document[0]

    matrix = fitz.Matrix(2,2)

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image = Image.frombytes(
        "RGB",
        [pixmap.width, pixmap.height],
        pixmap.samples,
    )

    document.close()

    return image