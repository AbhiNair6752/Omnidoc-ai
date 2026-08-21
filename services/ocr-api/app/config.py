from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    max_file_size_mb: int = 10

    allowed_extensions: set[str] = {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    allowed_content_types: set[str] = {
        "application/pdf",
        "image/jpeg",
        "image/png",
        "image/webp",
    }

settings = Settings()