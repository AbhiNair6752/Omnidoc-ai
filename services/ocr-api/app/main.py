from fastapi import FastAPI

app = FastAPI(
    title="OmniDoc AI",
    description="Multimodal Document Intelligence API",
    version="0.1.0"
)

@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": "omnidoc-ocr-api"
    }