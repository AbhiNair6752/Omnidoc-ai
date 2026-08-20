from fastapi import FastAPI
from app.routes.ocr import router as ocr_router

app = FastAPI(
    title="OmniDoc AI",
    description="Multimodal Document Intelligence API",
    version="0.1.0"
)
app.include_router(ocr_router)



@app.get("/health")
def healthcheck():
    return {
        "status": "healthy",
        "service": "omnidoc-ocr-api"
    }