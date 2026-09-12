from fastapi import FastAPI, UploadFile, File,  HTTPException
from app.services.document_service import DocumentService
import os
import shutil
import uuid
from app.graph.workflow import build_workflow
from app.tasks.document_task import process_document_task

app = FastAPI(
    title="OmniDoc AI",
    description="AI powered document Intelligence API",
    version="1.0.0"
)

document_service = DocumentService()

workflow = build_workflow()

UPLOAD_DIRECTORY = "uploads"

jobs = {}

os.makedirs(
    UPLOAD_DIRECTORY,
    exist_ok=True
)

def process_document_background(
        job_id: str,
        file_path: str
):

    try:

        jobs[job_id]["status"] = "processing"

        initial_state = {
            "image_path": file_path
        }

        final_state = workflow.invoke(
            initial_state
        )

        jobs[job_id]["status"] = "completed"

        jobs[job_id]["result"] = {

            "document_type":
                final_state.get("document_type"),

            "final_document_type":
                final_state.get("final_document_type"),

            "classification_confidence":
                final_state.get(
                    "classification_confidence"
                ),

            "extracted_text":
                final_state.get("extracted_text"),

            "structured_data":
                final_state.get("structured_data")
        }
    except Exception as error:
         jobs[job_id]["status"] = "failed"

         jobs[job_id]["error"] = str(error)

@app.get("/")
def health_check():
    return {
        "status": "success",
        "message": "Omnidoc is running"
    }

@app.post("/document/process")
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    file_extension = os.path.splitext(
        file.filename
    )[1]

    unique_filename = (
        f"{uuid.uuid4()}{file_extension}"
    )

    file_path = os.path.join(
        UPLOAD_DIRECTORY,
        unique_filename
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    task = process_document_background.delay(
        file_path
    )
    """job_id = str(uuid.uuid4())

    jobs[job_id] = {
        "status": "queued",

        "result": None,

        "error": None
    }

    background_tasks.add_task(
        process_document_background,
        job_id,
        file_path
    )"""

    return {

        "job_id": task.id,

        "status": "queued",

        "message": "Document uploaded successfully and process has started"
    }


    """initial_state = {

        "image_path": file_path

    }

    final_state = workflow.invoke(
        initial_state
    )

    return {

        "document_type":
            final_state.get("document_type"),

        "classification_confidence":
            final_state.get(
                "classification_confidence"
            ),

        "extracted_text":
            final_state.get("extracted_text"),

        "structured_data":
            final_state.get("structured_data")

    }"""

@app.get("/document/{job_id}")
def get_document_status(
    job_id: str
):

    task = process_document_background.AsyncResult(
        job_id
    )

    response = {
        "job_id": job_id,

        "status": task.status
    }

    if task.successful(): 
        response["result"] = task.result 
    elif task.failed(): 
        response["error"] = str( task.result ) 

    return response