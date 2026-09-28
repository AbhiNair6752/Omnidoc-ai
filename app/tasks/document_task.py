import os

from app.worker import celery_app
from app.graph.workflow import build_workflow
from app.storage import download_file

workflow = build_workflow()

@celery_app.task
def process_document_task(s3_key: str):

    local_path = os.path.join(
        "/tmp",
        os.path.basename(s3_key)
    )

    download_file(s3_key, local_path)

    try:
        initial_state = {
            "image_path": local_path
        }

        final_state = workflow.invoke(
            initial_state
        )

        return {
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

    finally:
        if os.path.exists(local_path):
            os.remove(local_path)