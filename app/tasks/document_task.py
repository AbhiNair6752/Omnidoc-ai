from app.worker import celery_app
from app.graph.workflow import build_workflow

workflow = build_workflow()

@celery_app.task
def process_document_task(image_path: str):

    initial_state = {
        "image_path": image_path
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