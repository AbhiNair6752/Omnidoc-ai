from app.classification.document_classifier import DocumentClassifier
from app.ocr.engine import OCREngine

classifier = DocumentClassifier()
ocr_engine = OCREngine()

def classify_document(state):
    print("LangGraph: classifying document....")

    result = classifier.predict(
        state["image_path"]
    )

    return {
        "document_type": result["document_type"],
        "classification_confidence": result["confidence"]
    }

def extract_text(state):

    print("Langgraph: extracting text...")

    extracted_text = ocr_engine.extract_text(
        state["image_path"]
    )

    return {
        "extracted_text": extracted_text
    }


def route_after_policy(state):

    decision = state.get(
        "decision",
        {}
    )

    policy_decision = decision.get(
        "decision"
    )

    print(
        f"Langgraph: policy decision=",
        f"{policy_decision}"
    )

    if policy_decision == "APPROVED":
        return "approved"

    if policy_decision == "MANUAL_REVIEW":
        return "manual_review"

    if policy_decision == "REJECTED":
        return "rejected"

    return "manual_review"