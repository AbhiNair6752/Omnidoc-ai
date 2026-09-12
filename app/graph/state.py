from typing import TypedDict

class DocumentState(TypedDict, total=False):

    image_path: str

    document_type: str

    classification_confidence: float

    extracted_text: str

    final_document_type: str

    structured_data: dict

    relevant_policies: list

    decision: str

    error: str