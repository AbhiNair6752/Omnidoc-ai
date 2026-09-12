import json

from app.rag.retriever import retrieve_relevant_chunks

def retrieve_policies(state):

    document_type = state.get(
        "final_document_type",
        state["document_type"]
    )

    structured_data = state.get(
        "structured_data",
        {}
    )

    query = f"""
Find the business policies and validation rules
that apply to this document.

Document type:

{document_type}

Extracted structured data:

{json.dumps(structured_data, indent=2)}
"""

    relevant_policies = retrieve_relevant_chunks(
        query=query,
        limit=5
    )

    return {
        "relevant_policies": relevant_policies
    }