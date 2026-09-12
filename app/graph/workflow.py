from langgraph.graph import StateGraph, START, END

from app.graph.state import DocumentState
from app.graph.nodes import (
    classify_document,
    extract_text
)

from app.graph.llm_node import (understand_document, verify_document_type)

def build_workflow():

    graph = StateGraph(DocumentState)

    graph.add_node(
        "classify_document",
        classify_document
    )

    graph.add_node(
        "extract_text",
        extract_text
    )

    graph.add_node(
        "understand_document",
        understand_document
    )

    graph.add_node(
        "verify_document_type",
        verify_document_type
    )

    graph.add_edge(
        START,
        "classify_document"
    )

    graph.add_edge(
        "classify_document",
        "extract_text"
    )

    graph.add_edge(
        "extract_text",
        "verify_document_type"
    )

    graph.add_edge(
        "verify_document_type",
        "understand_document"
    )

    graph.add_edge(
        "understand_document",
        END
    )

    return graph.compile()