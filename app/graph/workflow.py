from langgraph.graph import StateGraph, START, END

from app.graph.state import DocumentState
from app.graph.nodes import (
    classify_document,
    extract_text,
    route_after_policy
)

from app.graph.llm_node import (understand_document, verify_document_type)

from app.graph.mcp_node import validate_invoice_with_mcp

from app.graph.rag_node import retrieve_policies

from app.graph.policy_node import evaluate_policy

from app.graph.manual_review_node import create_manual_review_with_mcp

from app.graph.customer_node import resolve_customer

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
        "retrieve_policies",
        retrieve_policies
    )

    graph.add_node(
        "evaluate_policy",
        evaluate_policy
    )

    graph.add_node(
        "verify_document_type",
        verify_document_type
    )

    graph.add_node(
        "validate_invoice_mcp",
        validate_invoice_with_mcp
    )

    graph.add_node(
        "create_manual_review_mcp",
        create_manual_review_with_mcp
    )

    graph.add_node(
        "resolve_customer",
        resolve_customer
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
        "retrieve_policies"
    )

    graph.add_edge(
        "retrieve_policies",
        "evaluate_policy"
    )

    graph.add_conditional_edges(
        "evaluate_policy",
        route_after_policy,
        {
            "approved": "resolve_customer",
            "manual_review": "create_manual_review_mcp",
            "rejected": END
        }
    )

    graph.add_edge(
        "resolve_customer",
        "validate_invoice_mcp"
    )

    graph.add_edge(
        "validate_invoice_mcp",
        END
    )

    graph.add_edge(
        "create_manual_review_mcp",
        END
    )

    return graph.compile()