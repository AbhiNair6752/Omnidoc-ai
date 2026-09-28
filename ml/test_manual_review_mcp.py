from app.graph.manual_review_node import create_manual_review_with_mcp

state = {
    "final_document_type": "invoice",

    "decision": {
        "decision": "MANUAL_REVIEW",
        "reason": "Invoice requires manual approval"
    },

    "structured_data": {
        "invoice_number": "O12345",
        "invoice_date": "2026-06-02",
        "total_amount": 150000
    }
}

result = create_manual_review_with_mcp(
    state
)

print(

    "\n==========MANUAL REVIEW RESULT==============\n"
)

print(result)