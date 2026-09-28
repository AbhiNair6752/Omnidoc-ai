from app.graph.mcp_node import validate_invoice_with_mcp

state = {
    "structured_data": {
        "customer_id": "C001",
        "invoice_number": "O12345",
        "invoice_date": "2026-06-02",
        "total_amount": 282.00
    }
}

result = validate_invoice_with_mcp(
    state
)

print("\n=========LANGGRAPH MCP RESULT===========\n")

print(result)