from mcp.server.mcpserver import MCPServer

import uuid
import httpx

mcp = MCPServer(
    "Omnidoc MCP Server" 
)

ENTERPRISE_API = (
    "http://localhost:9000"
)

@mcp.tool()
def get_customer_details(
    customer_id: str
) -> dict:


    response = httpx.get(
        f"{ENTERPRISE_API}/customers/{customer_id}"
    )

    response.raise_for_status()

    return response.json()

@mcp.tool()
def create_manual_review_case(
    document_type: str,
    reason: str,
    structured_data: dict
)-> dict:

    case_id = str(
        uuid.uuid4()
    )

    return {
        "case_id": case_id,

        "status": "MANUAL_REVIEW",

        "document_type": document_type,

        "reason": reason,

        "structured_data": structured_data
    }

@mcp.tool()
def verify_pan(
    pan_number: str
) -> dict:

    response = httpx.post(
        f"{ENTERPRISE_API}/pan/verify",
        json={
            "pan_number": pan_number
        }
    )

    response.raise_for_status()

    return response.json()

@mcp.tool()
def get_account_balance(
    customer_id: str
) -> dict:

    response = httpx.get(
        f"{ENTERPRISE_API}/accounts/{customer_id}"
    )

    response.raise_for_status()

    return response.json()


@mcp.tool()
def check_loan_eligibility(
    customer_id: str
) -> dict:

    response = httpx.post(
        f"{ENTERPRISE_API}/loan/eligibility",
        params={
            "customer_id": customer_id
        }
    )

    response.raise_for_status()

    return response.json()



if __name__ == "__main__":
    mcp.run(
        transport="streamable-http"
    )