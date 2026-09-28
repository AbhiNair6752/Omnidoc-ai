from mcp.server.mcpserver import MCPServer

import uuid
import httpx
import os

mcp = MCPServer(
    "Omnidoc MCP Server" 
)

ENTERPRISE_API = os.environ.get("ENTERPRISE_API_URL", "http://localhost:9000")

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


@mcp.tool()
def verify_customer_identity(
    customer_id: str,
    pan_number: str
)-> dict:

    customer_response = httpx.get(
        f"{ENTERPRISE_API}/customers/{customer_id}"
    )

    customer_response.raise_for_status()

    customer_data = customer_response.json()


    if customer_data.get("status") == "NOT_FOUND":

        return {
            "identity_verified": False,
            "reason": "Customer not found",
            "customer": customer_data
        }

    pan_response = httpx.post(
        f"{ENTERPRISE_API}/pan/verify",
        json={
            "pan_number": pan_number
        }
    )

    pan_response.raise_for_status()

    pan_data = pan_response.json()

    if pan_data.get("status") == "VERIFIED":

        return {
            "identity_verified": True,
            "reason": "Customer and PAN verification successful",
            "customer": customer_data,
            "pan": pan_data
        }

    return {
        "identity_verified": False,
        "reason": "PAN verification failed",
        "customer": customer_data,
        "pan": pan_data
    }

@mcp.tool()
def validate_invoice(
    customer_id: str,
    invoice_number: str,
    invoice_date: str,
    total_amount: float
) -> dict:


    if not invoice_number:

        return {
            "valid": False,
            "decision": "REJECTED",
            "reason": "Invoice data is missing"
        }

    if total_amount <= 0:

        return {
            "valid": False,
            "decision": "REJECTED",
            "reason": "Invoice total must be greater than zero"
        }

    if not invoice_date:
        return {
            "valid": False,
            "decision": "REJECTED",
            "reason": "Invoice date is missing"
        }

    response = httpx.get(
        f"{ENTERPRISE_API}/customers/{customer_id}"
    )

    response.raise_for_status()

    customer = response.json()


    if customer.get("status") == "NOT_FOUND":

        return {
            "valid": False,
            "decision": "REJECTED",
            "reason": "Customer doesn't exist"
        }

    if customer.get("status") != "ACTIVE":

        return {
            "valid": False,
            "decision": "MANUAL_REVIEW",
            "reason": "Customer is not active"
        }

    return {
        "valid": True,
        "decision": "APPROVED",
        "reason": "Invoice passed enterprise validation",
        "customer": customer,
        "invoice_number": invoice_number,
        "invoice_date": invoice_date,
        "total_amount": total_amount
    }





if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=8000
   )