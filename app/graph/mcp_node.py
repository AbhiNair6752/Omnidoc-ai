import asyncio
import json
import os

from mcp import Client


MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000/mcp")


async def call_validate_invoice(
    invoice_data
):

    async with Client(
        MCP_SERVER_URL
    ) as client:

        result = await client.call_tool(
            "validate_invoice",
            {
                "customer_id":
                    invoice_data["customer_id"],

                "invoice_number":
                    invoice_data["invoice_number"],

                "invoice_date":
                    invoice_data["invoice_date"],

                "total_amount":
                    invoice_data["total_amount"]
            }
        )

        for content in result.content:

            if getattr(
                content,
                "type",
                None
            ) == "text":

                return json.loads(
                    content.text
                )

        return None


def validate_invoice_with_mcp(state):

    print(
        "LangGraph: calling MCP validate_invoice..."
    )

    structured_data = state.get(
        "structured_data",
        {}
    )

    customer = state.get(
        "customer",
        {}
    )

    customer_id = customer.get(
        "customer_id"
    )

    if not customer_id:

        return {
            "mcp_validation_result": {
                "valid": False,
                "decision": "MANUAL_REVIEW",
                "reason":
                    "Customer could not be resolved"
            }
        }

    invoice_data = {
        "customer_id":
            customer_id,

        "invoice_number":
            structured_data["invoice_number"],

        "invoice_date":
            structured_data["invoice_date"],

        "total_amount":
            structured_data["total_amount"]
    }

    result = asyncio.run(
        call_validate_invoice(
            invoice_data
        )
    )

    print(
        "MCP validation result:"
    )

    print(
        result
    )

    return {
        "mcp_validation_result": result
    }