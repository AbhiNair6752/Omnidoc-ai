import asyncio
import json

from mcp import Client

MCP_SERVER_URL = "http://localhost:8000/mcp"

async def create_review_case(
        document_type,
        reason,
        structured_data
):
    async with Client(
        MCP_SERVER_URL
    ) as client:

        result = await client.call_tool(
            "create_manual_review_case",
            {
                "document_type": document_type,
                "reason": reason,
                "structured_data": structured_data
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

def create_manual_review_with_mcp(
        state
):

    print(
        "Langgraph: creating manual review case through mcp..."
    )

    document_type = state.get(
        "final_document_type",
        state.get("document_type")
    )

    decision = state.get(
        "decision",
        {}
    )

    reason = decision.get(
        "reason",
        "Document requires manual review"
    )

    structured_data = state.get(
        "structured_data",
        {}
    )

    result = asyncio.run(
        create_review_case(
            document_type,
            reason,
            structured_data
        )
    )

    print("MCP manual review result:")

    print(result)

    return {
        "mcp_validation_result": result
    }


