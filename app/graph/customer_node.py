import asyncio
import json

from mcp import Client

MCP_SERVER_URL= "http://localhost:8000/mcp"

async def resolve_customer_with_mcp(
        customer_name
):
    async with Client(
        MCP_SERVER_URL
    ) as client:

        result = await client.call_tool(
            "resolve_customer",
            {
                "customer_name": customer_name
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

def resolve_customer(state):

     print(
        "LangGraph: resolving customer through MCP..."
     )

     structured_data = state.get(
        "structured_data",
        {}
     )

     customer_name = structured_data.get(
        "customer_name"
     )

     if not customer_name:

        return {
            "error": "Customer name is missing"
        }

     customer = asyncio.run(
        resolve_customer_with_mcp(
            customer_name
        )
     )

     print(
        "Resolved customer:"
     )

     print(
        customer
     )

     return {
        "customer": customer
     }