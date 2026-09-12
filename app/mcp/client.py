import asyncio

from mcp import Client

async def main():

    async with Client(
        "http://localhost:8000/mcp"
    ) as client:

        print("\n===== MCP SERVER INFO =====\n")

        print(
            "Server:",
            client.server_info
        )

        print(
            "Protocol:",
            client.protocol_version
        )

        print(
            "\n===== DISCOVERED TOOLS =====\n"
        )

        result = await client.list_tools()

        for tool in result.tools:

            print(
                f"Tool name: {tool.name}"
            )

            print(
                f"Description: {tool.description}"
            )

            print(
                f"Input schema: {tool.input_schema}"
            )

            print(
                "-----------------------------"
            )

        print(
            "\n=====Calling mcp tool====\n"
        )

        tool_result = await client.call_tool(
            "get_customer_details",
            {
              "customer_id": "C001"
            }
        )

        print(
            "\n======Tool Result======\n"
        )

        print(
            tool_result
        )


if __name__ == "__main__":
    asyncio.run(main())