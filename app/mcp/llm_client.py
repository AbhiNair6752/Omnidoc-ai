import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp import Client


load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

def validate_tool_call(
        tool_name,
        arguments,
        available_tools
):

    if tool_name not in available_tools:
        return (
            False,
            f"Invalid tool name: {tool_name}"
        )

    selected_tool = available_tools[
        tool_name
    ]

    input_schema = selected_tool.input_schema

    required_arguments = input_schema.get(
        "required",
        []
    )

    provided_argument = set(
        arguments.keys()
    )

    missing_arguments = [
        argument
        for argument in required_arguments
        if argument not in provided_argument
    ]

    if missing_arguments:

        return (
            False,
            f"Missing required arguments:"
            f"{missing_arguments}"
        ) 

    allowed_arguments = set(

        input_schema.get(
            "properties",
            {}
        ).keys()
    )

    invalid_arguments = (

        provided_argument 

        -

        allowed_arguments
    )

    if invalid_arguments:

        return (
            False,
            f"Invalid arguments:"
            f"{invalid_arguments}"
            f"Allowed arguments:"
            f"{allowed_arguments}"
        )

    return (
        True,
        None
    )

def extract_tool_result(
    tool_response
):

    # --------------------------------------------------------
    # MCP reported an error
    # --------------------------------------------------------

    if tool_response.is_error:

        return {
            "error": True,
            "message": "MCP tool execution failed"
        }

    # --------------------------------------------------------
    # MCP returned content
    # --------------------------------------------------------

    for content in tool_response.content:

        if getattr(
            content,
            "type",
            None
        ) == "text":

            try:

                return json.loads(
                    content.text
                )

            except json.JSONDecodeError:

                return content.text

    # --------------------------------------------------------
    # Structured content fallback
    # --------------------------------------------------------

    if tool_response.structured_content:

        return tool_response.structured_content

    return None


async def main():

    # ========================================================
    # CONNECT TO MCP SERVER
    # ========================================================

    async with Client(
        "http://localhost:8000/mcp"
    ) as mcp_client:

        # ====================================================
        # DISCOVER MCP TOOLS
        # ====================================================

        tool_result = await mcp_client.list_tools()

        print(
            "\n===== MCP TOOLS =====\n"
        )

        for tool in tool_result.tools:

            print(
                f"Tool: {tool.name}"
            )

            print(
                f"Description: "
                f"{tool.description}"
            )

            print(
                f"Schema: "
                f"{tool.input_schema}"
            )

            print(
                "----------------------"
            )

        # ====================================================
        # CREATE TOOL LOOKUP
        # ====================================================

        available_tools = {
            tool.name: tool
            for tool in tool_result.tools
        }

        # ====================================================
        # USER REQUEST
        # ====================================================

        user_request = """
Verify PAN ABCDE1234Y for customer C001.
If the PAN is valid, check whether customer C001
is eligible for a loan.
"""

        # ====================================================
        # STORE PREVIOUS TOOL RESULTS
        # ====================================================

        tool_results = []

        # ====================================================
        # MAXIMUM AGENT STEPS
        # ====================================================

        MAX_STEPS = 5

        # ====================================================
        # AGENT LOOP
        # ====================================================

        for step in range(MAX_STEPS):

            print(
                f"\n===== AGENT STEP "
                f"{step + 1} =====\n"
            )

            # ------------------------------------------------
            # LLM PROMPT
            # ------------------------------------------------

            prompt = f"""
You are an AI assistant that can use
enterprise tools through MCP.

Available tools:

{json.dumps(
    [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema
        }
        for tool in tool_result.tools
    ],
    indent=2
)}

User request:

{user_request}

Previous tool results:

{json.dumps(
    tool_results,
    indent=2
)}

Your job is to determine the NEXT action
required to answer the user's request completely.

If another enterprise tool is required,
return:

{{
    "action": "tool",
    "tool": "tool_name",
    "arguments": {{}}
}}

If the user's request has been completely
answered using the available information,
return:

{{
    "action": "final"
}}

Rules:

- Use an available MCP tool when required.
- Do not invent information.
- Do not call a tool unnecessarily.
- Use previous tool results when deciding
  what information is still missing.
- Respect dependencies between actions.
- If one action must succeed before another
  action can be performed, wait for the first
  tool result.
- If a required prerequisite fails, do not
  execute dependent actions.
- Return ONLY valid JSON.
"""

            # ------------------------------------------------
            # ASK LLM
            # ------------------------------------------------

            response = llm.invoke(
                prompt
            )

            response_text = (
                response.content.strip()
            )

            print(
                "\n===== LLM DECISION =====\n"
            )

            print(
                response_text
            )

            # ------------------------------------------------
            # REMOVE MARKDOWN JSON FENCE
            # ------------------------------------------------

            if response_text.startswith(
                "```json"
            ):

                response_text = (
                    response_text
                    .replace(
                        "```json",
                        "",
                        1
                    )
                )

            if response_text.endswith(
                "```"
            ):

                response_text = (
                    response_text[:-3]
                )

            response_text = (
                response_text.strip()
            )

            # ------------------------------------------------
            # PARSE LLM JSON
            # ------------------------------------------------

            try:

                decision = json.loads(
                    response_text
                )

            except json.JSONDecodeError:

                print(
                    "\n===== INVALID LLM RESPONSE =====\n"
                )

                print(
                    "The LLM did not return valid JSON."
                )

                break

            # =================================================
            # FINAL ACTION
            # =================================================

            if decision.get("action") == "final":

                print(
                    "\n===== AGENT DECISION =====\n"
                )

                print(
                    "No more tools are required."
                )

                break

            # =================================================
            # TOOL ACTION
            # =================================================

            if decision.get("action") != "tool":

                print(
                    "\n===== INVALID ACTION =====\n"
                )

                print(
                    decision
                )

                break

            # ------------------------------------------------
            # Extract tool information
            # ------------------------------------------------

            tool_name = decision.get(
                "tool"
            )

            arguments = decision.get(
                "arguments",
                {}
            )

            print(
                "\n===== TOOL REQUEST =====\n"
            )

            print(
                f"Tool: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # =================================================
            # VALIDATE TOOL CALL
            # =================================================

            is_valid, validation_error = (
                validate_tool_call(
                    tool_name,
                    arguments,
                    available_tools
                )
            )

            # ------------------------------------------------
            # Invalid tool call
            # ------------------------------------------------

            if not is_valid:

                print(
                    "\n===== VALIDATION FAILED =====\n"
                )

                print(
                    validation_error
                )

                # ------------------------------------------------
                # Give the LLM another chance in the next
                # iteration by storing the validation error.
                # ------------------------------------------------

                tool_results.append({
                    "type": "validation_error",
                    "tool": tool_name,
                    "arguments": arguments,
                    "error": validation_error
                })

                continue

            # =================================================
            # VALID TOOL CALL
            # =================================================

            print(
                "\n===== VALIDATION SUCCESSFUL =====\n"
            )

            print(
                f"Tool: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # =================================================
            # CALL MCP TOOL
            # =================================================

            tool_response = (
                await mcp_client.call_tool(
                    tool_name,
                    arguments
                )
            )

            print(
                "\n===== MCP RESULT =====\n"
            )

            print(
                tool_response
            )

            # =================================================
            # EXTRACT ACTUAL RESULT
            # =================================================

            tool_data = extract_tool_result(
                tool_response
            )

            print(
                "\n===== EXTRACTED TOOL DATA =====\n"
            )

            print(
                json.dumps(
                    tool_data,
                    indent=2
                )
            )

            # =================================================
            # STORE RESULT
            # =================================================

            tool_results.append({

                "tool": tool_name,

                "arguments": arguments,

                "result": tool_data

            })

        # ====================================================
        # FINAL LLM RESPONSE
        # ====================================================

        final_prompt = f"""
You are an AI assistant.

The user asked:

{user_request}

The following enterprise tools were used:

{json.dumps(
    tool_results,
    indent=2
)}

Provide the final answer to the user.

Rules:

- Use ONLY the information contained
  in the tool results.
- Do not invent information.
- Clearly answer every part of the
  user's request.
- If the information is unavailable,
  explicitly say so.
- Keep the answer concise and clear.
"""

        final_response = llm.invoke(
            final_prompt
        )

        print(
            "\n========= FINAL LLM RESPONSE =========\n"
        )

        print(
            final_response.content
        )



if __name__ == "__main__":

    asyncio.run(main())