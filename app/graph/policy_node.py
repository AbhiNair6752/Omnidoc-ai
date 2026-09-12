import os
import json

from dotenv import load_dotenv

from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

def evaluate_policy(state):

    structured_data = state.get(
        "structured_data",
        {}
    )

    relevant_policies = state.get(
        "relevant_policies",
        []
    )

    policy_text = "\n\n".join(
        policy["text"]
        for policy in relevant_policies
    )

    prompt = f"""
You are a document compliance decision engine.

Your job is to evaluate the extracted document
data against the company policies.

DOCUMENT DATA:

{json.dumps(structured_data, indent=2)}

COMPANY POLICIES:

{policy_text}

Evaluate the document carefully.

Decision rules:

APPROVED:
All mandatory information is available and
no manual approval rule is triggered.

MANUAL_REVIEW:
Critical information is missing or the invoice
requires manual approval.

REJECTED:
The document contains invalid or inconsistent
mandatory information.

IMPORTANT:
- Do not invent missing information.
- Use only the extracted document data.
- Follow the company policies provided above.

Return ONLY valid JSON in this format:

{{
    "decision": "APPROVED",
    "reason": "Brief explanation"
}}"""

    response = llm.invoke(prompt)

    response_text = response.content.strip()

    if response_text.startswith("```json"):

        response_text = response_text.replace(
            "```json",
            "",
            1
        )

    if response_text.endswith("```"):

        response_text = response_text[:-3]

    response_text = response_text.strip()

    try:

        decision = json.loads(
            response_text
        )
    except json.JSONDecodeError:

        decision = {
            "decision": "MANUAL_REVIEW",
            "reason": (
                "Unable to parse the policy "
                "decision from the LLM."
            )
        }
    return {
        "decision": decision
    }