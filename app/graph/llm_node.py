import os
import json

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from app.schemas.document_schemas import InvoiceData 

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def understand_document(state):

    document_type = state.get("final_document_type",
                              state["document_type"])

    extracted_text = state["extracted_text"]

    prompt = f"""

You are an intelligent document processing system.

Document type detected:

{document_type}

OCR extracted the following text:

{extracted_text}

Your job is to understand the document and extract the
important information.

Return ONLY valid JSON.

The JSON should contain the relevant fields based on the
document type.
"""

    if document_type == "invoice":

        structured_llm = llm.with_structured_output(
            InvoiceData
        )

        result = structured_llm.invoke(prompt)

        structured_data = result.model_dump()
    else :

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

             structured_data = json.loads(response_text)

        except json.JSONDecodeError:

           structured_data = {
            "raw_response": response_text
        }

    return {
        "structured_data": structured_data
    }


def verify_document_type(state):

    document_type = state["document_type"]

    confidence = state["classification_confidence"]

    extracted_text = state["extracted_text"]

    if confidence > 0.70:

        return {
            "final_document_type": document_type
        }

    print(
        "CNN confidence is low. "
        "Verifying document type using LLM..."
    )

    prompt = f"""
You are a document classification system.

A CNN classified the document as:

{document_type}

The CNN confidence is:

{confidence}

Below is the OCR extracted text:

{extracted_text}

Determine the actual document type.

Choose the most appropriate document type.

Return ONLY the document type as plain text.
"""

    response = llm.invoke(prompt)

    final_document_type = response.content.strip()

    return {
        "final_document_type": final_document_type
    }