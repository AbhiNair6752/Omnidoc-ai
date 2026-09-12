from app.graph.workflow import build_workflow


def main():

    print("Building LangGraph workflow....")

    workflow = build_workflow()

    initial_state = {
        "image_path": "sample_document.jpg"
    }

    print("Running workflow")

    final_state = workflow.invoke(
        initial_state
    )


    print("\n===== FINAL STATE =====\n")

    print(
        f"Document type: "
        f"{final_state['document_type']}"
    )

    print(
        f"Classification confidence: "
        f"{final_state['classification_confidence']}"
    )

    print("\n===== EXTRACTED TEXT =====\n")

    print(
        final_state["extracted_text"]
    )

    print("\n======STRUCTURED DATA======\n")

    print(
        final_state["structured_data"]
    )


if __name__ == "__main__":
    main()