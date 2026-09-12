from app.graph.rag_node import retrieve_policies
from app.graph.policy_node import evaluate_policy

def main():

    test_state = {
        "document_type": "file folder",

        "final_document_type": "invoice",

        "structured_data": {

            "date": "06/02/2026",

            "invoiceFrom": "ATLAS ARCHITEC",

            "customer": {
                "name": "JOHN PETERSON"
            },

            "subtotal": 282,

            "total": 282
        }
    }

    rag_result = retrieve_policies(
        test_state
    )

    test_state.update(
        rag_result
    )

    print(
        "\n=========RETRIEVED POLICIES======\n"
    )

    for policy in test_state[
        "relevant_policies"
    ]:

        print(
            policy["text"]
        )

        print(
            f"score: {policy['score']}"
        )

        print(
            "------------------"
        )

    decision_result = evaluate_policy(
        test_state
    )

    print(
        "\n=========POLICY DECISION=========\n"
    )

    print(
        decision_result["decision"]
    )


if __name__ == "__main__":

    main()