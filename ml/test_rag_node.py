from app.graph.rag_node import retrieve_policies

def main():

    test_state = {
        "document_type": "file folder",

        "final_document_type": "invoice",

        "structured_data": {

            "invoiceNumber": "O12345",

            "date": "06/02/2026",

            "invoiceFrom": "ATLAS ARCHITEC",

            "customer": {
                "name": "JOHN PETERSON"
            },

            "subtotal": 282,

            "total": 282
        }
    }

    result = retrieve_policies(
        test_state
    )

    print("\n=====RELEVANT POLICIES======\n")

    for policy in result["relevant_policies"]:

        print(
            policy["text"]
        )

        print(
            f"score: {policy["score"]}"
        )

        print(
            "---------------------"
        )



if __name__=="__main__":
    main()