from app.classification.document_classifier import DocumentClassifier

def main():

    image_path = "sample_document.jpg"

    print("Creating document classifier")

    classifier = DocumentClassifier()

    print("classifying document...")

    result = classifier.predict(
        image_path
    )

    print("\n========Classification Result=========\n")

    print(
        f"Document type:"
        f"{result['document_type']}"
    )

    print(
        f"Confidence:"
        f"{result['confidence']}"
    )


if __name__ == "__main__":
    main()