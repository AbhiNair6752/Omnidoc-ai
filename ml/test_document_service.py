from app.services.document_service import DocumentService

def main():

    image_path = "sample_document.jpg"

    print("creating Document service")

    document_service = DocumentService()

    print("Processing document")

    response = document_service.process_document(
        image_path
    )

    print("\n=======Document Response=========\n")

    print(response)


if __name__=="__main__":
    main()
