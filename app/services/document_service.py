from app.ocr.engine import OCREngine
from app.classification.document_classifier import DocumentClassifier

class DocumentService:

    def __init__(self):

        self.ocr_engine = OCREngine()

        self.document_classifier = DocumentClassifier()

    def process_document(self, image_path: str):
        print("Starting document processing")

        classification = self.document_classifier.predict(
            image_path
        )

        extracted_text = self.ocr_engine.extract_text(
            image_path
        )

        response = {
            "status" : "success",
            "document_path": image_path,
            "document_type": classification["document_type"],
            "classification_confidence": classification["confidence"],
            "extracted_text": extracted_text
        }

        return response