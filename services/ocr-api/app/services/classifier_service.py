from pathlib import Path

SUPPORTED_DOCUMENT_TYPES = {
    "invoice",
    "salary_slip",
    "bank_statement",
    "identity_document",
    "receipt",
    "contract",
    "unknown",
}

class DocumentClassifier:
    def classify(self, file_path: Path) -> tuple[str, float]:
        """
        Classify a document.

        This is currently a placeholder for the
        Hugging Face vision model that we will integrate next.
        """

        return "unknown", 0.0

        