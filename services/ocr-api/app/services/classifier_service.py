from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from app.services.document_processor import get_document_image

MODEL_NAME = "microsoft/dit-base-finetuned-rvlcdip"


"""SUPPORTED_DOCUMENT_TYPES = {
    "invoice",
    "salary_slip",
    "bank_statement",
    "identity_document",
    "receipt",
    "contract",
    "unknown",
}"""


class DocumentClassifier:

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME
        )
        self.model.eval()


    def classify(self, file_path: Path) -> tuple[str, float]:
       image = get_document_image(file_path)

       inputs = self.processor(
          images=image,
          return_tensors="pt"
       )

       with torch.no_grad():
        outputs=self.model(**inputs)

       probabilities = torch.softmax(
          outputs.logits,
          dim=-1
       )
       confidence, predicted_class = torch.max(
            probabilities,
            dim=-1,
        )
       document_type = self.model.config.id2label[
            predicted_class.item()
        ]
        
       return document_type, confidence.item()

        