from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from ml.models.cnn_model import CNN

class DocumentClassifier:

    def __init__(self):

        self.device = torch.device("cpu" 
        )

        self.class_names = [
            "letter",
            "form",
            "email",
            "handwritten",
            "advertisement",
            "scientific report",
            "scientific publication",
            "specification",
            "file folder",
            "news article",
            "budget",
            "invoice",
            "presentation",
            "questionnaire",
            "resume",
            "memo"
        ]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = (0.5,),
                std = (0.5,)
            )
        ])

        self.model = CNN(
            use_dropout=True,
            use_batch_norm=True,
            num_classes=16
        )

        model_path = (
            Path(__file__).resolve()
            .parents[2]
            /"ml"
            /"models"
            /"best_cnn_model.pth"
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)

        self.model.eval()

        print(
            f"Dcoument classifier loaded on {self.device}"
        )

    def predict(self, image_path: str):

        image = Image.open(image_path)

        image = image.convert("L")

        image = self.transform(image)

        image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():

            outputs = self.model(image)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted_index = torch.max(
                probabilities,
                dim=1
            )
        predicted_index = predicted_index.item()

        confidence = confidence.item()

        document_type = self.class_names[
            predicted_index
        ]

        return {
            "document_type": document_type,
            "confidence": round(
                confidence,
                4
            )
        }