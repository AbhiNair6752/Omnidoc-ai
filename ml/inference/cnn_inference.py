import torch
from torchvision import transforms
from PIL import Image

from ml.models.cnn_model import CNN

class CNNInference:

    def __init__(self, model_path, device=None):

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = CCN(
            use_dropout=True,
            use_batch_norm=True
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.1307,),
                (0.3081,)
            )
        ])

    def predict(self, image_path):

        image = Image.open(image_path).convert("L")

        image = self.transform(image)

        image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():

            outputs = self.model(image)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        return {
            "prediction": prediction.item(),
            "confidence": confidence.item()
        }