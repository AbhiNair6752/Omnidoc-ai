import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from ml.models.cnn_model import CNN

BATCH_SIZE = 64

MODEL_PATH = "ml/models/cnn_mnist_model.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

test_dataset = datasets.MNIST(
    root="ml/datasets",
    train=False,
    download=True,
    transform=transform
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CNN(
    use_dropout=True,
    use_batch_norm=True
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

all_predictions = []

all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)

accuracy = (
    all_predictions == all_labels
).mean()

print(
    f"\nAccuracy:"
    f"{accuracy * 100:.2f}%"
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print(
    f"Precision"
    f"{precision:.4f}"
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print(
    f"Recall:"
    f"{recall: 4f}"
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)

print(
    f"F1 score:"
    f"{f1:.4f}"
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\n Confusion Matrix:")

print(cm)


print("\nClassificationreport")

print(
    classification_report(
        all_labels,
        all_predictions
    )
)