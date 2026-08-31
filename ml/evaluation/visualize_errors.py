import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.models.cnn_model import CNN


BATCH_SIZE = 64

MODEL_PATH = "ml/models/cnn_mnist_model.pth"

NUM_IMAGES = 12

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"using device:{device}")

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

misclassified_images = []

misclassified_labels = []

misclassified_predictions = []

misclassified_confidences = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = outputs.argmax(
            dim=1
        )


        for i in range(len(labels)):

            actual = labels[i].item()

            predicted = predictions[i].item()

            confidence = probabilities[
                i,
                predicted
            ].item()

            if actual != predicted:

                misclassified_images.append(
                    images[i].cpu()
                )

                misclassified_labels.append(
                    actual
                )

                misclassified_predictions.append(
                    predicted
                )

                misclassified_confidences.append(
                    confidence
                )
print(
    f"\nTotal misclassified images: "
    f"{len(misclassified_images)}"
)

num_to_display = min(
    NUM_IMAGES,
    len(misclassified_images)
)


plt.figure(
    figsize=(12, 8)
)

for i in range(num_to_display):

    image = misclassified_images[i]

    actual = misclassified_labels[i]

    predicted = misclassified_predictions[i]

    confidence = misclassified_confidences[i]


    # Undo normalization for visualization

    image = image * 0.3081 + 0.1307


    plt.subplot(
        3,
        4,
        i + 1
    )


    plt.imshow(
        image.squeeze(),
        cmap="gray"
    )
    plt.title(
        f"Actual: {actual}\n"
        f"Predicted: {predicted}\n"
        f"Confidence: {confidence * 100:.1f}%"
    )


    plt.axis("off")


plt.tight_layout()

plt.show()
