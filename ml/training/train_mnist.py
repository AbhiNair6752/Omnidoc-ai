import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ml.models.mnist_model import MNISTModel


BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root = "ml/datasets",
    train = True,
    download = True,
    transform = transform
)

sample_image, sample_label = train_dataset[0]

print("Sample image shape:", sample_image.shape)
print("Sample label:", sample_label)
print("Sample label type:", type(sample_label))

test_dataset = datasets.MNIST(
    root="ml/datasets",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = MNISTModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

for epoch in range(EPOCHS):
    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        print("Before device:")
        print("Images:", images.shape)
        print("Labels:", labels.shape)


        images = images.to(device)
        labels = labels.to(device)

        print("After device:")
        print("Images:", images.shape)
        print("Labels:", labels.shape)

        optimizer.zero_grad()

        outputs = model(images)
        print("Images shape:", images.shape)
        print("Outputs shape:", outputs.shape)
        print("Labels shape:", labels.shape)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
    )

def evaluate_model(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels .to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (predictions == labels).sum().item()

    accuracy =correct / total

    return accuracy


accuracy = evaluate_model(
    model,
    test_loader,
    device
)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

torch.save(
    model.state_dict(),
    "ml/models/mnist_model.pth",
)

print("Model saved succesfully")