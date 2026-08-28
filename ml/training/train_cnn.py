import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

from ml.models.cnn_model import CNNModel

def evaluate_model(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    return correct / total

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1)
    ),
    transforms.ToTensor()
])

evaluation_transform = transforms.ToTensor()

full_train_dataset = datasets.MNIST(
    root = "ml/datasets",
    train = True,
    download = True,
    transform = train_transform
)

validation_dataset_full = datasets.MNIST(
    root = "ml/datasets",
    train=True,
    download=True,
    transform=evaluation_transform
)

generator =  torch.Generator().manual_seed(42)

indices = torch.randperm(
    len(full_train_dataset),
    generator=generator
)

train_indices = indices[:50000]
validation_indices = indices[50000:]

train_dataset = Subset(
    full_train_dataset,
    train_indices
)

validation_dataset = Subset(
    validation_dataset_full,
    validation_indices
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples:{len(validation_dataset)}")

test_dataset = datasets.MNIST(
    root = "ml/datasets",
    train = False,
    download = True,
    transform = evaluation_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size = BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = CNNModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

mlflow.set_experiment ("Omnidoc_MNIST_CNN")

with mlflow.start_run():

    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": 0.5,
        "batch_norm": True,
        "augmentation": True
    })

    for epoch in range(EPOCHS):

       model.train()

       running_loss = 0.0

       for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

       average_loss = running_loss / len(train_loader)

       validation_accuracy = evaluate_model(
        model,
        validation_loader,
        device
       )

       mlflow.log_metric(
          "training_loss",
          average_loss,
          step = epoch + 1
       )

       mlflow.log_metric(
          "validation_accuracy",
          validation_accuracy,
          step=epoch + 1
       )

       print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f}"
        f"Validation accuracy:"
        f"{validation_accuracy * 100:.2f}%"
      )



accuracy = evaluate_model(
    model,
    test_loader,
    device
)

mlflow.log_metric(
   "test_accuracy",
   accuracy
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

torch.save(
    model.state_dict(),
    "ml/models/cnn_mnist_model.pth"
)

"""mlflow.log_artifact(
   "ml/models/cnn_mnist_model.pth"
)"""

mlflow.pytorch.log_model(
   model,
   name="mnist_cnn_model"
)


print("CNN model saved successfully.")