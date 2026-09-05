import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow
import mlflow.pytorch

from ml.models.cnn_model import CNN

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

IMAGE_SIZE = 224

USE_DROPOUT = True
USE_BATCH_NORM = True

TRAIN_DIR = "ml/atasets/documents/train"
VALIDATION_DIR = "ml/atasets/documents/validation"
TEST_DIR = "ml/atasets/documents/test"

MODEL_PATH = "ml/models/document_classifier.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using device: {device}")

train_transform = transforms.Compose([
    transforms.Lambda(
        lambda image: image.convert("RGB")
    )
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])

evaluation_tranform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])


train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

validation_dataset = datasets.ImageFolder(
    VALIDATION_DIR,
    transform=evaluation_tranform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=evaluation_tranform
)

print("class mapping:")
print(train_dataset.class_to_idx)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(validation_dataset)}")
print(f"Test samples: {len(test_dataset)}")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
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

NUM_CLASSES = len(train_dataset.classes)

print(f"the number of classes are: {NUM_CLASSES}")

model = CNN(
    use_dropout=USE_DROPOUT,
    use_batch_norm=USE_BATCH_NORM,
    num_classes=NUM_CLASSES
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

def evaluate_model(model, data_loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    accuracy = correct / total
    return accuracy

mlflow.set_experiment(
    "OmniDoc_Document_Classifier"
)

best_validation_accuracy = 0.0

with mlflow.start_run(
    run_name = "Baseline_Document_CNN"
):
    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "image_size": IMAGE_SIZE,
        "dropout": USE_DROPOUT,
        "batch_norm": USE_BATCH_NORM,
        "num_classes": NUM_CLASSES
    })

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        average_loss = (
            running_loss /
            len(train_loader)
        )


        validation_accuracy = evaluate_model(
            model,
            validation_loader,
            device
        )

        mlflow.log_metric(
            "validation_accuracy",
            validation_accuracy,
            step = epoch + 1
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )


        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "Best model checkpoint saved"
            )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    best_model = CNN(
        num_classes=NUM_CLASSES,
        use_dropout=USE_DROPOUT,
        use_batch_norm=USE_BATCH_NORM
    )

    best_model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    best_model.to(device)

    test_accuracy = evaluate_model(
        best_model,
        test_loader,
        device
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )


    mlflow.log_metric(
        "test_accuracy",
        test_accuracy
    )

    example_image, _ = test_dataset[0]

    example_input = (
        example_image
        .unsqueeze(0)
        .to(device)
    )

    mlflow.pytorch.log_model(
        best_model,
        name="document_classifier",
        input_example=example_input
    )

print("Document classifier training completed.")
