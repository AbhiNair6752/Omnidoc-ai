import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from datasets import load_dataset
from torchvision import transforms

import mlflow
import mlflow.pytorch

from ml.models.cnn_model import CNN

BATCH_SIZE=32
LEARNING_RATE=0.001
EPOCHS=10

NUM_CLASSES=16

USE_DROPOUT=True
USE_BATCH_NORM=True
USE_AUGMENTATION=True

IMAGE_SIZE=224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if USE_AUGMENTATION:

    train_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.RandomRotation(
            degrees=10
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1,0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])

else:

    train_transform = transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,)
        )
    ])

evaluate_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,)
    )
])

dataset = load_dataset(
    "ucf-crcv/rvl-cdip"
)

full_dataset = dataset["train"]

print(
    f"Total training images: {len(full_dataset)}"
)

train_size = int(
    0.8 * len(full_dataset)
)

validation_size = (
    len(full_dataset) - train_size
)

generator = torch.Generator().manual_seed(42)

train_dataset, validation_dataset = random_split(
    full_dataset,
    [train_size, validation_size],
    generator=generator
)

print(
    f"Training samples: {len(train_dataset)}"
)

print(
    f"Validation samples: {len(validation_dataset)}"
)

def train_collate_fn(batch):

    images = []
    labels = []

    for item in batch:

        image = item["image"]
        label = item["label"]

        image = image.convert("L")

        image = train_transform(image)

        images.append(image)

        labels.append(label)

    images = torch.stack(images)

    labels = torch.tensor(labels)

    return images, labels

def validation_collate_fn(batch):

    images = []
    labels = []

    for item in batch:

        image = item["image"]
        label = item["label"]

        image = image.convert("L")

        image = evaluate_transform(image)

        images.append(image)
        labels.append(label)

    images = torch.stack(images)

    labels = torch.tensor(labels)

    return images, labels

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=train_collate_fn
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=validation_collate_fn
)

def evaluate_model(
        model,
        data_loader,
        device
):
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

mlflow.set_experiment(
    "Omnidoc_RVLCDIP_CNN"
)

with mlflow.start_run(
    run_name = "CNN_BN_Dropout_Augmentation"
):
    mlflow.log_params({
         "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "dropout": USE_DROPOUT,

        "batch_norm": USE_BATCH_NORM,

        "augmentation": USE_AUGMENTATION,

        "image_size": IMAGE_SIZE,

        "num_classes": NUM_CLASSES
    })


    best_validation_accuracy = 0.0


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
            "training_loss",
            average_loss,
            step=epoch + 1

        )

        mlflow.log_metric(
            "validation_accuracy",
            validation_accuracy,
            step= epoch + 1
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {average_loss:.4f} "
            f"Validation Accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = (
                validation_accuracy
            )

            torch.save(
                model.state_dict(),
                "ml/models/best_cnn_model.pth"
            )

            print(
                "Best model saved"
            )

    model.load_state_dict(
        torch.load(
            "ml/models/best_cnn_model.pth",
            map_location = device
        )
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy * 100:.2f}%"
    )


    example_image, _ = full_dataset[0]

    example_image = example_image.convert("L")

    example_input = evaluate_transform(
        example_image
    )

    example_input = (
        example_input
        .unsqueeze(0)
        .to(device)
    )

    mlflow.pytorch.log_model(
        model,
        name="best_rvlcdip_cnn_model",
        input_example=example_input
    )

    print(
        "Best CNN model logged to ML flow"
    )

