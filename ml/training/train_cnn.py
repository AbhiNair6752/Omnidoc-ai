import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
import numpy as np
from mlflow.types import Schema, TensorSpec
from mlflow.models import ModelSignature

from ml.models.cnn_model import CNN

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

USE_DROPOUT = True
USE_BATCH_NORM = True
USE_AUGMENTATION = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if USE_AUGMENTATION:
     transform = transforms.Compose([
     transforms.RandomRotation(10),
     transforms.RandomAffine(
        degrees=0,
        translate=(0.1,0.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

else:

   transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
   ])

evaluation_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

full_train_dataset = datasets.MNIST(
    root = "ml/datasets",
    train = True,
    download = True,
    transform = transform
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

model = CNN(
   use_dropout=USE_DROPOUT,
   use_batch_norm = USE_BATCH_NORM
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

best_validation_accuracy = 0.0
best_model_state = None



mlflow.set_experiment ("Omnidoc_MNIST_CNN")

with mlflow.start_run(
   run_name = "CNN_BN_Dropout_No_Augmentation"
):

    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": 0.5,
        "batch_norm": USE_BATCH_NORM,
        "augmentation": USE_AUGMENTATION
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

       if validation_accuracy > best_validation_accuracy:

           best_validation_accuracy = validation_accuracy

           best_model_state = {
               key: value.cpu().clone()
               for key, value in model.state_dict().items()
           }

           print(
               f"New best model found!"
               f"validation accuracy:"
               f"{best_validation_accuracy * 100: .2f}%"
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

    model.load_state_dict(
        best_model_state
    )

    model = model.to(device)

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

    example_image, _ =  test_dataset[0]

    example_input = (
        example_image
        .unsqueeze(0)
        .to(device)
    )

    signature = ModelSignature(
        inputs=Schema([
            TensorSpec(
                np.dtype("float32"),
                (-1, 1, 28, 28)
            )
        ]),
        outputs=Schema([
            TensorSpec(
                np.dtype("float32"),
                (-1, 10)
            )
        ])
    )

    mlflow.pytorch.log_model(
      model,
      name="mnist_cnn_model",
      input_example=example_input,
      signature=signature
    )


    print("CNN model logged to MLflow successfully.")



