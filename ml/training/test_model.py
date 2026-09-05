import torch

from ml.models.cnn_model import CNN

model = CNN(
    use_dropout=True,
    use_batch_norm=True,
    num_classes=5
)

x = torch.randn(
    4,
    3,
    224,
    224
)

output = model(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)