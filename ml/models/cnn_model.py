import torch
from torch import nn

class CNN(nn.Module):

    def __init__(
        self,
        use_dropout=True,
        use_batch_norm=True,
        num_classes=10
    ):

        super().__init__()

        layers = []

        # ----------------------------------------------------
        # First Convolution Block
        # ----------------------------------------------------

        layers.append(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            )
        )

        if use_batch_norm:

            layers.append(
                nn.BatchNorm2d(32)
            )

        layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2))

        # ----------------------------------------------------
        # Second Convolution Block
        # ----------------------------------------------------

        layers.append(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            )
        )

        if use_batch_norm:

            layers.append(
                nn.BatchNorm2d(64)
            )

        layers.append(nn.ReLU())

        layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            )
        )

        if use_batch_norm:
            layers.append(
                nn.BatchNorm2d(128)
            )

        layers.append(nn.ReLU())

        layers.append(
            nn.MaxPool2d(kernel_size=2)
        )

        layers.append(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1
            )
        )

        if use_batch_norm:
            layers.append(
                nn.BatchNorm2d(128)
            )
        layers.append(nn.ReLU())

        layers.append(
            nn.MaxPool2d(kernel_size=2)
        )

        self.features = nn.Sequential(*layers)


        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (7,7)
        )

        # ----------------------------------------------------
        # Fully Connected Layers
        # ----------------------------------------------------

        classifier_layers = [

            nn.Flatten(),

            nn.Linear(
                128 * 7 * 7,
                128
            ),

            nn.ReLU()
        ]

        if use_dropout:

            classifier_layers.append(
                nn.Dropout(p=0.5)
            )

        classifier_layers.append(
            nn.Linear(128, num_classes)
        )

        self.classifier = nn.Sequential(
            *classifier_layers
        )


    def forward(self, x):

        x = self.features(x)

        x = self.adaptive_pool(x)

        x = self.classifier(x)

        return x