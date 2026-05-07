"""
CNN architectures -- defined at MODULE LEVEL so torch.save/load works correctly.
Import from here in any step that needs to build or load a CNN.
"""

import torch.nn as nn


class CNNBaseline(nn.Module):
    """
    Original 3-layer CNN.
    Lecture 7: Conv(3x3) -> ReLU -> MaxPool -> Dropout, repeated 3 times.
    Input:  (N, 1, 28, 28)
    Output: (N, 10)
    Spatial: 28 -> 14 -> 7 -> 3  =>  128 x 3 x 3 = 1152 flattened
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1,   32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32,  64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.40),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 128 x 3 x 3 = 1152
            nn.Linear(1152, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNNoDropout(nn.Module):
    """
    Same improved architecture WITHOUT dropout.
    Used in Step 07 to demonstrate overfitting (Lecture 6).
    Input: (N, 1, 28, 28)   Output: (N, 10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 64 x 7 x 7 = 3136
            nn.Linear(3136, 256), nn.ReLU(),
            nn.Linear(256,  128), nn.ReLU(),
            nn.Linear(128,  10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNImproved(nn.Module):
    """
    Improved CNN with BatchNorm + Dropout + L2 (via optimizer weight_decay).
    Lecture 6: regularisation techniques applied.
    Canonical order: Conv -> BN -> ReLU -> Pool (BN before activation, before pooling).
    Input: (N, 1, 28, 28)   Output: (N, 10)
    """
    def __init__(self, dropout_conv=0.25, dropout_dense=0.4):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14  (Conv -> BN -> ReLU is the canonical order)
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_conv),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_conv),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                       # 64 x 7 x 7 = 3136
            nn.Linear(3136, 256), nn.ReLU(), nn.Dropout(dropout_dense),
            nn.Linear(256,  128), nn.ReLU(), nn.Dropout(dropout_dense * 0.75),
            nn.Linear(128,  10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
