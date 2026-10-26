"""The thesis CNN (thesis/cnn_thesis.ipynb, Keras) ported layer for layer to PyTorch."""

import torch.nn as nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(c_out),
        nn.Dropout(0.25),
        nn.MaxPool2d(2),
    )


class ThesisCNN(nn.Module):
    def __init__(self, img_size: int = 256, in_chans: int = 1, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_chans, 32), _block(32, 64), _block(64, 128), _block(128, 256)
        )
        side = img_size // 16
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * side * side, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
