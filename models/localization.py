"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based object localizer.

    Encoder (VGG11) → regression head → [cx, cy, w, h] in pixel space.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.3),
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)          # [B, 4096]
        bbox     = self.head(features)      # [B, 4] in [0,1]
        bbox     = bbox * 224.0             # scale to pixel space
        return bbox