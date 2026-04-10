"""Localization modules
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder


class VGG11Localizer(nn.Module):
    """VGG11-based object localizer.

    Encoder (VGG11) → regression head → [cx, cy, w, h] in pixel space.
    """

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Args:
            in_channels: Number of input channels.
            dropout_p:   Dropout probability (passed to encoder FC layers).
        """
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Regression head: 4096 → 4 (cx, cy, w, h)
        # ReLU at output ensures positive width/height
        # No activation on cx/cy — they can be anywhere in image
        self.head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
            nn.ReLU(inplace=True),   # ensures all outputs >= 0 (pixel coords)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 3, 224, 224] input tensor.

        Returns:
            [B, 4] bounding box (cx, cy, w, h) in pixel space.
        """
        features = self.encoder(x)       # [B, 4096]
        bbox     = self.head(features)   # [B, 4]
        return bbox