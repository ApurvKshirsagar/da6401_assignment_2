"""Classification components
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead.

    The encoder outputs a 4096-dim feature vector.
    The head maps it to num_classes logits.
    """

    def __init__(
        self,
        num_classes: int   = 37,
        in_channels: int   = 3,
        dropout_p:   float = 0.5,
    ):
        """
        Args:
            num_classes: Number of output classes (37 breeds).
            in_channels: Number of input channels.
            dropout_p:   Dropout probability in encoder FC layers.
        """
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)

        # Classification head: 4096 → num_classes
        self.head = nn.Linear(4096, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 3, 224, 224] input tensor.

        Returns:
            [B, num_classes] classification logits.
        """
        features = self.encoder(x)        # [B, 4096]
        logits   = self.head(features)    # [B, num_classes]
        return logits