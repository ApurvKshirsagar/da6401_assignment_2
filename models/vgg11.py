"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from models.layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    Architecture follows the original VGG11 paper (Simonyan & Zisserman, 2014)
    with BatchNorm added after every conv layer, and CustomDropout in the
    FC layers.

    Design choices:
      - BatchNorm after every Conv2d: stabilises training, allows higher LR.
      - CustomDropout only in FC layers (p=0.5): conv layers have spatial
        redundancy so dropout there hurts more than it helps.
      - return_features=True returns skip maps before each MaxPool for U-Net.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # ── Block 1: 1 conv, 64 filters ──────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 64 × 112 × 112

        # ── Block 2: 1 conv, 128 filters ─────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 128 × 56 × 56

        # ── Block 3: 2 convs, 256 filters ────────────────────────────────
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 256 × 28 × 28

        # ── Block 4: 2 convs, 512 filters ────────────────────────────────
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 512 × 14 × 14

        # ── Block 5: 2 convs, 512 filters ────────────────────────────────
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 512 × 7 × 7

        # ── Adaptive pool: guarantees 7×7 for any input size ─────────────
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # ── FC layers (used by classifier + localizer heads) ─────────────
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # ── Encoder forward ───────────────────────────────────────────────
        x  = self.block1(x)
        s1 = self.pool1(x)           # [B,  64, 112, 112] — after pool

        x  = self.block2(s1)
        s2 = self.pool2(x)           # [B, 128,  56,  56]

        x  = self.block3(s2)
        s3 = self.pool3(x)           # [B, 256,  28,  28]

        x  = self.block4(s3)
        s4 = self.pool4(x)           # [B, 512,  14,  14]

        x  = self.block5(s4)
        s5 = self.pool5(x)           # [B, 512,   7,   7]

        bottleneck = self.adaptive_pool(s5)   # [B, 512, 7, 7]

        # ── FC layers ─────────────────────────────────────────────────────
        flat     = bottleneck.view(bottleneck.size(0), -1)
        features = self.fc(flat)                              # [B, 4096]

        if return_features:
            feature_dict = {
                "skip1":      s1,          # [B,  64, 112, 112]
                "skip2":      s2,          # [B, 128,  56,  56]
                "skip3":      s3,          # [B, 256,  28,  28]
                "skip4":      s4,          # [B, 512,  14,  14]
                "skip5":      s5,          # [B, 512,   7,   7]
                "bottleneck": bottleneck,  # [B, 512,   7,   7]
            }
            return features, feature_dict

        return features

class VGG11EncoderNoBN(nn.Module):
    """VGG11 encoder WITHOUT BatchNorm — for ablation study (Section 2.1)."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
        )

    def forward(self, x, return_features=False):
        x  = self.block1(x); s1 = self.pool1(x)
        x  = self.block2(s1); s2 = self.pool2(x)
        x  = self.block3(s2); s3 = self.pool3(x)
        x  = self.block4(s3); s4 = self.pool4(x)
        x  = self.block5(s4); s5 = self.pool5(x)
        bn = self.adaptive_pool(s5)
        flat = bn.view(bn.size(0), -1)
        features = self.fc(flat)
        if return_features:
            return features, {"skip1":s1,"skip2":s2,"skip3":s3,"skip4":s4,"skip5":s5,"bottleneck":bn}
        return features