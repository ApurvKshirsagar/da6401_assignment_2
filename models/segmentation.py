"""Segmentation model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder


class DecoderBlock(nn.Module):
    """One decoder stage: ConvTranspose2d upsample + conv refinement.

    Takes upsampled features, concatenates skip connection,
    then refines with two conv layers.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        Args:
            in_channels:   channels coming from the previous decoder stage.
            skip_channels: channels from the encoder skip connection.
            out_channels:  channels to output.
        """
        super().__init__()

        # ConvTranspose2d doubles spatial resolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )

        # After concat: (in_channels//2 + skip_channels) → out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2 + skip_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    feature map from previous decoder stage [B, C, H, W]
            skip: skip connection from encoder           [B, C', 2H, 2W]
        """
        x = self.upsample(x)

        # Handle odd spatial sizes (pad if needed)
        if x.shape != skip.shape:
            x = nn.functional.pad(
                x,
                [0, skip.shape[3] - x.shape[3],
                 0, skip.shape[2] - x.shape[2]]
            )

        x = torch.cat([x, skip], dim=1)   # concat along channel dim
        x = self.conv(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation network using VGG11 as encoder.

    Architecture:
      Encoder : VGG11 blocks 1-5 (pretrained weights can be loaded)
      Decoder : 5 symmetric decoder blocks with ConvTranspose2d upsampling
      Head    : 1×1 conv → num_classes logits

    Skip connections: encoder block outputs are concatenated with
    upsampled decoder features at each stage.

    Loss choice: CrossEntropyLoss + DiceLoss combined.
    CrossEntropy handles per-pixel classification well.
    Dice handles class imbalance (foreground << background pixels).
    """

    def __init__(
        self,
        num_classes: int   = 3,
        in_channels: int   = 3,
        dropout_p:   float = 0.5,
    ):
        """
        Args:
            num_classes: Number of segmentation classes (3: fg/bg/border).
            in_channels: Number of input channels.
            dropout_p:   Dropout probability (used in encoder FC — not active
                         for segmentation since we skip the FC layers).
        """
        super().__init__()

        # ── Encoder (VGG11 conv blocks only — no FC layers used) ──────────
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ── Decoder blocks ─────────────────────────────────────────────────
        # Each DecoderBlock(in_ch, skip_ch, out_ch):
        #   in_ch   = channels from previous stage
        #   skip_ch = channels from corresponding encoder skip
        #   out_ch  = output channels

        # bottleneck: 512×7×7
        # dec4: upsample to 14×14, concat skip4(512) → 512
        self.dec4 = DecoderBlock(512, 512, 512)

        # dec3: upsample to 28×28, concat skip3(256) → 256
        self.dec3 = DecoderBlock(512, 256, 256)

        # dec2: upsample to 56×56, concat skip2(128) → 128
        self.dec2 = DecoderBlock(256, 128, 128)

        # dec1: upsample to 112×112, concat skip1(64) → 64
        self.dec1 = DecoderBlock(128, 64, 64)

        # dec0: upsample to 224×224, no skip (encoder input level)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Output head: 1×1 conv → num_classes ───────────────────────────
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 3, 224, 224] input tensor.

        Returns:
            [B, num_classes, 224, 224] segmentation logits.
        """
        # ── Encoder — get skip maps ───────────────────────────────────────
        # We call encoder with return_features=True to get skip connections
        # We do NOT use the FC layers — only the conv feature maps
        _, feat = self.encoder(x, return_features=True)

        bottleneck = feat["bottleneck"]   # [B, 512,  7,  7]
        s4         = feat["skip4"]        # [B, 512, 14, 14]
        s3         = feat["skip3"]        # [B, 256, 28, 28]
        s2         = feat["skip2"]        # [B, 128, 56, 56]
        s1         = feat["skip1"]        # [B,  64,112,112]

        # ── Decoder ───────────────────────────────────────────────────────
        x = self.dec4(bottleneck, s4)    # [B, 512, 14, 14]
        x = self.dec3(x, s3)             # [B, 256, 28, 28]
        x = self.dec2(x, s2)             # [B, 128, 56, 56]
        x = self.dec1(x, s1)             # [B,  64,112,112]
        x = self.dec0(x)                 # [B,  32,224,224]

        # ── Output head ───────────────────────────────────────────────────
        logits = self.head(x)            # [B, num_classes, 224, 224]
        return logits