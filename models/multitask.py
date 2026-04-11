"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.segmentation import DecoderBlock


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    Loads weights from 3 separately trained checkpoints and
    initialises a unified model with:
      - Shared VGG11 encoder backbone
      - Classification head  (37-class logits)
      - Localization head    (4 bbox coords, cxcywh pixel space)
      - Segmentation decoder (3-class pixel-wise logits)

    A single forward pass produces all three outputs simultaneously.
    """

    def __init__(
        self,
        num_breeds:       int = 37,
        seg_classes:      int = 3,
        in_channels:      int = 3,
        classifier_path:  str = "checkpoints/classifier.pth",
        localizer_path:   str = "checkpoints/localizer.pth",
        unet_path:        str = "checkpoints/unet.pth",
    ):
        """
        Initialize the shared backbone/heads using trained weights.

        Args:
            num_breeds:      Number of breed classes.
            seg_classes:     Number of segmentation classes.
            in_channels:     Number of input channels.
            classifier_path: Path to classifier checkpoint.
            localizer_path:  Path to localizer checkpoint.
            unet_path:       Path to unet checkpoint.
        """
        import gdown
        # Only download if checkpoint doesn't exist locally
        if not os.path.exists(classifier_path):
            gdown.download(id="1wKgOPjYftH2sp8CDcp7AsVB96EgGqlCO",
                           output=classifier_path, quiet=False)
        if not os.path.exists(localizer_path):
            gdown.download(id="1fiBsa4ec8apZPhVzTvFPrptmquxziDhL",
                           output=localizer_path, quiet=False)
        if not os.path.exists(unet_path):
            gdown.download(id="1_6vt6ENCLSMdDOQPhJtfeuozo938K2EJ",
                           output=unet_path, quiet=False)

        super().__init__()

        # ── Shared encoder backbone ───────────────────────────────────────
        self.encoder     = VGG11Encoder(in_channels=in_channels)  # cls + seg
        self.loc_encoder = VGG11Encoder(in_channels=in_channels)  # localization

        # ── Classification head: 4096 → num_breeds ───────────────────────
        self.cls_head = nn.Linear(4096, num_breeds)

        # ── Localization head: 4096 → 4 (cxcywh) ────────────────────────
        self.loc_head = nn.Sequential(
            nn.Linear(4096, 512),   # idx 0 — matches ckpt head.0
            nn.BatchNorm1d(512),    # idx 1 — matches ckpt head.1
            nn.ReLU(inplace=True),  # idx 2
            nn.Linear(512, 4),      # idx 3 — matches ckpt head.4
            nn.Sigmoid(),           # idx 4
        )

        # ── Segmentation decoder ──────────────────────────────────────────
        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64,  64)
        self.dec0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.seg_head = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ── Load weights from checkpoints ─────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_weights(classifier_path, localizer_path, unet_path, device)

    # ──────────────────────────────────────────────────────────────────────
    def _load_weights(
        self,
        classifier_path: str,
        localizer_path:  str,
        unet_path:       str,
        device:          torch.device,
    ):
        """Load and merge weights from all three checkpoints."""

        def load_sd(path):
            ckpt = torch.load(path, map_location=device)
            return ckpt.get("state_dict", ckpt)

        # ── 1. Classifier → shared encoder + cls_head ────────────────────
        if os.path.exists(classifier_path):
            sd = load_sd(classifier_path)
            enc_sd = {k.replace("encoder.", ""): v
                      for k, v in sd.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc_sd, strict=False)
            cls_sd = {k.replace("head.", ""): v
                      for k, v in sd.items() if k.startswith("head.")}
            self.cls_head.load_state_dict(cls_sd, strict=False)
            print(f"Loaded classifier weights from {classifier_path}")

        # ── 2. Localizer → dedicated loc_encoder + head ───────────────────
        if os.path.exists(localizer_path):
            sd = load_sd(localizer_path)
            # Load into dedicated loc_encoder (does NOT touch shared encoder)
            enc_sd = {k.replace("encoder.", ""): v
                      for k, v in sd.items() if k.startswith("encoder.")}
            self.loc_encoder.load_state_dict(enc_sd, strict=False)
            # Load localizer head with index remapping
            # Checkpoint: head.0 (Linear 4096→512), head.1 (BN), head.4 (Linear 512→4)
            # loc_head:   idx 0  (Linear),           idx 1  (BN), idx 3  (Linear)
            loc_sd = {}
            for k, v in sd.items():
                if not k.startswith("head."):
                    continue
                suffix = k[len("head."):]
                parts  = suffix.split(".", 1)
                idx    = int(parts[0])
                rest   = parts[1] if len(parts) > 1 else ""
                if idx in (0, 1):
                    new_key = f"{idx}.{rest}"
                elif idx == 4:
                    new_key = f"3.{rest}"
                else:
                    continue
                loc_sd[new_key] = v
            missing, unexpected = self.loc_head.load_state_dict(loc_sd, strict=False)
            print(f"Loaded localizer head — missing:{len(missing)} unexpected:{len(unexpected)}")
            print(f"Loaded localizer weights from {localizer_path}")

        # ── 3. U-Net → segmentation decoder ──────────────────────────────
        if os.path.exists(unet_path):
            sd = load_sd(unet_path)
            for block_name in ["dec4", "dec3", "dec2", "dec1", "dec0"]:
                block_sd = {k.replace(f"{block_name}.", ""): v
                            for k, v in sd.items()
                            if k.startswith(f"{block_name}.")}
                if block_sd:
                    getattr(self, block_name).load_state_dict(
                        block_sd, strict=True
                    )
            seg_sd = {k.replace("head.", ""): v
                      for k, v in sd.items() if k.startswith("head.")}
            self.seg_head.load_state_dict(seg_sd, strict=True)
            print(f"Loaded U-Net weights from {unet_path}")

    # ──────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """Single forward pass → all three task outputs.

        Args:
            x: [B, 3, 224, 224] input tensor.

        Returns:
            dict with keys:
              'classification': [B, num_breeds] logits
              'localization':   [B, 4] bbox (cx,cy,w,h) pixel space
              'segmentation':   [B, seg_classes, 224, 224] logits
        """
        # ── Shared encoder (classifier + segmentation) ────────────────────
        features, feat_dict = self.encoder(x, return_features=True)

        # ── Classification ────────────────────────────────────────────────
        cls_out = self.cls_head(features)              # [B, 37]

        # ── Localization (dedicated encoder) ──────────────────────────────
        loc_features = self.loc_encoder(x)             # [B, 4096]
        loc_out = self.loc_head(loc_features) * 224.0  # [B, 4] pixel space

        # ── Segmentation ──────────────────────────────────────────────────
        bottleneck = feat_dict["bottleneck"]
        s4 = feat_dict["skip4"]
        s3 = feat_dict["skip3"]
        s2 = feat_dict["skip2"]
        s1 = feat_dict["skip1"]

        d = self.dec4(bottleneck, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        d = self.dec0(d)
        seg_out = self.seg_head(d)                     # [B, 3, 224, 224]

        return {
            "classification": cls_out,
            "localization":   loc_out,
            "segmentation":   seg_out,
        }