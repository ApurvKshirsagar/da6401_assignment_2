"""Custom IoU loss
"""

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Computes 1 - IoU for each predicted/target box pair.
    Loss is guaranteed to be in [0, 1].

    Boxes must be in (x_center, y_center, width, height) pixel format.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.

        Args:
            eps: Small value to avoid division by zero.
            reduction: 'mean' (default) | 'sum' | 'none'
        """
        super().__init__()

        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none', got '{reduction}'"
            )

        self.eps       = eps
        self.reduction = reduction

    def forward(
        self,
        pred_boxes:   torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes  (cx, cy, w, h) in pixel space.
            target_boxes: [B, 4] target boxes     (cx, cy, w, h) in pixel space.

        Returns:
            Scalar loss (reduction='mean'/'sum') or [B] tensor (reduction='none').
        """
        # ── 1. Convert cxcywh → xyxy ──────────────────────────────────────
        pred_x1   = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1   = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2   = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2   = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1    = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1    = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2    = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2    = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # ── 2. Intersection corners ───────────────────────────────────────
        inter_x1  = torch.max(pred_x1, tgt_x1)
        inter_y1  = torch.max(pred_y1, tgt_y1)
        inter_x2  = torch.min(pred_x2, tgt_x2)
        inter_y2  = torch.min(pred_y2, tgt_y2)

        # clamp to 0 — no negative intersection
        inter_w   = (inter_x2 - inter_x1).clamp(min=0)
        inter_h   = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h                        # [B]

        # ── 3. Individual box areas ───────────────────────────────────────
        pred_area  = pred_boxes[:, 2]  * pred_boxes[:, 3]    # w * h
        tgt_area   = target_boxes[:, 2] * target_boxes[:, 3]

        # ── 4. Union ──────────────────────────────────────────────────────
        union_area = pred_area + tgt_area - inter_area        # [B]

        # ── 5. IoU — clamped to [0,1] ────────────────────────────────────
        iou        = (inter_area / (union_area + self.eps)).clamp(0, 1)  # [B]

        # ── 6. Loss = 1 - IoU  (range [0, 1]) ────────────────────────────
        loss       = 1.0 - iou                                # [B]

        # ── 7. Reduction ──────────────────────────────────────────────────
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:   # "none"
            return loss