"""Reusable custom layers
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer using inverted dropout.

    Randomly zeroes elements of the input tensor with probability p
    during training, and scales surviving elements by 1/(1-p) to
    preserve expected values. During evaluation, input passes through
    unchanged.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability. Must be in [0, 1).
        """
        super().__init__()

        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")

        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor — dropped+scaled during training,
            unchanged during eval.
        """
        # During eval: no dropout, return as-is
        if not self.training:
            return x

        # p=0 means keep everything — skip mask creation
        if self.p == 0.0:
            return x

        # Build binary mask: each element is 1 with prob (1-p), 0 with prob p
        # torch.bernoulli samples from Bernoulli(1-p) — no nn.Dropout used
        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full(x.shape, keep_prob, device=x.device, dtype=x.dtype))

        # Inverted dropout: scale up by 1/(1-p) so expected value is preserved
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        """Show p in print(model) output."""
        return f"p={self.p}"