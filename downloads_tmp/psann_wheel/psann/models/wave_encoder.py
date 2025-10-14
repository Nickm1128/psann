"""
Wave encoder utilities for frozen feature extraction.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .wave_resnet import WaveResNet


class WaveEncoder(nn.Module):
    """Wraps :class:`WaveResNet` to expose latent features conveniently."""

    def __init__(self, *, return_features: bool = True, **wave_kwargs) -> None:
        super().__init__()
        self.backbone = WaveResNet(**wave_kwargs)
        self.return_features = return_features

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return latent features or model predictions.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            c: Optional context tensor of shape ``(batch, context_dim)``.

        Returns:
            torch.Tensor: Feature tensor of shape ``(batch, hidden_dim)`` when
            ``return_features`` is ``True``; otherwise projection of shape
            ``(batch, output_dim)``.
        """
        features = self.backbone.forward_features(x, c)
        if self.return_features:
            return features
        return self.backbone.head(features)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Project latent features through the backbone's head."""
        return self.backbone.head(features)
