"""
Feature-wise linear modulation (FiLM) layer.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class FiLM(nn.Module):
    """Applies feature-wise linear modulation conditioned on context.

    Args:
        in_dim: Dimension of the context vector ``c``.
        hidden_dim: Dimension of the feature representation ``h`` to modulate.
    """

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim and hidden_dim must be positive.")
        self.linear = nn.Linear(in_dim, 2 * hidden_dim, bias=True)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation.

        Args:
            h: Feature tensor of shape ``(..., hidden_dim)``.
            c: Context tensor of shape ``(..., in_dim)``.

        Returns:
            torch.Tensor: Modulated features with same shape as ``h``.
        """
        gamma, beta = self._compute_gamma_beta(c)
        return gamma * h + beta

    def _compute_gamma_beta(self, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma_beta = self.linear(c)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma, beta
