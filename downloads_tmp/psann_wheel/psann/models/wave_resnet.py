"""
WaveResNet backbone with sine residual blocks.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from ..initializers import apply_siren_init
from ..layers import SineResidualBlock


class WaveResNet(nn.Module):
    """Deep residual network equipped with sine activations and context modulation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        output_dim: int,
        *,
        first_layer_w0: float = 30.0,
        hidden_w0: float = 1.0,
        context_dim: Optional[int] = None,
        norm: Literal["none", "weight", "rms"] = "none",
        use_film: bool = True,
        use_phase_shift: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")

        self.stem = nn.Linear(input_dim, hidden_dim)
        self.stem_w0 = first_layer_w0

        self.blocks = nn.ModuleList(
            [
                SineResidualBlock(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    w0=hidden_w0,
                    norm=norm,
                    context_dim=context_dim,
                    use_film=use_film,
                    use_phase_shift=use_phase_shift,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.head = nn.Linear(hidden_dim, output_dim)
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.depth = depth

        apply_siren_init(self, first_layer_w0=first_layer_w0, hidden_w0=hidden_w0)

    def forward_features(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute latent features before the output head.

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.
            c: Optional context tensor of shape ``(batch, context_dim)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(batch, output_dim)``.
        """
        if c is not None and self.context_dim is None:
            raise ValueError("Context provided but model was constructed without context_dim.")

        h = self.stem(x)
        h = torch.sin(self.stem_w0 * h)

        for block in self.blocks:
            h = block(h, c)

        return h

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass producing predictions."""
        h = self.forward_features(x, c)
        return self.head(h)


def build_wave_resnet(**kwargs) -> WaveResNet:
    """Convenience factory for :class:`WaveResNet`."""
    return WaveResNet(**kwargs)
