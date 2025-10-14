"""
Residual sine blocks with optional context modulation.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
from torch import nn
from torch.nn.utils import weight_norm

from .film import FiLM


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the last feature dimension."""

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.scale = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        norm = torch.sqrt(norm + self.eps)
        return x * (self.scale / norm)


class SineResidualBlock(nn.Module):
    """Residual block leveraging sine activations and optional FiLM modulation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        use_bias: bool = True,
        w0: float = 1.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.sin,
        norm: Literal["none", "weight", "rms"] = "none",
        context_dim: Optional[int] = None,
        use_film: bool = True,
        use_phase_shift: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim, hidden_dim, and out_dim must be positive.")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be in [0, 1].")
        if norm not in {"none", "weight", "rms"}:
            raise ValueError(f"Unsupported norm type: {norm}")

        self.activation = activation
        self.w0 = w0
        self.use_phase_shift = use_phase_shift and context_dim is not None
        self.use_film = use_film and context_dim is not None

        input_linear = nn.Linear(in_dim, hidden_dim, bias=use_bias)
        output_linear = nn.Linear(hidden_dim, out_dim, bias=use_bias)

        if norm == "weight":
            input_linear = weight_norm(input_linear)
            output_linear = weight_norm(output_linear)

        self.input_linear = input_linear
        self.output_linear = output_linear
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.branch_norm: Optional[nn.Module] = None

        if norm == "rms":
            self.branch_norm = RMSNorm(out_dim)

        self.phase_shift: Optional[nn.Module] = None
        if self.use_phase_shift:
            self.phase_shift = nn.Linear(context_dim, hidden_dim, bias=True)

        self.film: Optional[FiLM] = None
        if self.use_film:
            self.film = FiLM(context_dim, hidden_dim)

        self.skip: Optional[nn.Module]
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(..., in_dim)``.
            c: Optional context tensor of shape ``(..., context_dim)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(..., out_dim)``.
        """
        residual = x if self.skip is None else self.skip(x)

        h = self.input_linear(x)
        if self.w0 != 1.0:
            h = self.w0 * h
        if self.use_phase_shift and c is not None and self.phase_shift is not None:
            phi = self.phase_shift(c)
            h = h + phi

        h = self.activation(h)

        if self.use_film and c is not None and self.film is not None:
            h = self.film(h, c)

        h = self.output_linear(h)

        if self.branch_norm is not None:
            h = self.branch_norm(h)

        h = self.dropout(h)

        return residual + h
