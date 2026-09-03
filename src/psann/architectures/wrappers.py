"""Architecture-local composition modules.

These are model building blocks, not sklearn estimators.  Keeping them beneath the
architecture package prevents the public builder registry from depending on the
legacy estimator implementation package.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, cast

import torch
import torch.nn as nn

from ..layers import SpectralGate1D
from ..models import WaveResNet


class _AttentionDenseModel(nn.Module):
    def __init__(
        self,
        token_backbone: nn.Module,
        attention_module: Optional[nn.Module],
        *,
        seq_len: int,
        token_dim: int,
        embed_dim: int,
        output_dim: int,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.token_backbone, self.attention = token_backbone, attention_module
        self.seq_len, self.token_dim, self.embed_dim = int(seq_len), int(token_dim), int(embed_dim)
        self.readout = nn.Linear(self.embed_dim, output_dim)
        if pool not in {"mean", "last"}:
            raise ValueError("pool must be 'mean' or 'last'.")
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.seq_len * self.token_dim:
            raise ValueError(
                "Attention-enabled models require their documented flattened token shape."
            )
        batch = x.shape[0]
        tokens = x.reshape(batch, self.seq_len, self.token_dim)
        embedded = self.token_backbone(tokens.reshape(batch * self.seq_len, self.token_dim))
        embedded = embedded.reshape(batch, self.seq_len, self.embed_dim)
        context = (
            embedded if self.attention is None else self.attention(embedded, embedded, embedded)[0]
        )
        return self.readout(context[:, -1, :] if self.pool == "last" else context.mean(dim=1))


class _AttentionConvModel(nn.Module):
    def __init__(
        self,
        conv_core: nn.Module,
        attention_module: nn.Module,
        *,
        spatial_shape: Tuple[int, ...],
        segmentation_head: bool,
    ) -> None:
        super().__init__()
        if not hasattr(conv_core, "forward_tokens"):
            raise TypeError("attention requires conv cores exposing forward_tokens.")
        self.conv_core, self.attention = conv_core, attention_module
        self.segmentation_head, self.spatial_shape = bool(segmentation_head), tuple(
            map(int, spatial_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_core = cast(Any, self.conv_core)
        tokens = conv_core.forward_tokens(x)
        batch, channels, spatial = tokens.shape[0], tokens.shape[1], tokens.shape[2:]
        seq = tokens.reshape(batch, channels, -1).transpose(1, 2)
        context = (
            self.attention(seq, seq, seq)[0].transpose(1, 2).reshape(batch, channels, *spatial)
        )
        if self.segmentation_head:
            return conv_core.head(context)
        return conv_core.fc(conv_core.pool(context).flatten(1))


class _FlattenedConvModel(nn.Module):
    """Expose a canonical convolutional core through the historical flat layout."""

    def __init__(self, core: nn.Module, *, input_shape: Tuple[int, ...], data_format: str) -> None:
        super().__init__()
        self.core = core
        self.input_shape = tuple(map(int, input_shape))
        self.data_format = data_format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != math.prod(self.input_shape):
            raise ValueError(
                "Flattened convolution inputs must match the documented feature shape."
            )
        shaped = x.reshape(x.shape[0], *self.input_shape)
        if self.data_format == "channels_last":
            shaped = shaped.movedim(-1, 1)
        return self.core(shaped)


class _WaveResNetSpectralDenseModel(nn.Module):
    def __init__(
        self, wave_core: WaveResNet, spectral_gate: SpectralGate1D, *, seq_len: int, token_dim: int
    ) -> None:
        super().__init__()
        self.wave, self.spectral = wave_core, spectral_gate
        self.seq_len, self.token_dim = int(seq_len), int(token_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.seq_len * self.token_dim:
            raise ValueError("Wave spectral models require their documented flattened token shape.")
        tokens = x.reshape(x.shape[0], self.seq_len, self.token_dim)
        if self.seq_len > 1:
            tokens = tokens + self.spectral(tokens)
        return (
            self.wave(tokens.reshape(x.shape[0], -1), context)
            if context is not None
            else self.wave(tokens.reshape(x.shape[0], -1))
        )


class _WaveResNetAttentionDenseModel(nn.Module):
    """Apply attention to flattened tokens before a WaveResNet readout."""

    def __init__(
        self, wave_core: WaveResNet, attention_module: nn.Module, *, seq_len: int, token_dim: int
    ) -> None:
        super().__init__()
        self.wave, self.attention = wave_core, attention_module
        self.seq_len, self.token_dim = int(seq_len), int(token_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.seq_len * self.token_dim:
            raise ValueError(
                "Wave attention models require their documented flattened token shape."
            )
        tokens = x.reshape(x.shape[0], self.seq_len, self.token_dim)
        tokens = self.attention(tokens, tokens, tokens)[0]
        flat = tokens.reshape(x.shape[0], -1)
        return self.wave(flat, context) if context is not None else self.wave(flat)


class _WaveResNetConvModel(nn.Module):
    def __init__(
        self,
        conv_core: nn.Module,
        wave_core: WaveResNet,
        *,
        spatial_shape: Tuple[int, ...],
        attention_module: Optional[nn.Module] = None,
        spectral_gate: Optional[SpectralGate1D] = None,
    ) -> None:
        super().__init__()
        if not hasattr(conv_core, "forward_tokens"):
            raise TypeError("WaveResNet convolutional mode requires conv_core.forward_tokens.")
        self.conv_core, self.wave = conv_core, wave_core
        self.attention, self.spectral = attention_module, spectral_gate
        self.spatial_shape = tuple(map(int, spatial_shape))

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return cast(Any, self.conv_core).forward_tokens(x)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        batch, channels = tokens.shape[:2]
        seq = tokens.reshape(batch, channels, -1).transpose(1, 2)
        if self.attention is not None:
            seq = self.attention(seq, seq, seq)[0]
        if self.spectral is not None and seq.shape[1] > 1:
            seq = seq + self.spectral(seq)
        flat = seq.reshape(batch, -1)
        return self.wave(flat, context) if context is not None else self.wave(flat)
