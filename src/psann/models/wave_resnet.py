"""
WaveResNet backbone with sine residual blocks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Literal, Optional

import torch
from torch import nn

from ..activations import SineParam
from ..initializers import apply_siren_init
from ..layers import SineResidualBlock
from ..types import ActivationConfig

_DEFAULT_LEARNABLE = ("amplitude", "frequency", "decay")


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
        residual_alpha_init: float = 0.0,
        activation_config: Optional[ActivationConfig] = None,
        trainable_params: Optional[Iterable[str] | str] = _DEFAULT_LEARNABLE,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim, hidden_dim, and output_dim must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")

        self.context_dim = context_dim
        self.hidden_dim = int(hidden_dim)
        self.residual_alpha_init = float(residual_alpha_init)
        self.norm = norm
        self.use_film_flag = bool(use_film)
        self.use_phase_shift_flag = bool(use_phase_shift)
        self.dropout_rate = float(dropout)
        self.hidden_w0 = float(hidden_w0)
        self.first_layer_w0 = float(first_layer_w0)

        activation_cfg = deepcopy(activation_config) if activation_config is not None else {}
        learnable_source = activation_cfg.get("learnable", trainable_params)
        activation_cfg["learnable"] = self._normalize_trainable(learnable_source)
        self._activation_config = activation_cfg
        self.trainable_params = tuple(activation_cfg["learnable"])

        depth = int(depth)
        self.stem = nn.Linear(input_dim, self.hidden_dim)
        self.stem_w0 = self.first_layer_w0
        self.stem_activation = self._build_activation_module(self.hidden_dim)

        self.blocks = nn.ModuleList([self._make_block() for _ in range(depth)])

        self.head = nn.Linear(self.hidden_dim, output_dim)
        self.depth = depth

        apply_siren_init(self, first_layer_w0=first_layer_w0, hidden_w0=hidden_w0)

    @staticmethod
    def _normalize_trainable(trainable: Optional[Iterable[str] | str]) -> tuple[str, ...]:
        if trainable is None:
            return _DEFAULT_LEARNABLE
        if isinstance(trainable, str):
            key = trainable.strip().lower()
            if key in {"", "none", "off"}:
                return ()
            if key in {"all", "*"}:
                return _DEFAULT_LEARNABLE
            if key == "damping":
                key = "decay"
            if key not in _DEFAULT_LEARNABLE:
                raise ValueError(f"Unsupported trainable parameter: {trainable!r}")
            return (key,)
        items = list(trainable)
        if not items:
            return ()
        normalized: list[str] = []
        for item in items:
            if not isinstance(item, str):
                raise TypeError("trainable_params entries must be strings.")
            key = item.strip().lower()
            if key in {"", "none", "off"}:
                if len(items) > 1:
                    raise ValueError(
                        "trainable_params cannot include 'none' alongside other entries."
                    )
                return ()
            if key in {"all", "*"}:
                return _DEFAULT_LEARNABLE
            if key == "damping":
                key = "decay"
            if key not in _DEFAULT_LEARNABLE:
                raise ValueError(f"Unsupported trainable parameter: {item!r}")
            if key not in normalized:
                normalized.append(key)
        return tuple(normalized)

    def _build_activation_module(self, out_features: int) -> SineParam:
        cfg = deepcopy(self._activation_config)
        return SineParam(out_features, **cfg)

    def _make_block(self) -> SineResidualBlock:
        return SineResidualBlock(
            self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            w0=self.hidden_w0,
            norm=self.norm,
            context_dim=self.context_dim,
            use_film=self.use_film_flag,
            use_phase_shift=self.use_phase_shift_flag,
            dropout=self.dropout_rate,
            residual_alpha_init=self.residual_alpha_init,
            activation=self._build_activation_module(self.hidden_dim),
        )

    def forward_features(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
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
        h = self.stem_activation(self.stem_w0 * h)

        for block in self.blocks:
            h = block(h, c)

        return h

    def forward(self, x: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass producing predictions."""
        h = self.forward_features(x, c)
        return self.head(h)

    def add_blocks(self, count: int) -> list[nn.Module]:
        """Append additional residual blocks, keeping existing weights intact."""
        if count <= 0:
            return []
        new_blocks: list[nn.Module] = []
        for _ in range(int(count)):
            block = self._make_block()
            apply_siren_init(block, first_layer_w0=self.hidden_w0, hidden_w0=self.hidden_w0)
            new_blocks.append(block)

        device = next(self.parameters()).device
        for block in new_blocks:
            block.to(device)
            self.blocks.append(block)
        self.depth = len(self.blocks)
        return new_blocks


def build_wave_resnet(**kwargs) -> WaveResNet:
    """Convenience factory for :class:`WaveResNet`."""
    return WaveResNet(**kwargs)
