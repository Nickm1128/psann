"""Parametric sine activation utilities for PSANN-LM.

Wraps the core `psann.activations.SineParam` with a simple config and a
factory for use inside MLPs/transformer blocks.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from torch import nn

from psann.architectures.components import build_activation
from ...architectures.compat import sine_policies


@dataclass
class SineConfig:
    amp_init: float = 1.0
    amp_init_std: float = 0.0
    freq_init: float = 1.0
    freq_init_std: float = 0.0
    damp_init: float = 0.01
    damp_init_std: float = 0.0
    trainable: bool = True
    decay_mode: str = "abs"  # "abs" | "relu" | "none"
    learnable: Optional[Iterable[str]] = None  # overrides trainable if provided
    # Optional per-parameter bounds applied after positivity transform
    amp_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    freq_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    damp_bounds: Optional[Tuple[Optional[float], Optional[float]]] = None
    # Optional init ranges; if provided, a scalar init is sampled uniformly
    amp_range: Optional[Tuple[float, float]] = None
    freq_range: Optional[Tuple[float, float]] = None
    damp_range: Optional[Tuple[float, float]] = None
    # Feature dimension for broadcasting (default last dim)
    feature_dim: int = -1


def build_sine(out_features: int, cfg: SineConfig | None = None) -> nn.Module:
    cfg = cfg or SineConfig()
    activation, _, _ = sine_policies(cfg)
    # Preserve the direct low-level feature-axis compatibility surface.
    values = {}
    import torch

    for old, name in (("amp", "amplitude"), ("freq", "frequency"), ("damp", "decay")):
        mean = getattr(cfg, old + "_init")
        std = getattr(cfg, old + "_init_std")
        rng = getattr(cfg, old + "_range")
        if std > 0:
            vector = torch.randn(out_features, dtype=torch.float32) * std + mean
            values[name] = vector.clamp_min(torch.finfo(vector.dtype).eps)
        elif rng is not None:
            values[name] = _random.uniform(*sorted(rng))
        else:
            values[name] = mean
    result = build_activation(activation, features=out_features, initial_values=values)
    setattr(result, "feature_dim", cfg.feature_dim)
    return result
