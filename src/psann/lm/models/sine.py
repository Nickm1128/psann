"""Parametric sine activation utilities for PSANN-LM.

Wraps the core `psann.activations.SineParam` with a simple config and a
factory for use inside MLPs/transformer blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict
import random as _random

from ...activations import SineParam


@dataclass
class SineConfig:
    amp_init: float = 1.0
    freq_init: float = 1.0
    damp_init: float = 0.01
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


def build_sine(out_features: int, cfg: SineConfig | None = None) -> SineParam:
    cfg = cfg or SineConfig()
    learnable: Iterable[str]
    if cfg.learnable is not None:
        learnable = tuple(cfg.learnable)
    else:
        learnable = ("amplitude", "frequency", "decay") if cfg.trainable else ()

    # Optionally sample scalar inits from provided ranges
    def _sample_or(x: float, rng: Optional[Tuple[float, float]]) -> float:
        if rng is None:
            return float(x)
        lo, hi = float(rng[0]), float(rng[1])
        if hi < lo:
            lo, hi = hi, lo
        return float(_random.uniform(lo, hi))

    amp_init = _sample_or(cfg.amp_init, cfg.amp_range)
    freq_init = _sample_or(cfg.freq_init, cfg.freq_range)
    damp_init = _sample_or(cfg.damp_init, cfg.damp_range)

    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    if cfg.amp_bounds is not None:
        bounds["amplitude"] = cfg.amp_bounds
    if cfg.freq_bounds is not None:
        bounds["frequency"] = cfg.freq_bounds
    if cfg.damp_bounds is not None:
        bounds["decay"] = cfg.damp_bounds

    return SineParam(
        out_features,
        amplitude_init=float(amp_init),
        frequency_init=float(freq_init),
        decay_init=float(damp_init),
        learnable=learnable,
        decay_mode=str(cfg.decay_mode),
        bounds=bounds if bounds else None,
        feature_dim=int(cfg.feature_dim),
    )
