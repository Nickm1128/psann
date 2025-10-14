"""
Aggregated utilities for PSANN, including legacy helpers and new tooling.
"""

from __future__ import annotations

import math
import os
import random
from typing import Optional

import numpy as np
import torch

from .diagnostics import jacobian_spectrum, mutual_info_proxy, ntk_eigens, participation_ratio
from .linear_probe import encode_and_probe, fit_linear_probe
from .synthetic import make_context_rotating_moons, make_regime_switch_ts

__all__ = [
    "choose_device",
    "seed_all",
    "init_siren_linear_",
    "jacobian_spectrum",
    "mutual_info_proxy",
    "ntk_eigens",
    "participation_ratio",
    "encode_and_probe",
    "fit_linear_probe",
    "make_context_rotating_moons",
    "make_regime_switch_ts",
]


def choose_device(device: str | torch.device | None = "auto") -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def seed_all(seed: Optional[int] = None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def init_siren_linear_(lin: torch.nn.Linear, *, is_first: bool, w0: float = 30.0) -> None:
    """Initialize a Linear layer following SIREN recommendations."""
    in_features = lin.in_features
    if in_features <= 0:
        raise ValueError("Linear layer must have positive in_features.")
    if is_first:
        bound = 1.0 / in_features
    else:
        bound = math.sqrt(6.0 / in_features) / w0
    torch.nn.init.uniform_(lin.weight, -bound, bound)
    if lin.bias is not None:
        torch.nn.init.uniform_(lin.bias, -bound, bound)
