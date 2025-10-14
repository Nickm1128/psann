"""
Initialization utilities for SIREN-style sine networks.
"""

from __future__ import annotations

import math
from typing import Iterator
import torch
from torch import nn


def siren_uniform_(tensor: torch.Tensor, fan_in: int, w0: float = 1.0) -> None:
    """In-place SIREN uniform initialization.

    Args:
        tensor: Weight tensor to initialize.
        fan_in: Incoming feature dimension for the tensor.
        w0: Frequency scaling parameter from the SIREN paper.
    """
    if fan_in <= 0:
        raise ValueError(f"fan_in must be positive, received {fan_in}.")
    if w0 <= 0:
        raise ValueError(f"w0 must be positive, received {w0}.")
    bound = math.sqrt(6.0 / fan_in) / w0
    with torch.no_grad():
        tensor.uniform_(-bound, bound)


def _iter_linears(module: nn.Module) -> Iterator[nn.Linear]:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            yield child


def apply_siren_init(
    module: nn.Module,
    *,
    first_layer_w0: float = 30.0,
    hidden_w0: float = 1.0,
) -> None:
    """Apply SIREN initialization to all ``nn.Linear`` layers within ``module``.

    Args:
        module: The module whose linear submodules will be initialized.
        first_layer_w0: ``w0`` value for the first linear layer encountered.
        hidden_w0: ``w0`` value for the remaining linear layers.
    """
    for idx, linear in enumerate(_iter_linears(module)):
        w0 = first_layer_w0 if idx == 0 else hidden_w0
        weight = _resolve_weight_parameter(linear)
        fan_in = weight.shape[1]
        siren_uniform_(weight, fan_in, w0=w0)
        if hasattr(linear, "weight_g"):
            nn.init.ones_(linear.weight_g)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)


def _resolve_weight_parameter(linear: nn.Linear) -> torch.Tensor:
    if hasattr(linear, "weight_v"):
        return linear.weight_v  # type: ignore[attr-defined]
    return linear.weight
