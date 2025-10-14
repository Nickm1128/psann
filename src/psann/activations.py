from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _softplus_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    # inverse of softplus: y = 1/beta * log(exp(beta*x) - 1)
    # Clamp to avoid log(0)
    eps = torch.finfo(x.dtype).eps
    return torch.log(torch.expm1(beta * x).clamp_min(eps)) / beta


class SineParam(nn.Module):
    """Sine activation with learnable amplitude A, frequency f, and decay d.

    Forward: A * exp(-d * g(z)) * sin(f * z)

    Parameters are per-output feature (per neuron) and broadcast over batch.
    """

    def __init__(
        self,
        out_features: int,
        *,
        amplitude_init: float = 1.0,
        frequency_init: float = 1.0,
        decay_init: float = 0.1,
        learnable: Iterable[str] | str = ("amplitude", "frequency", "decay"),
        decay_mode: str = "abs",
        bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        feature_dim: int = -1,
    ) -> None:
        super().__init__()

        assert out_features > 0
        self.out_features = int(out_features)
        if isinstance(learnable, str):
            if learnable.lower() == "none":
                learnable = ()
            elif learnable.lower() in ("all", "*"):
                learnable = ("amplitude", "frequency", "decay")
            else:
                learnable = (learnable,)  # single
        learnable = set(learnable)

        assert decay_mode in {"abs", "relu", "none"}
        self.decay_mode = decay_mode
        self.bounds = bounds or {}
        self.feature_dim = int(feature_dim)

        # Store raw parameters as vectors (out_features,); broadcast in forward
        A0 = torch.full((self.out_features,), float(amplitude_init))
        f0 = torch.full((self.out_features,), float(frequency_init))
        d0 = torch.full((self.out_features,), float(decay_init))

        # Use softplus inverse for a stable starting point
        self._A = nn.Parameter(_softplus_inverse(A0))
        self._f = nn.Parameter(_softplus_inverse(f0))
        self._d = nn.Parameter(_softplus_inverse(d0))

        self._A.requires_grad = "amplitude" in learnable
        self._f.requires_grad = "frequency" in learnable
        self._d.requires_grad = "decay" in learnable

    def _apply_bounds(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if key not in self.bounds:
            return x
        lo, hi = self.bounds[key]
        if lo is not None or hi is not None:
            x = x.clamp(min=lo if lo is not None else -math.inf, max=hi if hi is not None else math.inf)
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., feature_dim, ...). Supports (N,F) or (N,F,*) or (N,*,F) via feature_dim.
        A = F.softplus(self._A)
        f = F.softplus(self._f) + 1e-6
        d = F.softplus(self._d)

        # Apply optional bounds after positivity transform
        A = self._apply_bounds(A, "amplitude")
        f = self._apply_bounds(f, "frequency")
        d = self._apply_bounds(d, "decay")

        if self.decay_mode == "abs":
            g = z.abs()
        elif self.decay_mode == "relu":
            g = F.relu(z)
        else:  # none
            g = 0.0

        # Reshape parameters for broadcasting along feature_dim
        fd = self.feature_dim if self.feature_dim >= 0 else (z.ndim + self.feature_dim)
        shape = [1] * z.ndim
        shape[fd] = self.out_features
        A = A.view(*shape)
        f = f.view(*shape)
        d = d.view(*shape)

        return A * torch.exp(-d * g) * torch.sin(f * z)
