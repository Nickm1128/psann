from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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
        amplitude_init: float | torch.Tensor = 1.0,
        frequency_init: float | torch.Tensor = 1.0,
        decay_init: float | torch.Tensor = 0.1,
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

        def _as_init_vector(value: float | torch.Tensor, *, name: str) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                t = value.detach().to(device="cpu", dtype=torch.float32).flatten()
            else:
                t = torch.as_tensor(value, dtype=torch.float32).flatten()

            if t.numel() == 0:
                raise ValueError(f"{name}_init must be a scalar or a tensor with {self.out_features} elements")
            if t.numel() == 1:
                t = torch.full((self.out_features,), float(t.item()), dtype=torch.float32)
            elif t.numel() != self.out_features:
                raise ValueError(
                    f"{name}_init must be a scalar or have shape ({self.out_features},); got {tuple(t.shape)}"
                )
            else:
                t = t.reshape(self.out_features)

            eps = torch.finfo(t.dtype).eps
            return t.clamp_min(eps)

        # Store raw parameters as vectors (out_features,); broadcast in forward
        A0 = _as_init_vector(amplitude_init, name="amplitude")
        f0 = _as_init_vector(frequency_init, name="frequency")
        d0 = _as_init_vector(decay_init, name="decay")

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
            x = x.clamp(
                min=lo if lo is not None else -math.inf, max=hi if hi is not None else math.inf
            )
        return x

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., feature_dim, ...). Supports (N,F) or (N,F,*) or (N,*,F) via feature_dim.
        params = F.softplus(torch.stack((self._A, self._f, self._d), dim=0))
        A = params[0]
        f = params[1] + 1e-6
        d = params[2]

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


class PhaseSineParam(SineParam):
    """Sine activation with learnable amplitude, frequency, decay, and phase."""

    def __init__(
        self,
        out_features: int,
        *,
        amplitude_init: float | torch.Tensor = 1.0,
        frequency_init: float | torch.Tensor = 1.0,
        decay_init: float | torch.Tensor = 0.1,
        phase_init: float = 0.0,
        learnable: Iterable[str] | str = ("amplitude", "frequency", "decay"),
        phase_trainable: bool = True,
        decay_mode: str = "abs",
        bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        feature_dim: int = -1,
    ) -> None:
        super().__init__(
            out_features,
            amplitude_init=amplitude_init,
            frequency_init=frequency_init,
            decay_init=decay_init,
            learnable=learnable,
            decay_mode=decay_mode,
            bounds=bounds,
            feature_dim=feature_dim,
        )
        self._phi = nn.Parameter(torch.full((self.out_features,), float(phase_init)))
        self._phi.requires_grad = bool(phase_trainable)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        params = F.softplus(torch.stack((self._A, self._f, self._d), dim=0))
        A = params[0]
        f = params[1] + 1e-6
        d = params[2]

        A = self._apply_bounds(A, "amplitude")
        f = self._apply_bounds(f, "frequency")
        d = self._apply_bounds(d, "decay")

        if self.decay_mode == "abs":
            g = z.abs()
        elif self.decay_mode == "relu":
            g = F.relu(z)
        else:
            g = 0.0

        fd = self.feature_dim if self.feature_dim >= 0 else (z.ndim + self.feature_dim)
        shape = [1] * z.ndim
        shape[fd] = self.out_features
        A = A.view(*shape)
        f = f.view(*shape)
        d = d.view(*shape)
        phi = self._phi.view(*shape)

        return A * torch.exp(-d * g) * torch.sin(f * z + phi)


def _normalize_activation_name(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "sine": "psann",
        "respsann": "psann",
    }
    return aliases.get(key, key)


def _normalize_ratios(
    ratios: Optional[Sequence[float]],
    n: int,
    *,
    ratio_sum_tol: float,
) -> list[float]:
    if n <= 0:
        raise ValueError("activation_types must be non-empty")
    if ratios is None:
        return [1.0 / float(n)] * n
    if len(ratios) != n:
        raise ValueError("activation_ratios must have the same length as activation_types")
    out: list[float] = []
    for r in ratios:
        rf = float(r)
        if not math.isfinite(rf):
            raise ValueError("activation_ratios must all be finite")
        if rf < 0:
            raise ValueError("activation_ratios must be >= 0")
        out.append(rf)
    s = float(sum(out))
    if s <= 0:
        raise ValueError("activation_ratios must have at least one positive value")
    if abs(s - 1.0) > float(ratio_sum_tol):
        raise ValueError(
            f"activation_ratios must sum to 1 (within tol={float(ratio_sum_tol)}); got sum={s}"
        )
    # Renormalize so the sum is exactly 1.0 (within float error).
    out = [r / s for r in out]
    # Nudge last element to make the sum exactly 1.0 (deterministic).
    drift = 1.0 - float(sum(out))
    out[-1] = float(out[-1] + drift)
    return out


def _apportion_counts(total: int, ratios: Sequence[float]) -> list[int]:
    """Convert ratios (sum=1) into integer counts summing to total."""
    if total <= 0:
        raise ValueError("total must be positive")
    raw = [float(r) * float(total) for r in ratios]
    base = [int(math.floor(x)) for x in raw]
    remainders = [x - float(b) for x, b in zip(raw, base)]
    remaining = int(total - sum(base))
    if remaining < 0:
        # Should not happen with floor, but guard.
        raise ValueError("Internal error: apportioned counts exceeded total")
    order = sorted(range(len(base)), key=lambda i: (remainders[i], -i), reverse=True)
    for j in range(remaining):
        base[order[j % len(base)]] += 1
    # Safety: ensure exact sum.
    if sum(base) != total:
        raise ValueError("Internal error: apportioned counts do not sum to total")
    return base


BuilderFn = Callable[[int], nn.Module]


class MixedActivation(nn.Module):
    """Apply different activation modules to different feature indices.

    This is designed for "per-neuron" activation heterogeneity: a fixed partition
    of the feature dimension is assigned to each activation type.
    """

    def __init__(
        self,
        out_features: int,
        *,
        activation_types: Sequence[str],
        activation_ratios: Optional[Sequence[float]] = None,
        ratio_sum_tol: float = 1e-3,
        seed: Optional[int] = None,
        layout: str = "random",  # "random" | "contiguous"
        feature_dim: int = -1,
        builders: Optional[Mapping[str, BuilderFn]] = None,
    ) -> None:
        super().__init__()
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        self.out_features = int(out_features)
        self.feature_dim = int(feature_dim)

        norm_types = [_normalize_activation_name(t) for t in activation_types]
        if not norm_types:
            raise ValueError("activation_types must be non-empty")
        if len(set(norm_types)) != len(norm_types):
            raise ValueError("activation_types must be unique")

        ratios = _normalize_ratios(
            activation_ratios, len(norm_types), ratio_sum_tol=float(ratio_sum_tol)
        )
        counts = _apportion_counts(self.out_features, ratios)

        key = str(layout or "random").lower()
        if key not in {"random", "contiguous"}:
            raise ValueError("layout must be 'random' or 'contiguous'")
        self.layout = key
        self.seed = seed

        perm = torch.arange(self.out_features, dtype=torch.long)
        if key == "random":
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(int(seed))
            perm = torch.randperm(self.out_features, generator=g)

        # Default builders (can be overridden by callers).
        default_builders: dict[str, BuilderFn] = {
            "relu": lambda n: nn.ReLU(),
            "tanh": lambda n: nn.Tanh(),
            "gelu": lambda n: nn.GELU(),
            "psann": lambda n: SineParam(n),
            "phase_psann": lambda n: PhaseSineParam(n),
        }
        merged_builders = dict(default_builders)
        if builders:
            merged_builders.update({str(k).lower(): v for k, v in builders.items()})

        self.acts = nn.ModuleDict()
        self._idx_attr: dict[str, str] = {}
        self._slices: dict[str, tuple[int, int]] = {}

        start = 0
        seen = torch.zeros(self.out_features, dtype=torch.bool)
        for t, c in zip(norm_types, counts):
            end = start + int(c)
            idx = perm[start:end]
            start = end
            if idx.numel() > 0:
                seen[idx] = True
            safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in t)
            attr = f"_idx_{safe}"
            self.register_buffer(attr, idx.to(dtype=torch.long), persistent=False)
            self._idx_attr[t] = attr
            if idx.numel() > 0:
                start_i = int(idx[0].item())
                end_i = int(idx[-1].item())
                if (end_i - start_i + 1) == int(idx.numel()):
                    try:
                        if bool(
                            (
                                idx
                                == torch.arange(
                                    start_i, start_i + int(idx.numel()), dtype=idx.dtype
                                )
                            ).all()
                        ):
                            self._slices[t] = (start_i, int(idx.numel()))
                    except Exception:
                        pass

            if int(c) == 0:
                continue
            builder = merged_builders.get(str(t).lower())
            if builder is None:
                raise ValueError(f"Unsupported activation type in mix: {t}")
            self.acts[t] = builder(int(c))

        if not bool(seen.all().item()):
            raise ValueError("Internal error: mixed activation indices do not cover all features")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fd = self.feature_dim if self.feature_dim >= 0 else (x.ndim + self.feature_dim)
        if fd < 0 or fd >= x.ndim:
            raise ValueError("feature_dim is out of range for input tensor")
        if fd != x.ndim - 1:
            x_moved = x.movedim(fd, -1)
            y = self._forward_last_dim(x_moved)
            return y.movedim(-1, fd)
        return self._forward_last_dim(x)

    def _forward_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != self.out_features:
            raise ValueError("Input last dimension must match out_features.")
        # Optimization: when a cheap, shape-agnostic activation is present (e.g. ReLU),
        # apply it to the full tensor once, then override the remaining (typically
        # parameterized) activations on their assigned indices.
        default_type = None
        for candidate in ("relu", "gelu", "tanh"):
            if candidate in self.acts:
                default_type = candidate
                break

        if default_type is None:
            out = x.clone()
        elif len(self.acts) == 1:
            return self.acts[default_type](x)
        else:
            # We will mutate `out` in-place to override indices, so ensure it does
            # not alias an autograd-saved tensor (e.g. output of ReLU).
            out = self.acts[default_type](x).clone()
        for t, act in self.acts.items():
            if t == default_type:
                continue
            idx = getattr(self, self._idx_attr[t])
            if idx.numel() == 0:
                continue
            sl = self._slices.get(t)
            if sl is not None:
                start, length = sl
                out_slice = out.narrow(-1, start, length)
                x_slice = x.narrow(-1, start, length)
                out_slice.copy_(act(x_slice))
                continue
            out.index_copy_(-1, idx, act(x.index_select(-1, idx)))
        return out
