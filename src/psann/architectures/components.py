"""Shared numerical components for separately packaged model implementations.

These exports delegate to the established core implementations. Importing this
module does not import any language-model, estimator, or training integration.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite, isqrt
from numbers import Real
from typing import Literal, Mapping

import torch
from torch import nn

from ..activations import MixedActivation
from ..layers.geo_sparse import build_geo_connectivity, expand_in_indices_to_edges
from ..layers.sine_residual import RMSNorm
from ..layers.spectral import SpectralGate1D
from ..nn import DropPath
from ..nn_geo_sparse import _build_activation
from .config import ActivationConfig, GeometryConfig, normalize_activation_config

__all__ = [
    "build_activation",
    "RMSNorm",
    "DropPath",
    "SpectralGate1D",
    "GeometryConnectivity",
    "build_geometry_connectivity",
]


def build_activation(
    config: ActivationConfig | Mapping[str, object] | Literal["gelu"],
    *,
    features: int,
    initial_values: Mapping[str, float | torch.Tensor] | None = None,
) -> nn.Module:
    """Build a core activation with the supplied feature width.

    Existing typed activation policies and equivalent mappings share the same
    normalization. The fixed ``"gelu"`` literal (or ``{"kind": "gelu"}``) builds
    PyTorch GELU, which has no learnable policy fields.
    """
    if isinstance(features, bool) or not isinstance(features, int):
        raise TypeError("features must be an integer.")
    if features <= 0:
        raise ValueError("features must be positive.")
    resolved: dict[str, float | torch.Tensor] = {}
    if initial_values is not None:
        if not isinstance(initial_values, Mapping):
            raise TypeError("initial_values must be a mapping.")
        for key, value in initial_values.items():
            path = f"initial_values.{key}"
            if key not in {"amplitude", "frequency", "decay"}:
                raise ValueError(f"{path} is unknown.")
            if isinstance(value, torch.Tensor):
                if value.ndim != 1 or value.shape[0] != features:
                    raise ValueError(f"{path} must be a one-dimensional tensor of length features.")
                if value.dtype == torch.bool or value.is_complex():
                    raise TypeError(
                        f"{path} must contain real numbers, not booleans or complex values."
                    )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{path} must be finite.")
                resolved[key] = value.detach().clone()
            elif isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{path} must be a finite real scalar or tensor.")
            elif not isfinite(value):
                raise ValueError(f"{path} must be finite.")
            else:
                resolved[key] = float(value)
    if isinstance(config, str):
        if config != "gelu":
            raise ValueError("activation literal must be 'gelu'.")
        config = ActivationConfig(kind="gelu")
    if isinstance(config, Mapping) and config.get("kind") == "gelu":
        unknown = set(config) - {"kind"}
        if unknown:
            raise ValueError(f"activation.{sorted(unknown)[0]} is not a GELU field.")
        config = ActivationConfig(kind="gelu")
    policy = normalize_activation_config(config)
    raw = {field.name: getattr(policy, field.name) for field in fields(policy)}
    if resolved and policy.kind not in {"psann", "phase-psann", "relu-sigmoid-psann", "mixed"}:
        raise ValueError("initial_values requires a parameterized activation.")
    if policy.activation_types is not None:
        raw["activation_types"] = tuple(name.replace("-", "_") for name in policy.activation_types)
        # MixedActivation presents last-axis slices to every child, independently
        # of the feature axis in the external input.
        child_raw = dict(raw, feature_dim=-1)
        names = tuple(name.replace("-", "_") for name in policy.activation_types)
        builders = {
            name: (lambda n, key=name: _build_activation(key, n, child_raw)) for name in names
        }
        result = MixedActivation(
            features,
            activation_types=names,
            activation_ratios=policy.activation_ratios,
            ratio_sum_tol=policy.ratio_sum_tol,
            seed=policy.mix_seed,
            layout=policy.mix_layout,
            feature_dim=policy.feature_dim,
            builders=builders,
        )
        for name, child in result.acts.items():
            if name not in {"psann", "phase_psann", "relu_sigmoid_psann"}:
                continue
            indices = getattr(result, result._idx_attr[name])
            child_values = dict(child_raw)
            for key, value in resolved.items():
                child_values[key + "_init"] = (
                    value[indices] if isinstance(value, torch.Tensor) else value
                )
            if resolved:
                result.acts[name] = _build_activation(name, indices.numel(), child_values)
        return result
    for key, value in resolved.items():
        raw[key + "_init"] = value
    return _build_activation(policy.kind.replace("-", "_"), features, raw)


@dataclass(frozen=True)
class GeometryConnectivity:
    """Immutable geometry indices; materialized tensors are caller-owned copies.

    ``indices`` contains one tuple of input indices for each output feature.
    ``as_tensors`` returns the gather table and flattened source/destination
    edge lists, all with dtype ``torch.long`` on the requested device.
    """

    shape: tuple[int, int]
    indices: tuple[tuple[int, ...], ...]

    @staticmethod
    def edge_indices(indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand a rectangular gather table into source/destination edge indices."""
        if indices.ndim != 2 or indices.dtype != torch.long:
            raise ValueError("connectivity.indices must be a two-dimensional long tensor.")
        return expand_in_indices_to_edges(indices)

    def __post_init__(self) -> None:
        # Reuse strict geometry dimension validation and detach caller containers.
        shape = GeometryConfig(shape=self.shape).shape
        assert shape is not None
        object.__setattr__(self, "shape", shape)
        size = shape[0] * shape[1]
        rows = tuple(tuple(row) for row in self.indices)
        if len(rows) != size or not rows or not rows[0]:
            raise ValueError("connectivity.indices must have shape (features, k) with k > 0.")
        if any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("connectivity.indices rows must have equal lengths.")
        for row in rows:
            if any(isinstance(i, bool) or not isinstance(i, int) for i in row):
                raise TypeError("connectivity.indices must contain integers.")
            if any(i < 0 or i >= size for i in row):
                raise ValueError("connectivity.indices contains an out-of-range index.")
        object.__setattr__(self, "indices", rows)

    def as_tensors(
        self, *, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.tensor(self.indices, dtype=torch.long, device=device)
        src, dst = expand_in_indices_to_edges(indices)
        return indices, src, dst


def build_geometry_connectivity(
    config: GeometryConfig, *, features: int | None = None
) -> GeometryConnectivity:
    """Build connectivity using the core local, random, or hash algorithm.

    An explicit shape must multiply to ``features`` when both are supplied.
    With an omitted shape, factor ``features`` into the closest rectangular grid.
    Bias and compute mode remain execution policies and do not change the indices.
    """
    if not isinstance(config, GeometryConfig):
        raise TypeError("geometry must be a GeometryConfig.")
    if features is not None:
        if isinstance(features, bool) or not isinstance(features, int):
            raise TypeError("features must be an integer.")
        if features <= 0:
            raise ValueError("features must be positive.")
    shape = config.shape
    if shape is None:
        if features is None:
            raise ValueError("geometry.shape or features is required.")
        height = next(h for h in range(isqrt(features), 0, -1) if features % h == 0)
        shape = (height, features // height)
    elif features is not None and shape[0] * shape[1] != features:
        raise ValueError("geometry.shape must multiply to features.")
    tensor = build_geo_connectivity(
        shape,
        k=config.k,
        pattern=config.pattern,
        radius=config.radius,
        offsets=config.offsets,
        wrap_mode=config.wrap_mode,
        seed=config.seed,
    )
    return GeometryConnectivity(shape, tuple(tuple(row) for row in tensor.tolist()))
