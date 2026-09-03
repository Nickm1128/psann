from __future__ import annotations

from .base import PSANNRegressor
from ..architectures.wrappers import (
    _AttentionConvModel,
    _AttentionDenseModel,
    _WaveResNetConvModel,
    _WaveResNetSpectralDenseModel,
)


def __getattr__(name: str):
    """Resolve deprecated variant names through the canonical facade layer.

    The old module files are retained solely for unversioned checkpoint reading;
    new imports from the package surface never activate their estimator classes.
    """

    if name in {
        "ResPSANNRegressor",
        "ResConvPSANNRegressor",
        "SGRPSANNRegressor",
        "WaveResNetRegressor",
        "GeoSparseRegressor",
    }:
        from ..estimators import compat

        return getattr(compat, name)
    raise AttributeError(name)


__all__ = [
    "PSANNRegressor",
    "_AttentionDenseModel",
    "_AttentionConvModel",
    "_WaveResNetSpectralDenseModel",
    "_WaveResNetConvModel",
]
