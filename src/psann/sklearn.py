from __future__ import annotations

"""Public sklearn-style estimator surface and serialization compatibility aliases."""

from ._sklearn import (
    GeoSparseRegressor,
    PSANNRegressor,
    ResConvPSANNRegressor,
    ResPSANNRegressor,
    SGRPSANNRegressor,
    WaveResNetRegressor,
    _AttentionConvModel,
    _AttentionDenseModel,
    _WaveResNetConvModel,
    _WaveResNetSpectralDenseModel,
)

__all__ = [
    "PSANNRegressor",
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "SGRPSANNRegressor",
    "WaveResNetRegressor",
    "GeoSparseRegressor",
]

for _cls in (
    PSANNRegressor,
    ResPSANNRegressor,
    ResConvPSANNRegressor,
    SGRPSANNRegressor,
    WaveResNetRegressor,
    GeoSparseRegressor,
    _AttentionDenseModel,
    _AttentionConvModel,
    _WaveResNetSpectralDenseModel,
    _WaveResNetConvModel,
):
    _cls.__module__ = __name__
