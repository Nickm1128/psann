from __future__ import annotations

"""Public sklearn-style estimator surface and serialization compatibility aliases."""

from ._sklearn import (
    GeoSparseRegressor,
    PSANNClassifier,
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
    "PSANNClassifier",
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "SGRPSANNRegressor",
    "WaveResNetRegressor",
    "GeoSparseRegressor",
]

for _cls in (
    PSANNRegressor,
    PSANNClassifier,
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
