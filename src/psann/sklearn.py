from __future__ import annotations

"""Public sklearn-style estimator surface and serialization compatibility aliases."""

from .estimators import PSANNRegressor
from .estimators.compat import (
    GeoSparseRegressor,
    ResConvPSANNRegressor,
    ResPSANNRegressor,
    SGRPSANNRegressor,
    WaveResNetRegressor,
)
from ._sklearn import (
    _AttentionConvModel,
    _AttentionDenseModel,
    _WaveResNetConvModel,
    _WaveResNetSpectralDenseModel,
)

__all__ = [
    "PSANNRegressor",
]

for _cls in (_AttentionDenseModel,
    _AttentionConvModel,
    _WaveResNetSpectralDenseModel,
    _WaveResNetConvModel,
):
    # Compatibility classes retain their historical pickle import path.  The
    # canonical estimator is defined in psann.estimators.regressor.
    if _cls is not PSANNRegressor:
        _cls.__module__ = __name__
