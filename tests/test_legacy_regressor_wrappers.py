from __future__ import annotations

import numpy as np
import pytest

from psann.estimators import PSANNRegressor
from psann.sklearn import (
    GeoSparseRegressor,
    ResConvPSANNRegressor,
    ResPSANNRegressor,
    SGRPSANNRegressor,
    WaveResNetRegressor,
)
from sklearn.base import clone


@pytest.mark.parametrize(
    "factory, X",
    [
        (
            lambda: ResPSANNRegressor(hidden_layers=2, epochs=1, batch_size=3, random_state=0),
            np.ones((6, 2), dtype=np.float32),
        ),
        (
            lambda: ResConvPSANNRegressor(hidden_layers=2, epochs=1, batch_size=3, random_state=0),
            np.ones((6, 1, 4, 4), dtype=np.float32),
        ),
        (
            lambda: WaveResNetRegressor(hidden_layers=2, epochs=1, batch_size=3, random_state=0),
            np.ones((6, 2), dtype=np.float32),
        ),
        (
            lambda: SGRPSANNRegressor(hidden_layers=2, epochs=1, batch_size=3, random_state=0),
            np.ones((6, 2), dtype=np.float32),
        ),
        (
            lambda: GeoSparseRegressor(
                shape=(1, 2), hidden_layers=2, epochs=1, batch_size=3, random_state=0
            ),
            np.ones((6, 2), dtype=np.float32),
        ),
    ],
)
def test_legacy_wrappers_warn_once_clone_and_run(factory, X):
    with pytest.warns(DeprecationWarning, match="deprecated") as warnings:
        wrapper = factory()
    assert len(warnings) == 1
    assert clone(wrapper).architecture == wrapper.architecture
    fitted = wrapper.fit(X, np.ones(len(X), dtype=np.float32))
    assert fitted.predict(X[:2]).shape == (2,)


def test_wildcard_canonical_surface_excludes_legacy_wrappers():
    namespace: dict[str, object] = {}
    exec("from psann.sklearn import *", namespace)
    assert namespace["PSANNRegressor"] is PSANNRegressor
    assert "WaveResNetRegressor" not in namespace


def test_wrapper_load_keeps_compatible_wrapper_type(tmp_path):
    X = np.ones((6, 2), dtype=np.float32)
    with pytest.warns(DeprecationWarning):
        estimator = ResPSANNRegressor(hidden_layers=2, epochs=1, batch_size=3).fit(
            X, np.ones(6, dtype=np.float32)
        )
    path = tmp_path / "residual.pt"
    estimator.save(str(path))
    with pytest.warns(DeprecationWarning):
        loaded = ResPSANNRegressor.load(str(path))
    assert isinstance(loaded, ResPSANNRegressor)
    assert loaded.predict(X[:2]).shape == (2,)
