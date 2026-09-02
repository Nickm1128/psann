from __future__ import annotations

import numpy as np
import pytest
import torch

from psann import (
    GeoSparseRegressor,
    PSANNRegressor,
    ResConvPSANNRegressor,
    ResPSANNRegressor,
    SGRPSANNRegressor,
    WaveResNetRegressor,
)

pytestmark = pytest.mark.gpu


def _flat_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(8, 4)
    return X, X.sum(axis=1, keepdims=True).astype(np.float32)


@pytest.mark.parametrize(
    ("estimator_cls", "kwargs", "shape"),
    [
        (PSANNRegressor, {}, "flat"),
        (ResPSANNRegressor, {}, "flat"),
        (ResConvPSANNRegressor, {}, "conv"),
        (WaveResNetRegressor, {}, "flat"),
        (SGRPSANNRegressor, {}, "sequence"),
        (GeoSparseRegressor, {"shape": (2, 2)}, "flat"),
    ],
)
def test_estimator_gpu_one_epoch_fit_predict(estimator_cls, kwargs, shape) -> None:
    X, y = _flat_data()
    if shape == "conv":
        X = X.reshape(8, 1, 2, 2)
    elif shape == "sequence":
        X = X.reshape(8, 2, 2)
    estimator = estimator_cls(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=4,
        device="cuda",
        random_state=3,
        **kwargs,
    )
    estimator.fit(X[:6], y[:6], verbose=0)
    prediction = estimator.predict(X[6:])
    assert prediction.shape[0] == 2
    assert next(estimator.model_.parameters()).device.type == "cuda"


@pytest.mark.parametrize(
    "estimator_cls, kwargs", [(PSANNRegressor, {}), (GeoSparseRegressor, {"shape": (2, 2)})]
)
def test_amp_and_checkpoint_map_locations_on_gpu(estimator_cls, kwargs, tmp_path) -> None:
    X, y = _flat_data()
    estimator = estimator_cls(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=4,
        device="cuda",
        amp=True,
        random_state=4,
        **kwargs,
    )
    estimator.fit(X[:6], y[:6], verbose=0)
    checkpoint = tmp_path / f"{estimator_cls.__name__}.pt"
    estimator.save(str(checkpoint))

    cpu_loaded = estimator_cls.load(str(checkpoint), map_location="cpu")
    assert next(cpu_loaded.model_.parameters()).device.type == "cpu"
    cuda_loaded = estimator_cls.load(str(checkpoint), map_location="cuda")
    assert next(cuda_loaded.model_.parameters()).device.type == "cuda"
    assert torch.cuda.is_available()
