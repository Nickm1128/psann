"""CUDA smoke coverage for the canonical architecture registry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psann.architectures import ArchitectureConfig, ConvolutionConfig, GeometryConfig
from psann.estimators import PSANNRegressor
from psann.lsm import LSMConv2dExpander

pytestmark = pytest.mark.gpu


def _flat() -> tuple[np.ndarray, np.ndarray]:
    values = np.linspace(-1.0, 1.0, 24, dtype=np.float32).reshape(6, 4)
    return values, values.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    ("architecture", "reshape"),
    [
        (ArchitectureConfig.dense(), None),
        (ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=4)), (6, 1, 2, 2)),
        (ArchitectureConfig.for_wave(), None),
        (ArchitectureConfig.for_sequence(), (6, 2, 2)),
        (ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(2, 2))), None),
    ],
)
def test_canonical_registry_architecture_fits_on_cuda(architecture, reshape) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X, y = _flat()
    if reshape is not None:
        X = X.reshape(reshape)
    estimator = PSANNRegressor(
        architecture=architecture,
        hidden_layers=2,
        hidden_units=8,
        epochs=1,
        batch_size=3,
        device="cuda",
        random_state=17,
    ).fit(X[:4], y[:4], verbose=0)
    prediction = estimator.predict(X[4:])
    assert prediction.shape[0] == 2
    assert next(estimator.model_.parameters()).device.type == "cuda"


def test_convolutional_lsm_checkpoint_persists_on_cuda(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    X = np.ones((8, 1, 2, 2), dtype=np.float32)
    y = X.mean(axis=(1, 2, 3))
    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=4)),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        device="cuda",
        random_state=17,
        lsm=LSMConv2dExpander(out_channels=2, hidden_layers=1, conv_channels=4, epochs=1),
        lsm_train=True,
    ).fit(X, y)
    path = tmp_path / "cuda-conv-lsm.pt"
    estimator.save(str(path))
    loaded = PSANNRegressor.load(str(path), map_location="cuda")
    np.testing.assert_allclose(loaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-6)
    assert next(loaded.model_.parameters()).device.type == "cuda"
