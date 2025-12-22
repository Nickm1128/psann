from __future__ import annotations

import numpy as np
import torch

from psann.sklearn import SGRPSANNRegressor


def _make_seq_dataset(
    n_samples: int = 32,
    steps: int = 12,
    features: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, steps, features)).astype(np.float32)
    mean_term = X.mean(axis=(1, 2), keepdims=False).astype(np.float32)[:, None]
    y = (X[:, -1, :1] + 0.1 * mean_term).astype(np.float32)
    return X, y


def test_sgr_psann_regressor_fit_predict() -> None:
    X, y = _make_seq_dataset()
    estimator = SGRPSANNRegressor(
        hidden_layers=2,
        hidden_width=8,
        epochs=2,
        batch_size=8,
        k_fft=8,
        random_state=0,
    )
    estimator.fit(X, y, verbose=0)
    preds = estimator.predict(X[:4])
    assert preds.shape == (4, 1)


def test_sgr_psann_regressor_short_sequence_window() -> None:
    X, y = _make_seq_dataset(steps=4)
    estimator = SGRPSANNRegressor(
        hidden_layers=1,
        hidden_width=8,
        epochs=1,
        batch_size=4,
        k_fft=16,
    )
    estimator.fit(X, y, verbose=0)
    preds = estimator.predict(X[:2])
    assert preds.shape == (2, 1)


def test_sgr_psann_gate_types_forward() -> None:
    estimator = SGRPSANNRegressor(
        hidden_layers=1,
        hidden_width=6,
        epochs=1,
        k_fft=8,
        gate_type="fourier_features",
        gate_groups="full",
    )
    core = estimator._build_dense_core(input_dim=12, output_dim=1, input_shape=(4, 3))
    x = torch.randn(2, 12)
    y = core(x)
    assert y.shape == (2, 1)
