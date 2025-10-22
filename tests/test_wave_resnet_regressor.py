from __future__ import annotations

import numpy as np
import pytest

from psann.sklearn import WaveResNetRegressor


def _make_dataset(n_samples: int = 48) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_samples, 4)).astype(np.float32)
    context = rng.standard_normal((n_samples, 2)).astype(np.float32)
    signal = X.sum(axis=1, keepdims=True) + 0.5 * context[:, :1] - 0.25 * context[:, 1:]
    y = signal.astype(np.float32)
    return X, y, context


def test_wave_resnet_regressor_requires_context_when_configured() -> None:
    X, y, _ = _make_dataset()
    estimator = WaveResNetRegressor(hidden_layers=2, hidden_width=16, epochs=2, context_dim=2)
    with pytest.raises(ValueError, match="expects a context array"):
        estimator.fit(X, y)


def test_wave_resnet_regressor_infers_context_dimension() -> None:
    X, y, context = _make_dataset()
    estimator = WaveResNetRegressor(hidden_layers=2, hidden_width=16, epochs=2, context_dim=None)
    estimator.fit(X, y, context=context)
    assert estimator.context_dim == 2
    preds = estimator.predict(X[:4], context=context[:4])
    assert preds.shape[0] == 4
    with pytest.raises(ValueError, match="provide a matching context array"):
        estimator.predict(X[:2])


def test_wave_resnet_regressor_responds_to_context() -> None:
    X, y, context = _make_dataset()
    estimator = WaveResNetRegressor(
        hidden_layers=2,
        hidden_width=16,
        epochs=3,
        batch_size=16,
        context_dim=2,
        random_state=7,
    )
    estimator.fit(X, y, context=context)
    base = estimator.predict(X[:6], context=context[:6])
    shifted_context = context[:6] + 0.75
    shifted = estimator.predict(X[:6], context=shifted_context)
    assert not np.allclose(base, shifted)


def test_wave_resnet_regressor_cosine_builder_auto_context() -> None:
    X, y, _ = _make_dataset()
    estimator = WaveResNetRegressor(
        hidden_layers=2,
        hidden_width=16,
        epochs=2,
        batch_size=16,
        context_builder="cosine",
        context_builder_params={"frequencies": 1, "include_sin": False},
    )
    estimator.fit(X, y)
    assert estimator.context_dim == X.shape[1]
    preds = estimator.predict(X[:5])
    assert preds.shape == (5, 1)
