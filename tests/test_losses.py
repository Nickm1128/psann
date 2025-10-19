import numpy as np
import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor


def _data(n=500, seed=0):
    rs = np.random.RandomState(seed)
    X = np.linspace(-2, 2, n).reshape(-1, 1).astype(np.float32)
    y = 0.6 * np.exp(-0.4 * np.abs(X)) * np.sin(3.0 * X) + 0.05 * rs.randn(*X.shape)
    return X, y.astype(np.float32)


def test_huber_loss_runs():
    X, y = _data()
    model = PSANNRegressor(
        epochs=20, loss="huber", loss_params={"delta": 0.5}, early_stopping=False
    )
    model.fit(X, y)
    preds = model.predict(X[:10])
    assert preds.shape[0] == 10


def test_custom_loss_callable_vector_output():
    X, y = _data()

    def custom_vec_loss(pred, target, scale=1.0):
        # Returns per-sample absolute error scaled; wrapper reduces it.
        return (pred - target).abs() * scale

    model = PSANNRegressor(
        epochs=10, loss=custom_vec_loss, loss_params={"scale": 0.5}, loss_reduction="mean"
    )
    model.fit(X, y)
    preds = model.predict(X[:8])
    assert preds.shape[0] == 8
