import numpy as np
import pytest

pytest.importorskip("torch")

from psann import GeoSparseRegressor


def test_geo_sparse_regressor_fit_predict():
    rs = np.random.RandomState(0)
    X = rs.randn(128, 4, 4).astype(np.float32)
    y = X.reshape(X.shape[0], -1).sum(axis=1, keepdims=True).astype(np.float32)

    model = GeoSparseRegressor(
        shape=(4, 4),
        hidden_layers=2,
        k=4,
        activation_type="relu",
        epochs=5,
        batch_size=32,
        lr=1e-3,
        random_state=0,
        early_stopping=False,
    )
    model.fit(X, y, verbose=0)
    preds = model.predict(X[:10])
    assert preds.shape == (10, 1)


def test_geo_sparse_regressor_fit_predict_with_relu_sigmoid_psann():
    rs = np.random.RandomState(1)
    X = rs.randn(96, 4, 4).astype(np.float32)
    y = X.reshape(X.shape[0], -1).sum(axis=1, keepdims=True).astype(np.float32)

    model = GeoSparseRegressor(
        shape=(4, 4),
        hidden_layers=2,
        k=4,
        activation_type="relu_sigmoid_psann",
        activation={"slope_init": 1.0, "clip_max": 1.0},
        epochs=4,
        batch_size=32,
        lr=1e-3,
        random_state=0,
        early_stopping=False,
    )
    model.fit(X, y, verbose=0)
    preds = model.predict(X[:8])
    assert preds.shape == (8, 1)
