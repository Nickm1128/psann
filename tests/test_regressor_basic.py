import numpy as np
import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor


def make_data(n=1000, seed=123):
    rs = np.random.RandomState(seed)
    X = np.linspace(-3, 3, n).reshape(-1, 1).astype(np.float32)
    y = 0.7 * np.exp(-0.3 * np.abs(X)) * np.sin(4.0 * X) + 0.05 * rs.randn(*X.shape)
    return X, y.astype(np.float32)


def test_fit_predict_score_smoke():
    X, y = make_data()
    n = len(X)
    idx = np.arange(n)
    np.random.RandomState(0).shuffle(idx)
    tr = idx[: int(0.8 * n)]
    te = idx[int(0.8 * n) :]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    model = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=150,
        lr=1e-3,
        batch_size=128,
        early_stopping=True,
        patience=20,
        random_state=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape[0] == X_test.shape[0]
    # should achieve a reasonable score
    score = model.score(X_test, y_test)
    assert score > 0.3


def test_scaler_resets_without_warm_start():
    rs = np.random.RandomState(123)
    X = rs.randn(64, 3).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True) + 0.05 * rs.randn(64, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=16,
        epochs=2,
        batch_size=16,
        scaler="standard",
        warm_start=False,
    )
    est.fit(X, y)
    first_state = est._scaler_state_
    assert first_state is not None
    assert first_state["n"] == X.shape[0]

    X_small = X[:16]
    y_small = y[:16]
    est.fit(X_small, y_small)
    second_state = est._scaler_state_
    assert second_state is not None
    assert second_state["n"] == X_small.shape[0]


def test_scaler_warm_start_accumulates_stats():
    rs = np.random.RandomState(9)
    X = rs.randn(50, 4).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True) + 0.1 * rs.randn(50, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=12,
        epochs=2,
        batch_size=25,
        scaler="standard",
        warm_start=True,
    )
    est.fit(X, y)
    first_state = est._scaler_state_
    assert first_state is not None
    assert first_state["n"] == X.shape[0]

    X_extra = rs.randn(30, 4).astype(np.float32)
    y_extra = (X_extra.sum(axis=1, keepdims=True) + 0.1 * rs.randn(30, 1)).astype(np.float32)
    est.fit(X_extra, y_extra)
    second_state = est._scaler_state_
    assert second_state is not None
    assert second_state["n"] == X.shape[0] + X_extra.shape[0]


def test_minmax_scaler_warm_start_tracks_global_range():
    rs = np.random.RandomState(21)
    X_first = rs.uniform(-1.0, 1.0, size=(40, 3)).astype(np.float32)
    y_first = (X_first.sum(axis=1, keepdims=True) + 0.05 * rs.randn(40, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=10,
        epochs=1,
        batch_size=20,
        scaler="minmax",
        warm_start=True,
    )
    est.fit(X_first, y_first)

    state_first = est._scaler_state_
    assert state_first is not None
    assert est._scaler_kind_ == "minmax"
    assert np.allclose(state_first["min"], X_first.min(axis=0), atol=1e-6)
    assert np.allclose(state_first["max"], X_first.max(axis=0), atol=1e-6)

    X_second = rs.uniform(-2.5, 2.0, size=(60, 3)).astype(np.float32)
    y_second = (X_second.sum(axis=1, keepdims=True) + 0.05 * rs.randn(60, 1)).astype(np.float32)
    est.fit(X_second, y_second)

    state_second = est._scaler_state_
    assert state_second is not None
    expected_min = np.minimum(X_first.min(axis=0), X_second.min(axis=0))
    expected_max = np.maximum(X_first.max(axis=0), X_second.max(axis=0))

    assert np.allclose(state_second["min"], expected_min, atol=1e-6)
    assert np.allclose(state_second["max"], expected_max, atol=1e-6)

    scale = np.where((expected_max - expected_min) > 1e-8, expected_max - expected_min, 1.0)
    stacked = np.vstack([X_first, X_second])
    transformed = (stacked - expected_min) / scale
    assert transformed.min() >= -1e-6
    assert transformed.max() <= 1.0 + 1e-6
