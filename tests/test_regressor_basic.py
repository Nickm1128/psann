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

