import numpy as np
import pytest

from psann import PSANNRegressor, StateConfig
from psann.utils import make_drift_series, make_shock_series, seed_all


@pytest.mark.parametrize(
    ("factory", "factor"),
    [
        (make_drift_series, 0.9),
        (make_shock_series, 1.01),
    ],
)
def test_online_streaming_rollouts(factory, factor):
    seed_all(7)
    X_t, y_t = factory(T=600, seed=13)
    X = X_t.numpy()
    y = y_t.numpy()

    n = X.shape[0]
    n_tr = int(0.6 * n)
    n_va = int(0.2 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va]
    X_te, y_te = X[n_tr + n_va :], y[n_tr + n_va :]

    model = PSANNRegressor(
        hidden_layers=1,
        hidden_units=48,
        epochs=110,
        batch_size=128,
        lr=1e-3,
        early_stopping=True,
        patience=20,
        stateful=True,
        stream_lr=5e-4,
        state=StateConfig(rho=0.99, beta=1.0, max_abs=3.0, init=1.0, detach=True),
        state_reset="none",
        random_state=3,
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), verbose=0)

    free_preds = model.predict_sequence(X_te, reset_state=True, return_sequence=True)
    online_preds = model.predict_sequence_online(X_te, y_te, reset_state=True, return_sequence=True)
    free_loss = float(np.mean((free_preds.reshape(y_te.shape) - y_te) ** 2))
    online_loss = float(np.mean((online_preds.reshape(y_te.shape) - y_te) ** 2))

    assert online_loss <= free_loss * factor
