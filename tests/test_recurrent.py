import numpy as np
import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor, StateConfig


def test_stateful_step_and_sequence():
    # Simple synthetic autoregressive-like mapping: y_t ~ sin(x_t)
    rs = np.random.RandomState(0)
    T = 400
    x = np.linspace(0, 10, T).astype(np.float32)
    X = np.sin(x) + 0.05 * rs.randn(T).astype(np.float32)
    X = X.reshape(-1, 1)
    y = np.roll(X, -1)  # next-step target (last wraps around)
    y = y[:-1]
    X = X[:-1]

    model = PSANNRegressor(
        hidden_layers=1,
        hidden_width=16,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        early_stopping=True,
        patience=10,
        stateful=True,
        state=StateConfig(rho=0.98, beta=1.0, max_abs=2.5, init=1.0, detach=True),
        state_reset="none",
    )

    model.fit(X, y, verbose=0)

    # Step API
    model.reset_state()
    y0 = model.step(X[0].ravel())
    assert np.isfinite(y0)

    # Sequence API
    out_last = model.predict_sequence(X[:50], reset_state=True, return_sequence=False)
    out_seq = model.predict_sequence(X[:50], reset_state=True, return_sequence=True)
    assert np.isfinite(out_last)
    assert out_seq.shape[0] == 50

    # Long run stability thanks to clipping
    long_seq = np.ones((2000, 1), dtype=np.float32) * 0.5
    out_last_long = model.predict_sequence(long_seq, reset_state=True, return_sequence=False)
    assert np.isfinite(out_last_long)
