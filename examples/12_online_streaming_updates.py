import numpy as np

from psann import PSANNRegressor


def make_series(T=1500, f=0.02, amp_f=0.002, noise=0.05, seed=0):
    rs = np.random.RandomState(seed)
    t = np.arange(T, dtype=np.float32)
    amp = 1.0 + 0.5 * np.sin(2 * np.pi * amp_f * t)
    x = amp * np.sin(2 * np.pi * f * t) + noise * rs.randn(T).astype(np.float32)
    return x.astype(np.float32)


if __name__ == "__main__":
    series = make_series(T=3000, seed=3)
    X = series[:-1].reshape(-1, 1)
    y = series[1:].reshape(-1, 1)

    n = len(X)
    n_tr = int(0.7 * n)
    n_va = int(0.15 * n)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va]
    X_te, y_te = X[n_tr + n_va :], y[n_tr + n_va :]

    model = PSANNRegressor(
        hidden_layers=2,
        hidden_width=32,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        stream_lr=3e-4,
        early_stopping=True,
        patience=20,
        stateful=True,
        state={"rho": 0.985, "beta": 1.0, "max_abs": 3.0, "init": 1.0, "detach": True},
        state_reset="none",
    )
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), verbose=1)

    # Free-running sequence prediction (no target updates)
    free_preds = model.predict_sequence(X_te, reset_state=True, return_sequence=True)
    free_mse = float(((free_preds.reshape(-1, 1) - y_te) ** 2).mean())

    # Online prediction with per-step target updates
    online_preds = model.predict_sequence_online(X_te, y_te, reset_state=True)
    online_mse = float(((online_preds.reshape(-1, 1) - y_te) ** 2).mean())

    print(f"Free-run MSE:  {free_mse:.6f}")
    print(f"Online-upd MSE:{online_mse:.6f}")
