import numpy as np

from psann import PSANNRegressor, LSMExpander


def make_data(n=5000, d=4, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.uniform(-3.0, 3.0, size=(n, d)).astype(np.float32)
    # Nonlinear target mixing multiple oscillatory components
    y = (
        np.sin(3.0 * X[:, 0])
        + 0.5 * np.sin(5.0 * (X[:, 1] + X[:, 2]))
        + 0.8 * np.exp(-0.3 * np.abs(X[:, 3])) * np.sin(2.0 * X[:, 3])
    )
    y += 0.4 * rs.randn(n)
    return X.astype(np.float32), y.astype(np.float32)[:, None]


if __name__ == "__main__":
    # Prepare data
    X, y = make_data()
    n = len(X)
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    tr = idx[: int(0.7 * n)]
    va = idx[int(0.7 * n) : int(0.85 * n)]
    te = idx[int(0.85 * n) :]
    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    X_test, y_test = X[te], y[te]

    # Baseline PSANN without LSM
    base = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=200,
        lr=1e-3,
        batch_size=256,
        early_stopping=True,
        patience=20,
        random_state=0,
    )
    base.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=1)
    r2_base = base.score(X_test, y_test)

    # Pre-fit LSM expander to increase feature dimensionality
    lsm = LSMExpander(output_dim=256, hidden_layers=12, hidden_width=256, sparsity=0.9, nonlinearity="sine", epochs=100, lr=1e-3, ridge=1e-4, random_state=0)
    lsm.fit(X_train)
    r2_lsm_recon = lsm.score_reconstruction(X_val)

    # PSANN with frozen, pre-fitted LSM as a preprocessing step
    with_lsm = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=200,
        lr=1e-3,
        batch_size=256,
        early_stopping=True,
        patience=20,
        random_state=0,
        lsm=lsm,
        lsm_train=False,  # use as fixed feature map (do not jointly fine-tune)
    )
    with_lsm.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=1)
    r2_lsm = with_lsm.score(X_test, y_test)

    print("Results on held-out test set:")
    print(f"- Baseline PSANN R^2:        {r2_base:.4f}")
    print(f"- With pre-fitted LSM R^2:   {r2_lsm:.4f}")
    print(f"LSM validation reconstruction R^2 (OLS): {r2_lsm_recon:.4f}")
