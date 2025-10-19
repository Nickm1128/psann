"""Joint PSANN + LSM supervised extras quickstart.

This script shows how to attach an LSMExpander to a PSANNRegressor while
training supervised extras alongside the primary regression target.
"""

import numpy as np

from psann import LSMExpander, PSANNRegressor


def make_supervised_extras_data(n: int = 6000, seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.5, 2.5, size=(n, 4)).astype(np.float32)
    latent = 0.6 * np.sin(X[:, 0]) + 0.4 * np.cos(1.5 * X[:, 1]) + 0.3 * X[:, 2] * X[:, 3]
    y = (latent + 0.2 * rng.randn(n)).astype(np.float32)
    extras = np.stack(
        [
            np.sin(latent) + 0.1 * rng.randn(n),
            np.cos(X[:, 2]) + 0.2 * latent,
        ],
        axis=1,
    ).astype(np.float32)
    return X, y.reshape(-1, 1).astype(np.float32), extras


def train_val_test_split(
    X, y, extras, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 1
):
    n = len(X)
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)

    n_test = int(test_ratio * n)
    n_val = int(val_ratio * n)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    return (
        (X[train_idx], y[train_idx], extras[train_idx]),
        (X[val_idx], y[val_idx], extras[val_idx]),
        (X[test_idx], y[test_idx], extras[test_idx]),
    )


def rmse(pred, target):
    return float(np.sqrt(np.mean(np.square(pred - target))))


if __name__ == "__main__":
    X, y, extras = make_supervised_extras_data()
    (X_tr, y_tr, e_tr), (X_va, y_va, e_va), (X_te, y_te, e_te) = train_val_test_split(X, y, extras)

    extras_dim = e_tr.shape[1]

    print("Training PSANN baseline with supervised extras...")
    baseline = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=120,
        lr=8e-4,
        batch_size=256,
        extras=extras_dim,
        early_stopping=True,
        patience=12,
        random_state=0,
    )
    baseline.set_extras_warm_start_epochs(10, freeze_until_plateau=False)
    baseline.fit(
        X_tr,
        y_tr,
        extras_targets=e_tr,
        validation_data=(X_va, y_va),
        verbose=1,
    )

    base_preds = baseline.predict(X_te)
    base_primary_rmse = rmse(base_preds[:, : y_te.shape[1]], y_te)
    base_extras_rmse = rmse(base_preds[:, y_te.shape[1] :], e_te)

    print(f"Baseline primary RMSE: {base_primary_rmse:.4f}")
    print(f"Baseline extras RMSE:   {base_extras_rmse:.4f}")

    print("\nPretraining LSM expander on inputs...")
    lsm = LSMExpander(
        output_dim=256,
        hidden_layers=10,
        hidden_width=256,
        sparsity=0.9,
        nonlinearity="sine",
        epochs=60,
        lr=1e-3,
        ridge=1e-4,
        random_state=0,
    )
    lsm.fit(X_tr)

    print("Training PSANN with frozen LSM features and supervised extras...")
    with_lsm = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=120,
        lr=8e-4,
        batch_size=256,
        extras=extras_dim,
        early_stopping=True,
        patience=12,
        random_state=0,
        lsm=lsm,
        lsm_train=False,
    )
    with_lsm.set_extras_warm_start_epochs(10, freeze_until_plateau=False)
    with_lsm.fit(
        X_tr,
        y_tr,
        extras_targets=e_tr,
        validation_data=(X_va, y_va),
        verbose=1,
    )

    lsm_preds = with_lsm.predict(X_te)
    lsm_primary_rmse = rmse(lsm_preds[:, : y_te.shape[1]], y_te)
    lsm_extras_rmse = rmse(lsm_preds[:, y_te.shape[1] :], e_te)

    print("\nResults on the held-out test split:")
    print(f"- Baseline primary RMSE: {base_primary_rmse:.4f}")
    print(f"- Baseline extras  RMSE: {base_extras_rmse:.4f}")
    print(f"- LSM     primary RMSE: {lsm_primary_rmse:.4f}")
    print(f"- LSM     extras  RMSE: {lsm_extras_rmse:.4f}")
