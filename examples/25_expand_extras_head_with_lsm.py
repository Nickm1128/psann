"""Expand a fitted PSANN+LSM regressor to supervise extras."""

import numpy as np

from psann import LSMExpander, PSANNRegressor, expand_extras_head


def make_dataset(n: int = 6000, seed: int = 0):
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


def split(X, y, extras, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 13):
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
    X, y, extras = make_dataset()
    (X_tr, y_tr, e_tr), (X_va, y_va, e_va), (X_te, y_te, e_te) = split(X, y, extras)

    print("Pretraining LSM expander on inputs...")
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

    print("Fitting base PSANN on primary target only...")
    base = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=120,
        lr=8e-4,
        batch_size=256,
        extras=0,
        early_stopping=True,
        patience=12,
        random_state=0,
        lsm=lsm,
        lsm_train=False,
    )
    base.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        verbose=1,
    )

    base_primary_rmse = rmse(base.predict(X_te).reshape(-1, 1), y_te)
    print(f"Base model primary RMSE: {base_primary_rmse:.4f}")

    extras_dim = e_tr.shape[1]
    print(f"\nExpanding extras head to dimension {extras_dim} and fine-tuning...")
    expanded = expand_extras_head(base, new_extras_dim=extras_dim)
    expanded.epochs = 80
    expanded.lr = 6e-4
    expanded.batch_size = 256
    expanded.set_extras_warm_start_epochs(8, freeze_until_plateau=False)

    expanded.fit(
        X_tr,
        y_tr,
        extras_targets=e_tr,
        validation_data=(X_va, y_va),
        verbose=1,
    )

    preds = expanded.predict(X_te)
    primary_preds = preds[:, : y_te.shape[1]]
    extras_preds = preds[:, y_te.shape[1] :]

    primary_rmse = rmse(primary_preds, y_te)
    extras_rmse = rmse(extras_preds, e_te)

    print("\nMetrics after extras fine-tuning:")
    print(f"- Primary RMSE: {primary_rmse:.4f}")
    print(f"- Extras  RMSE: {extras_rmse:.4f}")

    rollout_primary, rollout_extras = expanded.supervised_extras_rollout(X_te[:10])
    print("\nFirst rollout primary predictions:", np.round(rollout_primary[:5], 3))
    print("First rollout extras cache sample:", np.round(rollout_extras[:5], 3))
