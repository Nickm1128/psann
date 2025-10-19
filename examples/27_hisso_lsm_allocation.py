"""HISSO with an LSM expander on synthetic prices."""

import numpy as np

from psann import LSMExpander, PSANNRegressor, portfolio_log_return_reward
from psann.hisso import hisso_evaluate_reward, hisso_infer_series


def make_prices(T: int = 4096, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 48.0, T, dtype=np.float32)
    p1 = 80.0 * np.exp(0.0012 * t + 0.05 * np.sin(0.28 * t)) * (1.0 + 0.010 * rng.randn(T))
    p2 = 95.0 * np.exp(0.001 * t + 0.04 * np.cos(0.19 * t + 0.7)) * (1.0 + 0.011 * rng.randn(T))
    p3 = 70.0 * np.exp(0.0008 * t + 0.06 * np.sin(0.31 * t + 1.4)) * (1.0 + 0.012 * rng.randn(T))
    return np.stack([p1, p2, p3], axis=1).astype(np.float32)


def split_series(X: np.ndarray, n_train: int, n_val: int):
    train = X[:n_train]
    val = X[n_train : n_train + n_val]
    test = X[n_train + n_val :]
    return train, val, test


if __name__ == "__main__":
    prices = make_prices()
    train, val, test = split_series(prices, 2560, 768)

    hisso_window = 64
    trans_cost = 1e-3

    print("Pretraining LSM expander on inputs...")
    lsm = LSMExpander(
        output_dim=192,
        hidden_layers=6,
        hidden_width=192,
        sparsity=0.9,
        nonlinearity="sine",
        epochs=50,
        lr=8e-4,
        ridge=1e-4,
        random_state=1,
    )
    lsm.fit(train)

    print("First HISSO run with frozen LSM features...")
    est = PSANNRegressor(
        hidden_layers=2,
        hidden_width=72,
        activation_type="psann",
        epochs=60,
        lr=6e-4,
        batch_size=128,
        random_state=1,
        lsm=lsm,
        lsm_train=False,
    )
    est.fit(
        train,
        y=None,
        hisso=True,
        hisso_window=hisso_window,
        hisso_transition_penalty=trans_cost,
        hisso_reward_fn=lambda alloc, ctx: portfolio_log_return_reward(
            alloc, ctx, trans_cost=trans_cost
        ),
        verbose=1,
    )

    reward_before = hisso_evaluate_reward(est, test)
    print(f"Test log-return per episode after first run: {reward_before:.4f}")

    print("Continuing training with cached HISSO state...")
    est.epochs = 40
    est.lr = 4e-4
    est.fit(
        train,
        y=None,
        hisso=True,
        hisso_window=hisso_window,
        hisso_transition_penalty=trans_cost,
        hisso_reward_fn=lambda alloc, ctx: portfolio_log_return_reward(
            alloc, ctx, trans_cost=trans_cost
        ),
        verbose=0,
    )

    reward_after = hisso_evaluate_reward(est, test)
    alloc_test = hisso_infer_series(est, test)

    print(f"Test log-return per episode after continuation: {reward_after:.4f}")
    print("Allocation sample (first 5 steps):")
    print(np.round(alloc_test[:5], 3))
