"""Unsupervised HISSO allocation on synthetic prices."""

import numpy as np

from psann import PSANNRegressor, portfolio_log_return_reward
from psann.hisso import hisso_evaluate_reward, hisso_infer_series


def make_prices(T: int = 4096, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 48.0, T, dtype=np.float32)
    p1 = 120.0 * np.exp(0.0015 * t + 0.04 * np.sin(0.25 * t)) * (1.0 + 0.009 * rng.randn(T))
    p2 = 90.0 * np.exp(0.001 * t + 0.06 * np.cos(0.21 * t)) * (1.0 + 0.011 * rng.randn(T))
    p3 = 60.0 * np.exp(0.0006 * t + 0.07 * np.sin(0.33 * t + 1.2)) * (1.0 + 0.012 * rng.randn(T))
    return np.stack([p1, p2, p3], axis=1).astype(np.float32)


if __name__ == "__main__":
    prices = make_prices()
    n_train = 2560
    n_val = 768
    train = prices[:n_train]
    val = prices[n_train:n_train + n_val]
    test = prices[n_train + n_val :]

    hisso_window = 64
    trans_cost = 1e-3

    print("Training HISSO policy on synthetic prices...")
    est = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        activation_type="psann",
        epochs=70,
        lr=8e-4,
        batch_size=128,
        random_state=0,
    )

    est.fit(
        train,
        y=None,
        hisso=True,
        hisso_window=hisso_window,
        hisso_transition_penalty=trans_cost,
        hisso_reward_fn=lambda alloc, ctx: portfolio_log_return_reward(alloc, ctx, trans_cost=trans_cost),
        verbose=1,
    )

    val_reward = hisso_evaluate_reward(est, val)
    test_reward = hisso_evaluate_reward(est, test)
    alloc_test = hisso_infer_series(est, test)

    print(f"Validation log-return per episode: {val_reward:.4f}")
    print(f"Test log-return per episode:       {test_reward:.4f}")
    print("Allocation sample (first 5 steps):")
    print(np.round(alloc_test[:5], 3))
