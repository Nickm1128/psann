"""Episode training example: PSANN vs LSM+PSANN on two-asset allocation.

Optimizes allocations over episodes to maximize cumulative log return with
small transaction cost.
"""

import numpy as np
from psann import (
    PSANNRegressor,
    LSMExpander,
    EpisodeConfig,
    make_episode_trainer_from_estimator,
)


def make_prices(T=6000, seed=0):
    rs = np.random.RandomState(seed)
    t = np.linspace(0, 60, T)
    p1 = 100 * np.exp(0.0008 * t + 0.05 * np.sin(0.2 * t)) * (1 + 0.01 * rs.randn(T))
    p2 = 80 * np.exp(0.0005 * t + 0.08 * np.cos(0.15 * t)) * (1 + 0.012 * rs.randn(T))
    return np.stack([p1, p2], axis=1).astype(np.float32)


if __name__ == "__main__":
    prices = make_prices()
    X = prices  # features are the prices themselves here
    M = X.shape[1]

    # Baseline PSANN producing M allocations
    base = PSANNRegressor(hidden_layers=2, hidden_width=64, epochs=1, output_shape=(M,))
    y_dummy = np.zeros((len(X), M), dtype=np.float32)
    base.fit(X, y_dummy)

    cfg = EpisodeConfig(
        episode_length=64,
        batch_episodes=32,
        transition_penalty=0.001,
        random_state=0,
    )
    tr_base = make_episode_trainer_from_estimator(base, ep_cfg=cfg, lr=1e-3)
    print("[Base] Before:", tr_base.evaluate(X, n_batches=8))
    tr_base.train(X, epochs=50, verbose=1)
    print("[Base] After:", tr_base.evaluate(X, n_batches=8))

    # LSM + PSANN
    lsm = LSMExpander(output_dim=64, hidden_layers=2, hidden_width=64, sparsity=0.9, epochs=0)
    with_lsm = PSANNRegressor(hidden_layers=2, hidden_width=64, epochs=1, output_shape=(M,), lsm=lsm, lsm_train=False)
    with_lsm.fit(X, y_dummy)

    tr_lsm = make_episode_trainer_from_estimator(with_lsm, ep_cfg=cfg, lr=1e-3)
    print("[LSM] Before:", tr_lsm.evaluate(X, n_batches=8))
    tr_lsm.train(X, epochs=50, verbose=1)
    print("[LSM] After:", tr_lsm.evaluate(X, n_batches=8))
