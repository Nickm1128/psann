"""Episode training example: PSANN vs LSM+PSANN on two-asset allocation.

Optimizes allocations over episodes to maximize cumulative log return with
small transaction cost.
"""

import numpy as np

from psann import PSANNRegressor
from psann.episodic import EpisodeScheduleConfig, EpisodicTrainer, HISSOConfig
from psann.preprocessing import LSMConfig, PreprocessorConfig


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
    cfg = HISSOConfig(
        schedule=EpisodeScheduleConfig(episode_length=64, batch_episodes=32, random_state=0),
        transition_penalty=0.001,
    )
    tr_base = EpisodicTrainer(estimator=base, strategy=cfg)
    tr_base.fit(X, verbose=1)
    print("[Base] After:", tr_base.evaluate(X, n_batches=8))

    # LSM + PSANN
    preprocessor = PreprocessorConfig(
        LSMConfig.dense(output_dim=64, hidden_layers=2, hidden_units=64, sparsity=0.9)
    )
    with_lsm = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=1,
        output_shape=(M,),
        preprocessor=preprocessor,
    )
    tr_lsm = EpisodicTrainer(estimator=with_lsm, strategy=cfg)
    tr_lsm.fit(X, verbose=1)
    print("[LSM] After:", tr_lsm.evaluate(X, n_batches=8))
