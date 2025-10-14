import numpy as np
from psann import PSANNRegressor, EpisodeConfig, make_episode_trainer_from_estimator

# Synthetic two-asset price series
def make_prices(T=4000, seed=0):
    rs = np.random.RandomState(seed)
    t = np.linspace(0, 50, T)
    p1 = 100 * np.exp(0.001 * t + 0.05 * np.sin(0.2 * t)) * (1 + 0.01 * rs.randn(T))
    p2 = 80 * np.exp(0.0005 * t + 0.08 * np.cos(0.15 * t)) * (1 + 0.012 * rs.randn(T))
    return np.stack([p1, p2], axis=1).astype(np.float32)

if __name__ == "__main__":
    prices = make_prices()
    # Build a PSANN producing 2 allocations (for 2 assets)
    est = PSANNRegressor(hidden_layers=2, hidden_width=32, epochs=1, # we'll train with episodes
                         output_shape=(2,), stateful=False)
    # Initialize underlying model by a dummy fit with zero targets
    X = prices
    y = np.zeros((len(X), 2), dtype=np.float32)
    est.fit(X, y)

    cfg = EpisodeConfig(
        episode_length=64,
        batch_episodes=32,
        allocation_transform="softmax",
        transition_penalty=0.001,
        random_state=0,
    )
    trainer = make_episode_trainer_from_estimator(est, ep_cfg=cfg, lr=1e-3)
    print("Before training, eval reward:", trainer.evaluate(prices, n_batches=8))
    trainer.train(prices, epochs=50, verbose=1)
    print("After training, eval reward:", trainer.evaluate(prices, n_batches=8))
