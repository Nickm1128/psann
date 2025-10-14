"""Episode training with Conv PSANN on synthetic spatial features.

We simulate two assets whose "price maps" are noisy spatial grids. The model
is a Conv2d PSANN with a pooled (vector) head producing two allocations. We
provide a price_extractor that converts episode tensors into (B,T,M) prices by
global averaging spatial dimensions per channel.
"""

import numpy as np
import torch
from psann import PSANNRegressor, EpisodeConfig, make_episode_trainer_from_estimator


def make_spatial_prices(T=4000, H=8, W=8, seed=1):
    rs = np.random.RandomState(seed)
    t = np.linspace(0, 50, T)
    p1 = 100 * np.exp(0.0008 * t + 0.05 * np.sin(0.2 * t)) * (1 + 0.01 * rs.randn(T))
    p2 = 80 * np.exp(0.0005 * t + 0.08 * np.cos(0.15 * t)) * (1 + 0.012 * rs.randn(T))
    # Create spatial maps per asset by adding spatially varying noise
    X = np.zeros((T, 2, H, W), dtype=np.float32)
    for i in range(T):
        base1 = p1[i] * (1 + 0.02 * rs.randn(H, W))
        base2 = p2[i] * (1 + 0.02 * rs.randn(H, W))
        X[i, 0] = base1
        X[i, 1] = base2
    return X.astype(np.float32)


def price_extractor_channels_first(X_ep: torch.Tensor) -> torch.Tensor:
    # X_ep: (B, T, C, H, W) -> prices: (B, T, C)
    return X_ep.mean(dim=(-1, -2))


if __name__ == "__main__":
    X = make_spatial_prices()
    T, C, H, W = X.shape
    M = C

    est = PSANNRegressor(
        hidden_layers=2,
        hidden_width=24,
        epochs=1,
        preserve_shape=True,
        data_format="channels_first",
        conv_kernel_size=3,
        per_element=False,
        output_shape=(M,),
    )
    # Initialize by one pass fit
    y_dummy = np.zeros((T, M), dtype=np.float32)
    est.fit(X, y_dummy)

    cfg = EpisodeConfig(
        episode_length=32,
        batch_episodes=32,
        transition_penalty=0.001,
        spatial_pool="mean",
        random_state=0,
    )
    trainer = make_episode_trainer_from_estimator(
        est, ep_cfg=cfg, lr=1e-3
    )
    # Attach price extractor
    trainer.price_extractor = price_extractor_channels_first

    print("[Conv] Before:", trainer.evaluate(X, n_batches=8))
    trainer.train(X, epochs=50, verbose=1)
    print("[Conv] After:", trainer.evaluate(X, n_batches=8))
