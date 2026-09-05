"""HISSO with an LSM expander on synthetic prices."""

import numpy as np

from psann.architectures import ActivationConfig, ArchitectureConfig
from psann import PSANNRegressor
from psann.episodic import EpisodeScheduleConfig, EpisodicTrainer, HISSOConfig
from psann.preprocessing import LSMConfig, LSMPretrainingConfig, PreprocessorConfig


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

    preprocessor = PreprocessorConfig(
        LSMConfig.dense(
            output_dim=192,
            hidden_layers=6,
            hidden_units=192,
            sparsity=0.9,
            nonlinearity="sine",
            pretraining=LSMPretrainingConfig(epochs=50, lr=8e-4, ridge=1e-4),
            random_state=1,
        )
    )

    print("First HISSO run with frozen LSM preprocessing...")
    est = PSANNRegressor(
        output_shape=(train.shape[1],),
        architecture=ArchitectureConfig.dense(activation=ActivationConfig(kind="psann")),
        hidden_layers=2,
        epochs=60,
        lr=0.0006,
        batch_size=128,
        random_state=1,
        preprocessor=preprocessor,
        hidden_units=72,
    )
    trainer = EpisodicTrainer(
        estimator=est,
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=hisso_window),
            reward="finance",
            primary_transform="softmax",
            transition_penalty=trans_cost,
        ),
    )
    trainer.fit(train, verbose=1)

    reward_before = trainer.evaluate(test)
    print(f"Test log-return per episode after first run: {reward_before:.4f}")

    print("Continuing training with cached HISSO state...")
    est.epochs = 40
    est.lr = 4e-4
    trainer.fit(train, verbose=0)

    reward_after = trainer.evaluate(test)
    alloc_test = trainer.predict(test)

    print(f"Test log-return per episode after continuation: {reward_after:.4f}")
    print("Allocation sample (first 5 steps):")
    print(np.round(alloc_test[:5], 3))
