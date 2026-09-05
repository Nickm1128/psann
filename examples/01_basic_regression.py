import numpy as np

from psann.architectures import ActivationConfig, ArchitectureConfig
from psann import PSANNRegressor


def make_data(n=2000, seed=42):
    rs = np.random.RandomState(seed)
    X = np.linspace(-4, 4, n).reshape(-1, 1).astype(np.float32)
    y = 0.8 * np.exp(-0.25 * np.abs(X)) * np.sin(3.5 * X) + 0.05 * rs.randn(*X.shape)
    return X, y.astype(np.float32)


if __name__ == "__main__":
    X, y = make_data()
    n = len(X)
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    tr = idx[: int(0.8 * n)]
    te = idx[int(0.8 * n) :]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    model = PSANNRegressor(
        architecture=ArchitectureConfig.dense(
            activation=ActivationConfig(
                **{
                    "amplitude_init": 1.0,
                    "frequency_init": 1.0,
                    "decay_init": 0.1,
                    "learnable": ("amplitude", "frequency", "decay"),
                    "decay_mode": "abs",
                }
            )
        ),
        hidden_layers=2,
        epochs=300,
        lr=0.001,
        batch_size=128,
        device="auto",
        random_state=42,
        early_stopping=True,
        patience=30,
        hidden_units=64,
    )

    model.fit(X_train, y_train)
    r2 = model.score(X_test, y_test)
    print(f"R^2: {r2:.4f}")
