import numpy as np

from psann import GeoSparseRegressor


def make_data(n: int = 512, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 4, 4)).astype(np.float32)
    y = np.sin(X.reshape(n, -1).sum(axis=1, keepdims=True)).astype(np.float32)
    return X, y


if __name__ == "__main__":
    X, y = make_data()
    idx = np.arange(len(X))
    np.random.default_rng(1).shuffle(idx)
    split = int(0.8 * len(idx))
    tr_idx, te_idx = idx[:split], idx[split:]
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_test, y_test = X[te_idx], y[te_idx]

    model = GeoSparseRegressor(
        shape=(4, 4),
        hidden_layers=4,
        k=8,
        activation_type="relu",
        epochs=60,
        batch_size=64,
        lr=1e-3,
        random_state=0,
    )
    model.fit(X_train, y_train, verbose=0)
    print("R^2:", model.score(X_test, y_test))
