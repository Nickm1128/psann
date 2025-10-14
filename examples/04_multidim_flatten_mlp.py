import numpy as np

from psann import PSANNRegressor


def make_data(n=600, shape=(2, 4, 4), seed=0):
    rs = np.random.RandomState(seed)
    X = rs.randn(n, *shape).astype(np.float32)
    # Target: nonlinear function of flattened sum
    s = X.reshape(n, -1).sum(axis=1, keepdims=True)
    y = np.sin(s) * np.exp(-0.1 * np.abs(s))
    return X, y.astype(np.float32)


if __name__ == "__main__":
    X, y = make_data()
    n = len(X)
    idx = np.arange(n)
    np.random.RandomState(0).shuffle(idx)
    tr = idx[: int(0.8 * n)]
    te = idx[int(0.8 * n) :]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    # Default MLP path flattens features internally
    model = PSANNRegressor(
        hidden_layers=2,
        hidden_width=128,
        epochs=120,
        early_stopping=True,
        patience=15,
    )

    model.fit(X_train, y_train, verbose=1)
    print("R^2:", model.score(X_test, y_test))

