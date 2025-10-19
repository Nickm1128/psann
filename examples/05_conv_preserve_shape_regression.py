import numpy as np

from psann import PSANNRegressor


def make_image_regression(n=400, c=1, h=8, w=8, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.randn(n, c, h, w).astype(np.float32)
    # Vector target from image statistics
    s = X.reshape(n, -1).mean(axis=1, keepdims=True)
    y = np.cos(2.0 * s) * np.exp(-0.2 * np.abs(s))
    return X, y.astype(np.float32)


if __name__ == "__main__":
    X, y = make_image_regression()
    n = len(X)
    idx = np.arange(n)
    np.random.RandomState(123).shuffle(idx)
    tr = idx[: int(0.8 * n)]
    te = idx[int(0.8 * n) :]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    # Convolutional PSANN keeps (N,C,H,W) shape through the body
    model = PSANNRegressor(
        preserve_shape=True,
        data_format="channels_first",
        hidden_layers=2,
        hidden_width=32,
        conv_kernel_size=3,
        epochs=80,
        early_stopping=True,
        patience=10,
    )

    model.fit(X_train, y_train, verbose=1)
    print("R^2:", model.score(X_test, y_test))
