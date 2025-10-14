import numpy as np

from psann import PSANNRegressor


def make_segmentation(n=64, c=1, h=12, w=12, out_c=2, seed=0):
    rs = np.random.RandomState(seed)
    X = rs.randn(n, c, h, w).astype(np.float32)
    # Per-pixel multi-channel target (stacked sine/cosine-like maps)
    y1 = np.sin(X) * np.exp(-0.1 * np.abs(X))
    y2 = np.cos(X) * np.exp(-0.1 * np.abs(X))
    y = np.concatenate([y1, y2], axis=1)[:, :out_c]
    return X, y.astype(np.float32)


if __name__ == "__main__":
    # Two-channel per-pixel regression target
    X, y = make_segmentation(out_c=2)
    n = len(X)
    idx = np.arange(n)
    np.random.RandomState(4).shuffle(idx)
    tr = idx[: int(0.8 * n)]
    te = idx[int(0.8 * n) :]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    # Per-pixel outputs with segmentation head
    model = PSANNRegressor(
        preserve_shape=True,
        data_format="channels_first",
        per_element=True,
        output_shape=(2,),  # two output channels per pixel
        hidden_layers=2,
        hidden_width=24,
        conv_kernel_size=3,
        epochs=20,
        early_stopping=True,
        patience=5,
        loss="mse",
    )

    model.fit(X_train, y_train, verbose=1)
    Yhat = model.predict(X_test[:4])  # (N, 2, H, W)
    print("Pred shape:", Yhat.shape)
    # Quick quality check (MSE over a small batch)
    mse = ((Yhat - y_test[:4]) ** 2).mean()
    print(f"Batch MSE: {mse:.6f}")
