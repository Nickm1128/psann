import numpy as np
import torch

from psann import PSANNRegressor


def make_data(n=1500, seed=0):
    rs = np.random.RandomState(seed)
    X = np.linspace(-3.5, 3.5, n).reshape(-1, 1).astype(np.float32)
    y = 0.9 * np.exp(-0.3 * np.abs(X)) * np.sin(4.0 * X) + 0.05 * rs.randn(*X.shape)
    return X, y.astype(np.float32)


def mape_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    # Mean Absolute Percentage Error (vector output allowed)
    denom = torch.clamp(target.abs(), min=eps)
    return (pred - target).abs() / denom


if __name__ == "__main__":
    X, y = make_data()
    n = len(X)
    idx = np.arange(n)
    np.random.RandomState(1).shuffle(idx)
    tr = idx[: int(0.8 * n)]
    te = idx[int(0.8 * n) :]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    model = PSANNRegressor(
        hidden_layers=3,
        hidden_width=64,
        epochs=150,
        lr=1e-3,
        loss=mape_loss,
        loss_params={"eps": 1e-2},
        loss_reduction="mean",
        early_stopping=True,
        patience=20,
    )

    model.fit(X_train, y_train, verbose=1)

    r2 = model.score(X_test, y_test)
    print(f"R^2 (MAPE-trained): {r2:.4f}")

    # Note: custom loss functions are not serialized with save(); params will default to 'mse' on load
    path = "psann_custom_loss.pt"
    model.save(path)
    loaded = PSANNRegressor.load(path)
    print("Loaded model R^2 (evaluated with mse):", loaded.score(X_test, y_test))
