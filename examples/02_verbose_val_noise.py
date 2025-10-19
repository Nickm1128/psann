import numpy as np
from sklearn.model_selection import train_test_split

from psann import PSANNRegressor


def make_data(n=2000, seed=42):
    rs = np.random.RandomState(seed)
    X = np.linspace(-4, 4, n).reshape(-1, 1).astype(np.float32)
    y = 0.8 * np.exp(-0.25 * np.abs(X)) * np.sin(3.5 * X) + 0.05 * rs.randn(*X.shape)
    return X, y.astype(np.float32)


if __name__ == "__main__":
    X, y = make_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = PSANNRegressor(
        hidden_layers=2,
        hidden_width=64,
        epochs=200,
        lr=1e-3,
        batch_size=128,
        early_stopping=True,
        patience=20,
    )

    # Verbose training with validation and Gaussian input noise
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        verbose=1,
        noisy=0.02,
    )

    print("R^2 on val:", model.score(X_val, y_val))
