import numpy as np

from psann import PSANNRegressor


def test_readme_basic_regression_smoke():
    rs = np.random.RandomState(42)
    x = np.linspace(-2, 2, 128, dtype=np.float32).reshape(-1, 1)
    y = 0.8 * np.exp(-0.25 * np.abs(x)) * np.sin(3.5 * x)
    y = y + 0.01 * rs.randn(*y.shape).astype(np.float32)

    model = PSANNRegressor(
        hidden_layers=1,
        hidden_units=32,
        epochs=5,
        batch_size=32,
        lr=1e-3,
        early_stopping=False,
        random_state=42,
    )
    model.fit(x, y, verbose=0)
    preds = model.predict(x[:8])
    assert preds.shape[0] == 8
