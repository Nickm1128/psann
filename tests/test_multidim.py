import numpy as np
import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor


def test_fit_predict_with_3d_input():
    # Create simple 3D inputs (C,H,W) per sample; function of sum of pixels
    n, C, H, W = 256, 1, 4, 4
    rs = np.random.RandomState(0)
    X = rs.randn(n, C, H, W).astype(np.float32)
    # Target: nonlinear function of flattened sum to keep it simple
    s = X.reshape(n, -1).sum(axis=1, keepdims=True).astype(np.float32)
    y = np.sin(s) * np.exp(-0.1 * np.abs(s))

    model = PSANNRegressor(
        epochs=25,
        hidden_layers=2,
        hidden_width=32,
        early_stopping=True,
        patience=10,
        preserve_shape=True,
        data_format="channels_first",
    )
    model.fit(X, y, verbose=0)
    preds = model.predict(X[:10])
    assert preds.shape[0] == 10

    # Ensure save/load retains input shape using a local temp file
    path = "psann_tmp.pt"
    model.save(path)
    loaded = PSANNRegressor.load(path)
    preds2 = loaded.predict(X[:10])
    assert preds2.shape == preds.shape
    import os
    try:
        os.remove(path)
    except OSError:
        pass
