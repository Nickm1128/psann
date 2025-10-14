import numpy as np
import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor


def test_segmentation_2d_channels_first():
    rs = np.random.RandomState(0)
    N, C, H, W = 32, 1, 8, 8
    X = rs.randn(N, C, H, W).astype(np.float32)
    # Target: per-pixel nonlinear function
    y = np.sin(X) * np.exp(-0.1 * np.abs(X))  # (N,1,H,W)

    model = PSANNRegressor(
        preserve_shape=True,
        data_format="channels_first",
        per_element=True,
        hidden_layers=2,
        hidden_units=16,
        epochs=20,
        early_stopping=True,
        patience=5,
    )
    model.fit(X, y)
    out = model.predict(X[:4])
    assert out.shape == (4, 1, H, W)


def test_segmentation_2d_channels_last():
    rs = np.random.RandomState(1)
    N, H, W, C = 24, 6, 6, 1
    X = rs.randn(N, H, W, C).astype(np.float32)
    y = np.cos(X)  # (N,H,W,1)

    model = PSANNRegressor(
        preserve_shape=True,
        data_format="channels_last",
        per_element=True,
        hidden_layers=1,
        hidden_units=16,
        epochs=10,
    )
    model.fit(X, y)
    out = model.predict(X[:3])
    assert out.shape == (3, H, W, 1)
