from __future__ import annotations

import numpy as np

from psann import PSANNRegressor, ResPSANNRegressor
from psann.conv import PSANNConv2dNet, ResidualPSANNConv2dNet


def _make_image_data(seed: int, batch: int = 12, channels: int = 3, height: int = 8, width: int = 8):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((batch, channels, height, width)).astype(np.float32)
    y = rng.standard_normal((batch, 1)).astype(np.float32)
    return X, y


def test_with_conv_stem_dense_fit_predict() -> None:
    X, y = _make_image_data(seed=0)
    est = PSANNRegressor.with_conv_stem(
        hidden_layers=1,
        hidden_units=16,
        conv_channels=12,
        conv_kernel_size=3,
        epochs=2,
        batch_size=6,
        lr=1e-3,
        random_state=1,
    )

    est.fit(X, y, verbose=0)
    model = est.model_
    if hasattr(model, "core"):
        model = model.core
    assert isinstance(model, PSANNConv2dNet)

    preds = est.predict(X[:2])
    assert preds.shape == (2, 1)
    assert np.isfinite(preds).all()


def test_with_conv_stem_per_element_outputs_match_spatial_shape() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 2, 6, 6)).astype(np.float32)
    y = rng.standard_normal((10, 1, 6, 6)).astype(np.float32)

    est = PSANNRegressor.with_conv_stem(
        per_element=True,
        conv_channels=8,
        conv_kernel_size=3,
        hidden_layers=2,
        epochs=1,
        batch_size=5,
        lr=2e-3,
        random_state=2,
    )

    est.fit(X, y, verbose=0)
    preds = est.predict(X[:3])
    assert preds.shape == (3, 1, 6, 6)
    assert np.isfinite(preds).all()


def test_residual_with_conv_stem_uses_residual_conv_net() -> None:
    X, y = _make_image_data(seed=3)
    est = ResPSANNRegressor.with_conv_stem(
        hidden_layers=2,
        hidden_units=24,
        conv_channels=16,
        conv_kernel_size=3,
        epochs=2,
        batch_size=6,
        lr=1.5e-3,
        random_state=4,
    )

    est.fit(X, y, verbose=0)
    model = est.model_
    if hasattr(model, "core"):
        model = model.core
    assert isinstance(model, ResidualPSANNConv2dNet)

    preds = est.predict(X[:2])
    assert preds.shape == (2, 1)
    assert np.isfinite(preds).all()
