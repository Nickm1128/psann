from __future__ import annotations

import numpy as np
import pytest

from psann import PSANNRegressor, ResConvPSANNRegressor, WaveResNetRegressor


def _dense_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((48, 4)).astype(np.float32)
    y = (0.35 * X[:, 0] - 0.2 * X[:, 1] + 0.15 * X[:, 2] * X[:, 3]).astype(np.float32)
    return X, y


def _conv_dataset(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((24, 1, 4, 4)).astype(np.float32)
    y = X.mean(axis=(1, 2, 3), keepdims=False).reshape(-1, 1).astype(np.float32)
    return X, y


def test_alias_characterization_covers_constructor_and_set_params() -> None:
    with pytest.warns(DeprecationWarning, match="hidden_width.*deprecated"):
        dense = PSANNRegressor(hidden_width=12)
    assert dense.hidden_units == 12
    assert dense.hidden_width == 12

    with pytest.warns(UserWarning, match="hidden_units` overrides `hidden_width`"):
        preferred = PSANNRegressor(hidden_units=16, hidden_width=32)
    assert preferred.hidden_units == 16
    assert preferred.hidden_width == 16

    conv = ResConvPSANNRegressor(hidden_layers=1, hidden_units=6, epochs=1)
    with pytest.warns(DeprecationWarning, match="hidden_channels.*deprecated"):
        conv.set_params(hidden_channels=10)
    assert conv.conv_channels == 10


def test_resconv_save_load_roundtrip_preserves_predictions(tmp_path) -> None:
    X, y = _conv_dataset(seed=4)

    model = ResConvPSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        conv_channels=8,
        epochs=4,
        batch_size=4,
        lr=5e-3,
        early_stopping=True,
        patience=2,
        random_state=4,
    )
    model.fit(X[:16], y[:16], validation_data=(X[16:20], y[16:20]), verbose=0)

    preds_before = model.predict(X[20:])

    checkpoint_path = tmp_path / "resconv_characterization.pt"
    model.save(str(checkpoint_path))
    restored = ResConvPSANNRegressor.load(str(checkpoint_path))

    preds_after = restored.predict(X[20:])
    np.testing.assert_allclose(preds_after, preds_before, rtol=1e-6, atol=1e-6)

    first_param = next(restored.model_.parameters())
    assert first_param.device.type == "cpu"


def test_wave_resnet_accepts_validation_context_triple() -> None:
    X, y = _dense_dataset(seed=9)
    context = np.stack([X[:, 0], X[:, 1]], axis=1).astype(np.float32)

    model = WaveResNetRegressor(
        hidden_layers=2,
        hidden_units=12,
        epochs=6,
        batch_size=8,
        lr=4e-3,
        context_dim=2,
        random_state=9,
    )
    model.fit(
        X[:28],
        y[:28].reshape(-1, 1),
        context=context[:28],
        validation_data=(X[28:38], y[28:38].reshape(-1, 1), context[28:38]),
        verbose=0,
    )

    preds = model.predict(X[38:], context=context[38:])
    assert preds.shape == (10, 1)
    assert model._context_dim_ == 2
