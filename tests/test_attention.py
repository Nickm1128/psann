import numpy as np
import pytest

from psann import PSANNRegressor, WaveResNetRegressor


def _make_sequence_data(n: int = 24, steps: int = 4, feats: int = 3):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, steps, feats)).astype(np.float32)
    y = X.mean(axis=(1, 2), keepdims=False).astype(np.float32).reshape(n, 1)
    return X, y


def test_psann_attention_fit_and_predict():
    X, y = _make_sequence_data()
    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=16,
        epochs=2,
        batch_size=8,
        attention={"kind": "mha", "num_heads": 1},
    )
    est.fit(X, y, verbose=0)
    preds = est.predict(X)
    assert preds.shape == y.shape


def test_psann_attention_rejects_lsm():
    X, y = _make_sequence_data()
    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        attention={"kind": "mha"},
        lsm={"type": "lsmexpander", "output_dim": 16},
    )
    with pytest.raises(ValueError):
        est.fit(X, y, verbose=0)


def test_waveresnet_attention_basic_fit():
    X, y = _make_sequence_data(n=16, steps=6, feats=2)
    est = WaveResNetRegressor(
        hidden_layers=2,
        hidden_units=16,
        epochs=2,
        batch_size=8,
        attention={"kind": "mha", "num_heads": 2},
    )
    est.fit(X, y, verbose=0)
    preds = est.predict(X)
    assert preds.shape == y.shape


def test_conv_attention_preserve_shape():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 3, 4, 4)).astype(np.float32)
    y = X.mean(axis=(1, 2, 3), keepdims=False).astype(np.float32).reshape(12, 1)
    est = PSANNRegressor(
        preserve_shape=True,
        conv_channels=8,
        hidden_layers=1,
        epochs=2,
        batch_size=4,
        attention={"kind": "mha", "num_heads": 2},
    )
    est.fit(X, y, verbose=0)
    preds = est.predict(X)
    assert preds.shape == y.shape


def test_conv_attention_per_element_output():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(6, 2, 3, 3)).astype(np.float32)
    y = X.mean(axis=1, keepdims=True).astype(np.float32)
    est = PSANNRegressor(
        preserve_shape=True,
        per_element=True,
        conv_channels=6,
        hidden_layers=1,
        epochs=1,
        batch_size=3,
        attention={"kind": "mha", "num_heads": 1},
    )
    est.fit(X, y, verbose=0)
    preds = est.predict(X)
    assert preds.shape == y.shape


def test_waveresnet_conv_attention_via_helper():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, 8, 3)).astype(np.float32)  # (N, length, channels)
    y = X.mean(axis=(1, 2), keepdims=False).astype(np.float32).reshape(10, 1)
    est = WaveResNetRegressor.with_conv_stem(
        conv_channels=16,
        conv_kernel_size=3,
        data_format="channels_last",
        hidden_layers=2,
        hidden_units=16,
        epochs=2,
        batch_size=5,
        attention={"kind": "mha", "num_heads": 2},
    )
    est.fit(X, y, verbose=0)
    preds = est.predict(X)
    assert preds.shape == y.shape
