import numpy as np
import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor
from psann.lsm import LSMConv2dExpander, LSMExpander


def _make_dense_regression(seed: int = 0):
    rs = np.random.RandomState(seed)
    X = rs.randn(48, 3).astype(np.float32)
    w = rs.randn(3, 1).astype(np.float32)
    y = (X @ w + 0.05 * rs.randn(48, 1)).astype(np.float32)
    return X, y


def _make_conv_regression(seed: int = 1):
    rs = np.random.RandomState(seed)
    N, C, H, W = 18, 1, 5, 5
    X = rs.randn(N, C, H, W).astype(np.float32)
    # Simple nonlinear per-pixel target that preserves shape
    y = np.tanh(0.6 * X).astype(np.float32)
    return X, y


@pytest.mark.parametrize("lsm_train", [False, True])
def test_regressor_with_lsm_expander(lsm_train):
    X, y = _make_dense_regression()
    expander = LSMExpander(
        output_dim=12,
        hidden_layers=1,
        hidden_units=16,
        epochs=1,
        lr=5e-3,
        batch_size=12,
        random_state=123,
    )
    if not lsm_train:
        expander.fit(X, epochs=1)

    reg = PSANNRegressor(
        hidden_layers=1,
        hidden_units=24,
        epochs=8,
        batch_size=12,
        lr=5e-3,
        early_stopping=False,
        lsm=expander,
        lsm_train=lsm_train,
        lsm_pretrain_epochs=1,
        random_state=42,
    )

    reg.fit(X, y)
    preds = reg.predict(X[:10])
    assert preds.shape == y[:10].shape
    assert np.isfinite(preds).all()

    preproc = reg.model_.preproc
    if not lsm_train:
        assert all(not p.requires_grad for p in preproc.parameters())
    else:
        assert any(p.requires_grad for p in preproc.parameters())


@pytest.mark.parametrize("lsm_train", [False, True])
def test_regressor_with_lsmconv2d_expander(lsm_train):
    X, y = _make_conv_regression()
    expander = LSMConv2dExpander(
        out_channels=2,
        hidden_layers=1,
        hidden_channels=8,
        epochs=1,
        lr=5e-3,
        random_state=77,
    )
    if not lsm_train:
        expander.fit(X, epochs=1)

    reg = PSANNRegressor(
        preserve_shape=True,
        data_format="channels_first",
        per_element=True,
        hidden_layers=1,
        hidden_units=24,
        conv_channels=12,
        epochs=5,
        batch_size=6,
        lr=5e-3,
        early_stopping=False,
        lsm=expander,
        lsm_train=lsm_train,
        lsm_pretrain_epochs=1,
        random_state=99,
    )

    reg.fit(X, y)
    preds = reg.predict(X[:4])
    assert preds.shape == y[:4].shape
    assert np.isfinite(preds).all()

    conv_preproc = reg.model_.preproc
    score_source = conv_preproc if lsm_train else expander
    score = score_source.score_reconstruction(X)
    assert np.isfinite(score)
