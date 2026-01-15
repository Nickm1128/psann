import numpy as np
import pytest

torch = pytest.importorskip("torch")

from psann import PSANNRegressor
from psann.estimators._fit_utils import (
    _prepare_validation_tensors,
    normalise_fit_args,
    prepare_inputs_and_scaler,
)


def make_regressor(**kwargs) -> PSANNRegressor:
    base = dict(hidden_layers=1, hidden_units=4, epochs=1, batch_size=2, random_state=0)
    base.update(kwargs)
    return PSANNRegressor(**base)


def test_preserve_shape_flat_validation_matches_training_layout():
    rs = np.random.RandomState(0)
    X = rs.randn(6, 4, 3, 2).astype(np.float32)
    y = rs.randn(6).astype(np.float32)
    X_val = rs.randn(3, 4, 3, 2).astype(np.float32)
    y_val = rs.randn(3).astype(np.float32)

    est = make_regressor(
        preserve_shape=True,
        per_element=False,
        data_format="channels_last",
        scaler="standard",
    )

    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=(X_val, y_val),
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)
    assert prepared.train_inputs.ndim == 2

    device = torch.device("cpu")
    val_inputs_t, val_targets_t, ctx_t = _prepare_validation_tensors(
        est, prepared, fit_args.validation, device=device
    )

    assert val_inputs_t is not None and val_targets_t is not None
    assert ctx_t is None
    assert val_inputs_t.ndim == 2
    assert val_inputs_t.shape[1] == prepared.train_inputs.shape[1]
    assert val_targets_t.shape == (X_val.shape[0], 1)

    X_val_cf = np.moveaxis(X_val, -1, 1)
    n_val, c_val = X_val_cf.shape[0], X_val_cf.shape[1]
    X_val_2d = X_val_cf.reshape(n_val, c_val, -1).transpose(0, 2, 1).reshape(-1, c_val)
    X_val_2d = est._apply_fitted_scaler(X_val_2d)
    X_val_cf_scaled = X_val_2d.reshape(n_val, -1, c_val).transpose(0, 2, 1).reshape(X_val_cf.shape)
    X_val_scaled = np.moveaxis(X_val_cf_scaled, 1, -1)
    expected_flat = est._flatten(X_val_scaled)

    assert np.allclose(val_inputs_t.numpy(), expected_flat, atol=1e-5)


@pytest.mark.parametrize("data_format", ["channels_last", "channels_first"])
def test_per_element_validation_respects_layout_and_channels(data_format: str):
    rs = np.random.RandomState(1)
    if data_format == "channels_last":
        X = rs.randn(4, 5, 6, 2).astype(np.float32)
        y = rs.randn(4, 5, 6, 3).astype(np.float32)
        X_val = rs.randn(2, 5, 6, 2).astype(np.float32)
        y_val = rs.randn(2, 5, 6, 3).astype(np.float32)
        expected_input_shape = (2, 2, 5, 6)
        expected_target_shape = (2, 3, 5, 6)
    else:
        X = rs.randn(4, 2, 5, 6).astype(np.float32)
        y = rs.randn(4, 3, 5, 6).astype(np.float32)
        X_val = rs.randn(2, 2, 5, 6).astype(np.float32)
        y_val = rs.randn(2, 3, 5, 6).astype(np.float32)
        expected_input_shape = (2, 2, 5, 6)
        expected_target_shape = (2, 3, 5, 6)

    est = make_regressor(
        preserve_shape=True,
        per_element=True,
        data_format=data_format,
    )

    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=(X_val, y_val),
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)
    assert prepared.train_inputs.ndim >= 3
    device = torch.device("cpu")

    val_inputs_t, val_targets_t, ctx_t = _prepare_validation_tensors(
        est, prepared, fit_args.validation, device=device
    )

    assert val_inputs_t.shape == expected_input_shape
    assert val_targets_t.shape == expected_target_shape
    assert ctx_t is None


def test_per_element_validation_raises_on_spatial_mismatch():
    rs = np.random.RandomState(2)
    X = rs.randn(3, 4, 5, 2).astype(np.float32)
    y = rs.randn(3, 4, 5, 1).astype(np.float32)
    X_val = rs.randn(2, 4, 5, 2).astype(np.float32)
    y_val_bad = rs.randn(2, 3, 5, 1).astype(np.float32)  # height mismatch

    est = make_regressor(
        preserve_shape=True,
        per_element=True,
        data_format="channels_last",
    )

    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=(X_val, y_val_bad),
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)

    with pytest.raises(ValueError, match="validation y spatial dimensions do not match X"):
        _prepare_validation_tensors(est, prepared, fit_args.validation, device=torch.device("cpu"))


def test_per_element_target_scaler_standard_scales_targets():
    rs = np.random.RandomState(3)
    X = rs.randn(4, 2, 3, 3).astype(np.float32)
    y = (10.0 + 5.0 * rs.randn(4, 1, 3, 3)).astype(np.float32)

    est = make_regressor(
        preserve_shape=True,
        per_element=True,
        data_format="channels_first",
        target_scaler="standard",
    )
    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=None,
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)
    assert prepared.y_cf is not None
    y_scaled = prepared.y_cf
    mean = y_scaled.mean(axis=(0, 2, 3))
    std = y_scaled.std(axis=(0, 2, 3))
    assert np.allclose(mean, 0.0, atol=1e-3)
    assert np.allclose(std, 1.0, atol=1e-3)
    state = est._target_scaler_state_
    assert state is not None
    assert state["n"] == int(y.shape[0] * y.shape[2] * y.shape[3])


def test_flat_validation_raises_on_target_mismatch():
    rs = np.random.RandomState(3)
    X = rs.randn(5, 3, 2, 1).astype(np.float32)
    y = rs.randn(5, 2).astype(np.float32)
    X_val = rs.randn(2, 3, 2, 1).astype(np.float32)
    y_val_bad = rs.randn(2, 1).astype(np.float32)  # expects 2 targets, receives 1

    est = make_regressor(
        preserve_shape=True,
        per_element=False,
        data_format="channels_last",
    )

    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=(X_val, y_val_bad),
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)

    with pytest.raises(ValueError, match="validation y target dimension mismatch"):
        _prepare_validation_tensors(est, prepared, fit_args.validation, device=torch.device("cpu"))


def test_flat_validation_respects_output_shape_constraint():
    rs = np.random.RandomState(4)
    X = rs.randn(4, 2, 3, 1).astype(np.float32)
    y = rs.randn(4, 2).astype(np.float32)
    X_val = rs.randn(2, 2, 3, 1).astype(np.float32)
    y_val = rs.randn(2, 2).astype(np.float32)

    est = make_regressor(
        preserve_shape=True,
        per_element=False,
        data_format="channels_last",
        output_shape=(2,),
    )

    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=(X_val, y_val),
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)
    assert int(prepared.output_dim) == 2

    val_inputs_t, val_targets_t, _ = _prepare_validation_tensors(
        est, prepared, fit_args.validation, device=torch.device("cpu")
    )

    assert val_inputs_t.ndim == 2
    assert val_targets_t.shape == (X_val.shape[0], 2)
