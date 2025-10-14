from __future__ import annotations

import numpy as np

from psann import PSANNRegressor
from psann.estimators._fit_utils import NormalisedFitArgs, normalise_fit_args, prepare_inputs_and_scaler


def _make_fit_args(
    estimator: PSANNRegressor,
    X: np.ndarray,
    y: np.ndarray | None,
    *,
    validation_data=None,
    hisso: bool = False,
    noisy=None,
) -> NormalisedFitArgs:
    return normalise_fit_args(
        estimator,
        X,
        y,
        validation_data=validation_data,
        noisy=noisy,
        verbose=1,
        lr_max=0.05,
        lr_min=0.001,
        hisso=hisso,
        hisso_kwargs={"hisso_window": 16},
    )


def test_normalise_fit_args_coerces_validation_pair() -> None:
    est = PSANNRegressor(hidden_layers=1, hidden_units=4, epochs=1, batch_size=8)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((12, 3))
    y = rng.standard_normal((12, 2))
    X_val = rng.standard_normal((6, 3))
    y_val = rng.standard_normal((6, 2))

    fit_args = _make_fit_args(est, X, y, validation_data=(X_val, y_val))

    assert fit_args.X.dtype == np.float32
    assert fit_args.y is not None and fit_args.y.dtype == np.float32
    assert fit_args.validation is not None
    X_v, y_v = fit_args.validation
    assert X_v.shape == X_val.shape
    assert X_v.dtype == np.float32
    assert y_v.shape == y_val.shape
    assert y_v.dtype == np.float32
    assert fit_args.hisso is False
    assert fit_args.lr_max == 0.05
    assert fit_args.lr_min == 0.001


def test_prepare_inputs_and_scaler_flatten_flow() -> None:
    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=8,
        scaler="standard",
    )
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 5))
    y = rng.standard_normal((10, 1))

    fit_args = _make_fit_args(est, X, y)
    prepared, primary_dim = prepare_inputs_and_scaler(est, fit_args)

    assert primary_dim == 1
    assert prepared.train_inputs.shape == (10, 5)
    assert prepared.train_targets is not None
    assert prepared.train_targets.shape == (10, 1)
    assert prepared.scaler_transform is not None
    # Ensure scaler transform centres roughly around zero for the fitted data.
    transformed = prepared.scaler_transform(X.astype(np.float32))
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-5)


def test_prepare_inputs_preserve_shape_per_element() -> None:
    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        preserve_shape=True,
        per_element=True,
        data_format="channels_last",
    )
    rng = np.random.default_rng(2)
    X = rng.standard_normal((6, 3, 2, 1)).astype(np.float32)
    y = rng.standard_normal((6, 3, 2, 1)).astype(np.float32)

    fit_args = _make_fit_args(est, X, y)
    prepared, primary_dim = prepare_inputs_and_scaler(est, fit_args)

    assert primary_dim == 1
    assert prepared.X_cf is not None
    assert prepared.train_inputs.shape == prepared.X_cf.shape
    assert prepared.train_inputs.shape[1:] == (1, 3, 2)
    assert prepared.train_targets is not None
    assert prepared.train_targets.shape[1:] == (1, 3, 2)
    assert est._internal_input_shape_cf_ == (1, 3, 2)


def test_prepare_inputs_hisso_without_targets_defaults_primary_dim_one() -> None:
    est = PSANNRegressor(hidden_layers=1, hidden_units=4, epochs=1, batch_size=8)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((16, 4))

    fit_args = _make_fit_args(est, X, None, hisso=True)
    prepared, primary_dim = prepare_inputs_and_scaler(est, fit_args)

    assert primary_dim == 1
    assert prepared.train_targets is None
    assert prepared.primary_dim == 1
    assert prepared.output_dim == 1
