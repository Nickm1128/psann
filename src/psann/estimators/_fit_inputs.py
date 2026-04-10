from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from ._fit_types import NormalisedFitArgs, PreparedInputState

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


def prepare_inputs_and_scaler(
    estimator: "PSANNRegressor",
    fit_args: NormalisedFitArgs,
) -> Tuple[PreparedInputState, int]:
    """Apply scalers, reshape inputs, and derive primary dimensions."""

    estimator.input_shape_ = estimator._infer_input_shape(fit_args.X)

    if estimator.preserve_shape:
        return _prepare_preserve_shape_inputs(estimator, fit_args)

    return _prepare_flatten_inputs(estimator, fit_args)


def _prepare_flatten_inputs(
    estimator: "PSANNRegressor",
    fit_args: NormalisedFitArgs,
) -> Tuple[PreparedInputState, int]:
    X = fit_args.X
    y = fit_args.y
    context = fit_args.context

    X2d = estimator._flatten(X)
    scaler_transform = estimator._scaler_fit_update(X2d)
    if scaler_transform is not None:
        X_scaled = scaler_transform(X2d).reshape(X.shape[0], *estimator.input_shape_)
    else:
        X_scaled = X

    X_flat = estimator._flatten(X_scaled).astype(np.float32, copy=False)
    train_inputs = X_flat
    train_context = None
    context_dim = None
    if context is not None:
        train_context = np.asarray(context, dtype=np.float32)
        context_dim = int(train_context.shape[1])
    if train_context is None:
        auto_context = estimator._auto_context(train_inputs)
        if auto_context is not None:
            train_context = auto_context.astype(np.float32, copy=False)
            context_dim = int(train_context.shape[1])

    y_vec = None
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_vec = y_arr.reshape(-1, 1)
        else:
            y_vec = y_arr.reshape(y_arr.shape[0], -1)
        target_scaler_transform = estimator._target_scaler_fit_update(y_vec)
        if target_scaler_transform is not None:
            y_vec = target_scaler_transform(y_vec).astype(np.float32, copy=False)
        primary_dim = int(y_vec.shape[1])
    elif fit_args.hisso:
        if estimator.output_shape is not None:
            primary_dim = int(np.prod(estimator.output_shape))
        else:
            primary_dim = 1
    else:
        primary_dim = 1

    prepared = PreparedInputState(
        X_flat=train_inputs,
        X_cf=None,
        context=train_context,
        input_shape=estimator.input_shape_,
        internal_shape_cf=None,
        scaler_transform=scaler_transform,
        train_inputs=train_inputs,
        train_context=train_context,
        train_targets=y_vec,
        y_vector=y_vec,
        y_cf=None,
        context_dim=context_dim,
        primary_dim=primary_dim,
        output_dim=primary_dim,
    )

    return prepared, primary_dim


def _prepare_preserve_shape_inputs(
    estimator: "PSANNRegressor",
    fit_args: NormalisedFitArgs,
) -> Tuple[PreparedInputState, int]:
    X = fit_args.X
    y = fit_args.y
    context = fit_args.context

    if X.ndim < 3:
        raise ValueError(
            "preserve_shape=True requires inputs of shape (N, C, ...); "
            f"got X with shape {X.shape}."
        )

    if estimator.data_format not in {"channels_first", "channels_last"}:
        raise ValueError(
            "data_format must be 'channels_first' or 'channels_last'; "
            f"received {estimator.data_format!r}."
        )

    X_cf = np.moveaxis(X, -1, 1) if estimator.data_format == "channels_last" else X
    cf_shape = X_cf.shape
    estimator._internal_input_shape_cf_ = tuple(cf_shape[1:])

    samples, channels = X_cf.shape[0], int(X_cf.shape[1])
    X2d = X_cf.reshape(samples, channels, -1).transpose(0, 2, 1).reshape(-1, channels)
    scaler_transform = estimator._scaler_fit_update(X2d)
    if scaler_transform is not None:
        X2d = scaler_transform(X2d)
    X_cf = X2d.reshape(samples, -1, channels).transpose(0, 2, 1).reshape(cf_shape)
    X_cf = X_cf.astype(np.float32, copy=False)

    train_context = None
    context_dim = None
    if context is not None:
        ctx = np.asarray(context, dtype=np.float32)
        if ctx.ndim == 1:
            ctx = ctx.reshape(-1, 1)
        if ctx.shape[0] != X.shape[0]:
            raise ValueError(
                f"context has {ctx.shape[0]} samples but X has {X.shape[0]}; dimensions must match."
            )
        train_context = ctx.astype(np.float32, copy=False)
        context_dim = int(train_context.shape[1])

    y_cf = None
    y_vec = None

    if y is None:
        if not fit_args.hisso:
            raise ValueError("y must be provided when hisso=False (preserve_shape=True).")
        if estimator.output_shape is not None:
            primary_dim = int(np.prod(estimator.output_shape))
        else:
            primary_dim = int(np.prod(estimator._internal_input_shape_cf_))
    elif estimator.per_element:
        if estimator.output_shape is not None:
            if estimator.data_format == "channels_first":
                n_targets = int(estimator.output_shape[0])
            else:
                n_targets = int(estimator.output_shape[-1])
        elif estimator.data_format == "channels_first":
            n_targets = int(y.shape[1] if y.ndim == X_cf.ndim else 1)
        else:
            n_targets = int(y.shape[-1] if y.ndim == X.ndim else 1)
        if estimator.data_format == "channels_last":
            if y.ndim == X.ndim:
                y_cf = np.moveaxis(y, -1, 1)
            else:
                y_cf = y[:, None, ...]
        else:
            if y.ndim == X_cf.ndim:
                y_cf = y
            else:
                y_cf = y[:, None, ...]
        if y_cf is None:
            raise ValueError(
                "Unable to align targets with per-element configuration; "
                f"expected y.ndim in ({X_cf.ndim}, {X_cf.ndim - 1}), "
                f"received shape {np.shape(y)}."
            )
        y_cf = y_cf.astype(np.float32, copy=False)
        n_targets = int(y_cf.shape[1])
        y2d = y_cf.reshape(y_cf.shape[0], n_targets, -1).transpose(0, 2, 1).reshape(-1, n_targets)
        target_scaler_transform = estimator._target_scaler_fit_update(y2d)
        if target_scaler_transform is not None:
            y2d = target_scaler_transform(y2d)
            y_cf = (
                y2d.reshape(y_cf.shape[0], -1, n_targets).transpose(0, 2, 1).reshape(y_cf.shape)
            ).astype(np.float32, copy=False)
        y_vec = y_cf.reshape(y_cf.shape[0], -1)
        primary_dim = int(n_targets)
    else:
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_vec = y_arr.reshape(-1, 1)
        else:
            y_vec = y_arr.reshape(y_arr.shape[0], -1)
        target_scaler_transform = estimator._target_scaler_fit_update(y_vec)
        if target_scaler_transform is not None:
            y_vec = target_scaler_transform(y_vec).astype(np.float32, copy=False)
        if estimator.output_shape is not None:
            expected = int(np.prod(estimator.output_shape))
            if y_vec.shape[1] != expected:
                raise ValueError(
                    f"y has {y_vec.shape[1]} targets; expected {expected} from output_shape."
                )
        primary_dim = int(y_vec.shape[1])

    if estimator.data_format == "channels_last":
        X_scaled = np.moveaxis(X_cf, 1, -1)
    else:
        X_scaled = X_cf
    X_flat = estimator._flatten(X_scaled).astype(np.float32, copy=False)
    if train_context is None:
        auto_context = estimator._auto_context(X_flat)
        if auto_context is not None:
            train_context = auto_context.astype(np.float32, copy=False)
            context_dim = int(train_context.shape[1])
    use_cf_inputs = bool(
        estimator.per_element or getattr(estimator, "_use_channel_first_train_inputs_", False)
    )
    train_inputs = X_cf if use_cf_inputs else X_flat

    if use_cf_inputs and y_cf is not None:
        train_targets = y_cf
    elif y_vec is not None:
        train_targets = y_vec
    else:
        train_targets = None

    prepared = PreparedInputState(
        X_flat=X_flat,
        X_cf=X_cf.astype(np.float32, copy=False),
        context=train_context,
        input_shape=estimator.input_shape_,
        internal_shape_cf=estimator._internal_input_shape_cf_,
        scaler_transform=scaler_transform,
        train_inputs=train_inputs,
        train_context=train_context,
        train_targets=train_targets,
        y_vector=y_vec,
        y_cf=y_cf,
        context_dim=context_dim,
        primary_dim=primary_dim,
        output_dim=primary_dim,
    )

    return prepared, primary_dim
