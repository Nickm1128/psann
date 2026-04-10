from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..nn import WithPreprocessor
from ._fit_types import PreparedInputState, ValidationInput

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


def _prepare_validation_tensors(
    estimator: "PSANNRegressor",
    prepared: PreparedInputState,
    validation: Optional[ValidationInput],
    *,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if validation is None:
        return None, None, None

    if len(validation) == 2:
        X_val = np.asarray(validation[0], dtype=np.float32)
        y_val = np.asarray(validation[1], dtype=np.float32)
        ctx_val = None
    elif len(validation) == 3:
        X_val = np.asarray(validation[0], dtype=np.float32)
        y_val = np.asarray(validation[1], dtype=np.float32)
        ctx_val = np.asarray(validation[2], dtype=np.float32)
        if ctx_val.ndim == 1:
            ctx_val = ctx_val.reshape(-1, 1)
        if ctx_val.shape[0] != X_val.shape[0]:
            raise ValueError(
                f"validation context has {ctx_val.shape[0]} samples but X has {X_val.shape[0]}."
            )
    else:
        raise ValueError(
            f"validation_data must contain 2 or 3 elements; received {len(validation)}."
        )

    layout_cf = bool(
        estimator.preserve_shape
        and isinstance(prepared.train_inputs, np.ndarray)
        and prepared.train_inputs.ndim >= 3
    )

    X_val_cf = None
    X_val_cf_scaled = None
    inputs_np = None
    flat_for_context = None

    if estimator.preserve_shape:
        X_val_cf = np.moveaxis(X_val, -1, 1) if estimator.data_format == "channels_last" else X_val
        if prepared.internal_shape_cf is None:
            raise ValueError(
                "PreparedInputState missing channels-first shape for preserve_shape=True."
            )
        expected_cf = tuple(prepared.internal_shape_cf)
        actual_cf = tuple(X_val_cf.shape[1:])
        if actual_cf != expected_cf:
            expected_channels = expected_cf[0] if expected_cf else None
            actual_channels = actual_cf[0] if actual_cf else None
            if (
                expected_channels is not None
                and actual_channels is not None
                and tuple(expected_cf[1:]) == tuple(actual_cf[1:])
            ):
                raise ValueError(
                    f"validation_data channels mismatch: expected {expected_channels}, "
                    f"received {actual_channels}."
                )
            raise ValueError(
                "validation_data X spatial layout mismatch: "
                f"expected {expected_cf}, received {actual_cf}."
            )
        samples, channels = X_val_cf.shape[0], int(X_val_cf.shape[1])
        X_val_2d = X_val_cf.reshape(samples, channels, -1).transpose(0, 2, 1).reshape(-1, channels)
        X_val_2d = estimator._apply_fitted_scaler(X_val_2d)
        X_val_cf_scaled = (
            X_val_2d.reshape(samples, -1, channels).transpose(0, 2, 1).reshape(X_val_cf.shape)
        ).astype(np.float32, copy=False)
        if layout_cf:
            inputs_np = X_val_cf_scaled
        else:
            if estimator.data_format == "channels_last":
                X_val_scaled = np.moveaxis(X_val_cf_scaled, 1, -1)
            else:
                X_val_scaled = X_val_cf_scaled
            inputs_np = estimator._flatten(X_val_scaled).astype(np.float32, copy=False)
        flat_for_context = estimator._flatten(
            np.moveaxis(X_val_cf_scaled, 1, -1)
            if estimator.data_format == "channels_last"
            else X_val_cf_scaled
        ).astype(np.float32, copy=False)
    else:
        n_features = int(np.prod(estimator.input_shape_))
        actual_shape = tuple(X_val.shape[1:])
        expected_shape = tuple(estimator.input_shape_)
        if actual_shape != expected_shape and int(np.prod(actual_shape)) != n_features:
            raise ValueError(
                f"validation_data X has shape {actual_shape}, expected {expected_shape} "
                f"(prod must match {n_features})."
            )
        X_val_flat = estimator._flatten(X_val)
        X_val_flat_scaled = estimator._apply_fitted_scaler(X_val_flat).astype(
            np.float32,
            copy=False,
        )
        inputs_np = X_val_flat_scaled
        flat_for_context = X_val_flat_scaled

    if inputs_np is None:
        raise RuntimeError("Failed to prepare validation inputs; internal bug likely.")

    device_is_cpu = device.type == "cpu"

    def _to_tensor(arr: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(arr.astype(np.float32, copy=False))
        if device_is_cpu:
            return tensor
        return tensor.to(device=device, dtype=torch.float32)

    if ctx_val is None and flat_for_context is not None:
        auto_ctx = estimator._auto_context(flat_for_context.astype(np.float32, copy=False))
        if auto_ctx is not None:
            ctx_val = auto_ctx

    ctx_val_t = _to_tensor(ctx_val) if ctx_val is not None else None
    X_val_t = _to_tensor(inputs_np)

    if layout_cf and estimator.per_element:
        if X_val_cf is None or X_val_cf_scaled is None:
            raise RuntimeError(
                "Per-element validation requires channel-first tensors; inputs were missing."
            )
        if estimator.data_format == "channels_last":
            if y_val.ndim == X_val.ndim:
                y_val_cf = np.moveaxis(y_val, -1, 1)
            elif y_val.ndim == X_val.ndim - 1:
                y_val_cf = y_val[:, None, ...]
            else:
                raise ValueError(
                    "validation y ndim must be "
                    f"{X_val.ndim} or {X_val.ndim - 1} for data_format='channels_last'; "
                    f"received shape {tuple(y_val.shape)} (ndim={y_val.ndim})."
                )
        else:
            if y_val.ndim == X_val_cf.ndim:
                y_val_cf = y_val
            elif y_val.ndim == X_val_cf.ndim - 1:
                y_val_cf = y_val[:, None, ...]
            else:
                raise ValueError(
                    "validation y ndim must be "
                    f"{X_val_cf.ndim} or {X_val_cf.ndim - 1} for data_format='channels_first'; "
                    f"received shape {tuple(y_val.shape)} (ndim={y_val.ndim})."
                )
        if tuple(y_val_cf.shape[2:]) != tuple(X_val_cf.shape[2:]):
            raise ValueError(
                "validation y spatial dimensions do not match X; "
                f"expected {tuple(X_val_cf.shape[2:])}, received {tuple(y_val_cf.shape[2:])}."
            )
        if int(y_val_cf.shape[1]) != int(prepared.output_dim):
            raise ValueError(
                "validation y channel dimension mismatch: "
                f"expected {int(prepared.output_dim)}, received {int(y_val_cf.shape[1])}."
            )
        y_val_cf = y_val_cf.astype(np.float32, copy=False)
        n_val = int(y_val_cf.shape[0])
        n_targets = int(y_val_cf.shape[1])
        y_val_2d = y_val_cf.reshape(n_val, n_targets, -1).transpose(0, 2, 1).reshape(-1, n_targets)
        y_val_2d = estimator._apply_fitted_target_scaler(y_val_2d)
        y_val_cf = (
            y_val_2d.reshape(n_val, -1, n_targets).transpose(0, 2, 1).reshape(y_val_cf.shape)
        ).astype(np.float32, copy=False)
        return X_val_t, _to_tensor(y_val_cf), ctx_val_t

    y_val_flat = y_val.reshape(y_val.shape[0], -1).astype(np.float32, copy=False)
    expected_targets = int(prepared.output_dim)
    if y_val_flat.shape[1] != expected_targets:
        raise ValueError(
            "validation y target dimension mismatch: "
            f"expected {expected_targets}, received {y_val_flat.shape[1]}."
        )
    y_val_flat = estimator._apply_fitted_target_scaler(y_val_flat)
    return X_val_t, _to_tensor(y_val_flat), ctx_val_t


def _prepare_noise_tensor(
    estimator: "PSANNRegressor",
    prepared: PreparedInputState,
    noisy,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if noisy is None:
        return None

    if estimator.preserve_shape:
        if prepared.internal_shape_cf is None:
            raise ValueError(
                "PreparedInputState missing internal channels-first shape for noise construction."
            )
        internal_shape = prepared.internal_shape_cf
        if np.isscalar(noisy):
            std = np.full((1, *internal_shape), float(noisy), dtype=np.float32)
        else:
            arr = np.asarray(noisy, dtype=np.float32)
            if tuple(arr.shape) == internal_shape:
                std = arr.reshape(1, *internal_shape)
            elif (
                tuple(arr.shape) == estimator.input_shape_
                and estimator.data_format == "channels_last"
            ):
                std = np.moveaxis(arr, -1, 0).reshape(1, *internal_shape)
            elif arr.ndim == 1 and arr.size == int(np.prod(internal_shape)):
                std = arr.reshape(1, *internal_shape)
            else:
                raise ValueError(
                    f"noisy shape {arr.shape} not compatible with input shape {estimator.input_shape_}"
                )
        std_t = torch.from_numpy(std.astype(np.float32, copy=False))
        if device.type == "cpu":
            return std_t
        return std_t.to(device=device, dtype=torch.float32)

    n_features = int(np.prod(estimator.input_shape_))
    if np.isscalar(noisy):
        std = np.full((1, n_features), float(noisy), dtype=np.float32)
    else:
        arr = np.asarray(noisy, dtype=np.float32)
        if arr.ndim == 1 and arr.size == n_features:
            std = arr.reshape(1, n_features)
        elif arr.ndim == 2 and arr.shape[1] == n_features:
            std = arr[:1]
        else:
            raise ValueError(
                f"noisy shape {arr.shape} not compatible with flattened feature dimension {n_features}"
            )
    std_t = torch.from_numpy(std.astype(np.float32, copy=False))
    if device.type == "cpu":
        return std_t
    return std_t.to(device=device, dtype=torch.float32)


def _resolve_validation_inputs(
    estimator: "PSANNRegressor",
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    val_inputs = inputs
    preproc = None
    if isinstance(model, WithPreprocessor) and model.preproc is not None:
        preproc = model.preproc
    elif hasattr(estimator, "lsm") and estimator.lsm is not None:
        preproc = estimator.lsm
    if preproc is None:
        return val_inputs

    if hasattr(preproc, "eval"):
        preproc.eval()
    if hasattr(preproc, "forward"):
        with torch.no_grad():
            val_inputs = preproc(inputs)
    return val_inputs
