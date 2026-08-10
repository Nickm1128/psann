from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.nn as nn

from ._fit_types import NormalisedFitArgs, PreparedInputState

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


_BASE_ACTIVATIONS = {
    "psann",
    "sine",
    "respsann",
    "relu_sigmoid_psann",
    "rspsann",
    "rsp",
    "clipped_psann",
    "sigmoid",
    "parameterized_sigmoid",
    "relu",
    "tanh",
    "gelu",
    "silu",
}
_GEOSPARSE_ACTIVATIONS = _BASE_ACTIVATIONS | {
    "phase_psann",
    "phasepsann",
    "mixed",
}
_MODEL_SIGNATURE_FIELDS = (
    "hidden_layers",
    "hidden_units",
    "conv_channels",
    "conv_kernel_size",
    "activation",
    "activation_type",
    "w0",
    "preserve_shape",
    "data_format",
    "per_element",
    "attention",
    "stateful",
    "state",
    "output_shape",
    "lsm",
    "lsm_train",
    "w0_first",
    "w0_hidden",
    "norm",
    "drop_path_max",
    "residual_alpha_init",
    "phase_init",
    "phase_trainable",
    "use_spectral_gate",
    "k_fft",
    "gate_type",
    "gate_groups",
    "gate_init",
    "gate_strength",
    "pool",
    "first_layer_w0",
    "hidden_w0",
    "use_film",
    "use_phase_shift",
    "dropout",
    "context_dim",
    "shape",
    "k",
    "pattern",
    "radius",
    "offsets",
    "wrap_mode",
    "bias",
    "compute_mode",
    "geo_seed",
)


def _require_finite_positive(name: str, value: Any, *, allow_zero: bool = False) -> None:
    number = float(value)
    valid = math.isfinite(number) and (number >= 0.0 if allow_zero else number > 0.0)
    if not valid:
        relation = "finite and >= 0" if allow_zero else "finite and > 0"
        raise ValueError(f"{name} must be {relation}; received {value!r}.")


def _accelerator_available(device_type: str) -> bool:
    if device_type == "cuda":
        return bool(torch.cuda.is_available())
    if device_type == "mps":
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    if device_type == "xpu":
        backend = getattr(torch, "xpu", None)
        return bool(backend is not None and backend.is_available())
    return device_type == "cpu"


def resolve_training_device(
    requested: str | torch.device | None,
    *,
    fallback_policy: str,
) -> tuple[torch.device, Optional[dict[str, Any]]]:
    if fallback_policy not in {"warn", "error"}:
        raise ValueError("fallback_policy must be 'warn' or 'error'.")
    if requested is None or requested == "auto":
        return (
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            None,
        )
    try:
        resolved = torch.device(requested)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device request {requested!r}: {exc}") from exc
    if resolved.type not in {"cpu", "cuda", "mps", "xpu"}:
        raise ValueError(
            "device must resolve to 'cpu', 'cuda', 'mps', 'xpu', or 'auto'; "
            f"received {requested!r}."
        )
    if _accelerator_available(resolved.type):
        return resolved, None

    message = (
        f"Requested accelerator {resolved.type!r} is unavailable; "
        "falling back to CPU because fallback_policy='warn'."
    )
    if fallback_policy == "error":
        raise RuntimeError(
            f"Requested accelerator {resolved.type!r} is unavailable and "
            "fallback_policy='error'."
        )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    return (
        torch.device("cpu"),
        {
            "component": "device",
            "requested": str(resolved),
            "effective": "cpu",
            "reason": "requested accelerator is unavailable",
        },
    )


def configure_deterministic_mode(enabled: bool) -> None:
    torch.use_deterministic_algorithms(bool(enabled))
    cudnn = getattr(torch.backends, "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = bool(enabled)
        if enabled:
            cudnn.benchmark = False


def validate_fit_configuration(
    estimator: "PSANNRegressor",
    fit_args: NormalisedFitArgs,
) -> torch.device:
    """Reject invalid optimizer-driven training configuration before model creation."""

    integer_rules = {
        "epochs": (estimator.epochs, 1),
        "batch_size": (estimator.batch_size, 1),
        "hidden_layers": (estimator.hidden_layers, 0),
        "hidden_units": (estimator.hidden_units, 1),
        "conv_channels": (estimator.conv_channels, 1),
        "conv_kernel_size": (estimator.conv_kernel_size, 1),
        "num_workers": (estimator.num_workers, 0),
        "lsm_pretrain_epochs": (estimator.lsm_pretrain_epochs, 0),
    }
    for name, (value, minimum) in integer_rules.items():
        if int(value) < minimum:
            raise ValueError(f"{name} must be >= {minimum}; received {value!r}.")
    if estimator.early_stopping and int(estimator.patience) < 1:
        raise ValueError("patience must be >= 1 when early_stopping=True.")

    _require_finite_positive("lr", estimator.lr)
    _require_finite_positive("weight_decay", estimator.weight_decay, allow_zero=True)
    if estimator.stream_lr is not None:
        _require_finite_positive("stream_lr", estimator.stream_lr)
    if estimator.lsm_lr is not None:
        _require_finite_positive("lsm_lr", estimator.lsm_lr)
    if estimator.lsm_train and estimator.lsm is None:
        raise ValueError("lsm_train=True requires an lsm preprocessor.")

    optimizer = str(estimator.optimizer).strip().lower()
    if optimizer not in {"adam", "adamw", "sgd"}:
        raise ValueError(f"Unknown optimizer {estimator.optimizer!r}. Supported: adam, adamw, sgd.")
    if not (isinstance(estimator.loss, str) or callable(estimator.loss)):
        raise TypeError("loss must be a supported string or a callable.")
    if isinstance(estimator.loss, str):
        loss = estimator.loss.strip().lower()
        if loss not in {"mse", "l2", "l1", "mae", "smooth_l1", "huber_smooth", "huber"}:
            raise ValueError(
                f"Unknown loss {estimator.loss!r}. Supported: mse/l2, l1/mae, "
                "smooth_l1/huber_smooth, huber, or a callable."
            )
        params = estimator.loss_params or {}
        allowed_loss_params = {
            "mse": set(),
            "l2": set(),
            "l1": set(),
            "mae": set(),
            "smooth_l1": {"beta"},
            "huber_smooth": {"beta"},
            "huber": {"delta"},
        }[loss]
        unknown_loss_params = sorted(set(params) - allowed_loss_params)
        if unknown_loss_params:
            raise ValueError(
                f"Unsupported loss_params for {loss!r}: {', '.join(unknown_loss_params)}."
            )
        if loss in {"smooth_l1", "huber_smooth"}:
            _require_finite_positive("loss_params['beta']", params.get("beta", 1.0))
        if loss == "huber":
            _require_finite_positive("loss_params['delta']", params.get("delta", 1.0))
    reduction = str(estimator.loss_reduction).strip().lower()
    if reduction == "none":
        raise ValueError(
            "loss_reduction='none' is not supported for optimizer-driven training; "
            "use 'mean' or 'sum' so backward receives a scalar loss."
        )
    if reduction not in {"mean", "sum"}:
        raise ValueError("loss_reduction must be 'mean' or 'sum'.")

    activation = str(estimator.activation_type).strip().lower()
    allowed_activations = (
        _GEOSPARSE_ACTIVATIONS
        if estimator.__class__.__name__ == "GeoSparseRegressor"
        else _BASE_ACTIVATIONS
    )
    if activation not in allowed_activations:
        supported = ", ".join(sorted(allowed_activations))
        raise ValueError(
            f"Unknown activation_type {estimator.activation_type!r}. Supported: {supported}."
        )

    if estimator.state_reset not in {"batch", "epoch", "none"}:
        raise ValueError("state_reset must be 'batch', 'epoch', or 'none'.")
    if estimator.data_format not in {"channels_first", "channels_last"}:
        raise ValueError("data_format must be 'channels_first' or 'channels_last'.")
    if estimator.per_element and not estimator.preserve_shape:
        raise ValueError("per_element=True requires preserve_shape=True.")
    if estimator.output_shape is not None:
        if not estimator.output_shape or any(int(dim) <= 0 for dim in estimator.output_shape):
            raise ValueError("output_shape must contain only positive dimensions.")
    if estimator._attention_enabled() and estimator.lsm is not None:
        raise ValueError(
            "attention is incompatible with lsm preprocessors in the current training core."
        )

    if (fit_args.lr_max is None) ^ (fit_args.lr_min is None):
        raise ValueError("lr_max and lr_min must be provided together.")
    if fit_args.lr_max is not None and fit_args.lr_min is not None:
        _require_finite_positive("lr_max", fit_args.lr_max)
        _require_finite_positive("lr_min", fit_args.lr_min)
        if fit_args.lr_max < fit_args.lr_min:
            raise ValueError("lr_max must be >= lr_min.")
        if fit_args.scheduler != "none":
            raise ValueError("lr_max/lr_min cannot be combined with scheduler.")

    if fit_args.scheduler not in {"none", "step", "cosine"}:
        raise ValueError("scheduler must be 'none', 'step', or 'cosine'.")
    scheduler_params = fit_args.scheduler_params
    if fit_args.scheduler == "none" and scheduler_params:
        raise ValueError("scheduler_params requires scheduler='step' or 'cosine'.")
    if fit_args.scheduler == "step":
        unknown_scheduler_params = sorted(set(scheduler_params) - {"step_size", "gamma"})
        if unknown_scheduler_params:
            raise ValueError(
                "Unsupported scheduler_params for 'step': "
                f"{', '.join(unknown_scheduler_params)}."
            )
        if int(scheduler_params.get("step_size", 1)) < 1:
            raise ValueError("scheduler_params['step_size'] must be >= 1.")
        _require_finite_positive("scheduler_params['gamma']", scheduler_params.get("gamma", 0.1))
    if fit_args.scheduler == "cosine":
        unknown_scheduler_params = sorted(set(scheduler_params) - {"t_max", "eta_min"})
        if unknown_scheduler_params:
            raise ValueError(
                "Unsupported scheduler_params for 'cosine': "
                f"{', '.join(unknown_scheduler_params)}."
            )
        if int(scheduler_params.get("t_max", estimator.epochs)) < 1:
            raise ValueError("scheduler_params['t_max'] must be >= 1.")
        _require_finite_positive(
            "scheduler_params['eta_min']",
            scheduler_params.get("eta_min", 0.0),
            allow_zero=True,
        )

    if fit_args.nonfinite_policy not in {"error", "skip_step", "continue"}:
        raise ValueError("nonfinite_policy must be 'error', 'skip_step', or 'continue'.")
    if fit_args.fallback_policy not in {"warn", "error"}:
        raise ValueError("fallback_policy must be 'warn' or 'error'.")
    if fit_args.callback_error_policy not in {"raise", "warn"}:
        raise ValueError("callback_error_policy must be 'raise' or 'warn'.")
    amp_dtype = getattr(estimator, "amp_dtype", None)
    if isinstance(amp_dtype, str):
        amp_dtype_name = amp_dtype.strip().lower()
        if amp_dtype_name not in {
            "bf16",
            "bfloat16",
            "fp16",
            "float16",
            "fp32",
            "float32",
        }:
            raise ValueError("amp_dtype must be float16/fp16, bfloat16/bf16, or float32/fp32.")
        if estimator.amp and amp_dtype_name in {"fp32", "float32"}:
            raise ValueError(
                "amp_dtype must be float16 or bfloat16 when amp=True; "
                "disable amp for float32 training."
            )
    elif amp_dtype is not None:
        if amp_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("amp_dtype must be torch.float16, bfloat16, or float32.")
        if estimator.amp and amp_dtype == torch.float32:
            raise ValueError(
                "amp_dtype must be float16 or bfloat16 when amp=True; "
                "disable amp for float32 training."
            )
    if not str(getattr(estimator, "compile_backend", "")).strip():
        raise ValueError("compile_backend must be a non-empty string.")
    if not str(getattr(estimator, "compile_mode", "")).strip():
        raise ValueError("compile_mode must be a non-empty string.")
    if fit_args.checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0.")
    if fit_args.checkpoint_keep < 1:
        raise ValueError("checkpoint_keep must be >= 1.")
    if fit_args.checkpoint_every and fit_args.checkpoint_dir is None:
        raise ValueError("checkpoint_every requires checkpoint_dir.")
    if fit_args.checkpoint_dir is not None:
        for name in ("scaler", "target_scaler"):
            scaler = getattr(estimator, name, None)
            if scaler is not None and not isinstance(scaler, str):
                raise ValueError(
                    f"{name} must be a built-in scaler name when checkpoint_dir is set; "
                    "custom scaler objects are not part of the restricted training "
                    "checkpoint contract."
                )
    if fit_args.hisso:
        unsupported = []
        if fit_args.scheduler != "none":
            unsupported.append("scheduler")
        if fit_args.metrics:
            unsupported.append("metrics")
        if fit_args.callbacks:
            unsupported.append("callbacks")
        if fit_args.resume_from is not None:
            unsupported.append("resume_from")
        if fit_args.checkpoint_dir is not None:
            unsupported.append("checkpoint_dir")
        if fit_args.nonfinite_policy != "error":
            unsupported.append("nonfinite_policy")
        if unsupported:
            raise ValueError(
                "HISSO training does not yet support the following supervised training "
                f"options: {', '.join(unsupported)}."
            )

    for name, metric in fit_args.metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Metric names must be non-empty strings.")
        if not callable(metric):
            raise TypeError(f"Metric {name!r} must be callable.")
    for callback in fit_args.callbacks:
        if not callable(callback):
            raise TypeError("Every training callback must be callable.")
    if fit_args.logger is not None and not hasattr(fit_args.logger, "log"):
        raise TypeError("logger must be a logging.Logger-compatible object.")

    configure_deterministic_mode(fit_args.deterministic)
    resolved, fallback = resolve_training_device(
        estimator.device,
        fallback_policy=fit_args.fallback_policy,
    )
    estimator._resolved_training_device_ = resolved
    estimator._fit_fallbacks_ = [fallback] if fallback is not None else []
    return resolved


def _freeze_signature_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _freeze_signature_value(item)) for key, item in value.items())
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_signature_value(item) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return (value.__class__.__module__, value.__class__.__qualname__, repr(value))


def build_model_signature(
    estimator: "PSANNRegressor",
    prepared: PreparedInputState,
) -> tuple[Any, ...]:
    config = tuple(
        (
            name,
            _freeze_signature_value(getattr(estimator, name)),
        )
        for name in _MODEL_SIGNATURE_FIELDS
        if hasattr(estimator, name)
    )
    return (
        estimator.__class__.__module__,
        estimator.__class__.__qualname__,
        tuple(prepared.input_shape),
        tuple(prepared.internal_shape_cf or ()),
        tuple(prepared.train_inputs.shape[1:]),
        int(prepared.output_dim),
        prepared.context_dim,
        config,
    )


def validate_warm_start_signature(
    estimator: "PSANNRegressor",
    signature: tuple[Any, ...],
) -> None:
    existing = getattr(estimator, "_model_signature_", None)
    if not estimator.warm_start or not isinstance(getattr(estimator, "model_", None), nn.Module):
        return
    if existing is None:
        raise ValueError(
            "warm_start=True cannot safely reuse a legacy fitted model without a model "
            "signature; refit once with warm_start=False."
        )
    if existing != signature:
        raise ValueError(
            "warm_start=True is incompatible with the fitted model architecture or "
            "input/target shape. Restore the original configuration or set warm_start=False."
        )


def validate_prepared_finite_values(prepared: PreparedInputState) -> None:
    arrays = {
        "prepared training inputs": prepared.train_inputs,
        "prepared training targets": prepared.train_targets,
        "prepared training context": prepared.train_context,
    }
    for name, array in arrays.items():
        if array is None:
            continue
        value = np.asarray(array)
        if np.isnan(value).any():
            raise ValueError(
                f"{name} contains NaN after preprocessing; fix the scaler, context "
                "builder, or preprocessing configuration."
            )
        if np.isinf(value).any():
            raise ValueError(
                f"{name} contains infinity after preprocessing; fix the scaler, "
                "context builder, or preprocessing configuration."
            )


def validate_prediction_target_shape(
    model: nn.Module,
    prepared: PreparedInputState,
    *,
    device: torch.device,
) -> None:
    targets = prepared.train_targets
    if targets is None:
        return
    sample_count = min(2, int(prepared.train_inputs.shape[0]))
    inputs = torch.from_numpy(
        prepared.train_inputs[:sample_count].astype("float32", copy=False)
    ).to(device)
    targets_t = torch.from_numpy(targets[:sample_count].astype("float32", copy=False)).to(device)
    context_t = None
    if prepared.train_context is not None:
        context_t = torch.from_numpy(
            prepared.train_context[:sample_count].astype("float32", copy=False)
        ).to(device)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(inputs, context_t) if context_t is not None else model(inputs)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(
                "Model forward must return a torch.Tensor for optimizer-driven training."
            )
        if tuple(prediction.shape) != tuple(targets_t.shape):
            raise ValueError(
                "Model prediction shape does not match target shape before the first "
                f"optimizer step: prediction={tuple(prediction.shape)}, "
                f"target={tuple(targets_t.shape)}."
            )
    finally:
        if hasattr(model, "reset_state"):
            model.reset_state()
        model.train(was_training)


__all__ = [
    "build_model_signature",
    "configure_deterministic_mode",
    "resolve_training_device",
    "validate_fit_configuration",
    "validate_prepared_finite_values",
    "validate_prediction_target_shape",
    "validate_warm_start_signature",
]
