from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

from .nn import WithPreprocessor, PSANNNet, ResidualPSANNNet
from .conv import PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet, ResidualPSANNConv2dNet

if TYPE_CHECKING:
    from .sklearn import PSANNRegressor


@dataclass
class SupervisedExtrasConfig:
    """Metadata describing supervised extras behaviour for time-series models."""

    primary_dim: int
    extras_dim: int
    feature_dim: int
    append_to_inputs: bool
    weight: float = 1.0
    mode: str = "joint"
    cycle: int = 1

    @property
    def base_dim(self) -> int:
        if self.append_to_inputs:
            return int(self.feature_dim) - int(self.extras_dim)
        return int(self.feature_dim)




@dataclass
class ExtrasGrowthConfig:
    """User-facing surface for configuring extras head growth and scheduling."""

    extras_dim: int
    append_to_inputs: bool = False
    warm_start_epochs: Optional[int] = None
    freeze_until_plateau: bool = False
    loss_weight: Optional[float] = None
    loss_mode: Optional[str] = None
    loss_cycle: Optional[int] = None
    auto_expand_on_load: bool = True

    def with_default_loss(
        self,
        *,
        weight: Optional[float],
        mode: Optional[str],
        cycle: Optional[int],
    ) -> "ExtrasGrowthConfig":
        """Return a copy with unspecified loss schedule fields filled."""

        return ExtrasGrowthConfig(
            extras_dim=self.extras_dim,
            append_to_inputs=self.append_to_inputs,
            warm_start_epochs=self.warm_start_epochs,
            freeze_until_plateau=self.freeze_until_plateau,
            loss_weight=self.loss_weight if self.loss_weight is not None else weight,
            loss_mode=self.loss_mode if self.loss_mode is not None else mode,
            loss_cycle=self.loss_cycle if self.loss_cycle is not None else cycle,
            auto_expand_on_load=self.auto_expand_on_load,
        )


def ensure_extras_growth_config(value: Any | None, *, default_dim: Optional[int] = None) -> ExtrasGrowthConfig:
    """Coerce legacy extras arguments into :class:`ExtrasGrowthConfig`."""

    if isinstance(value, ExtrasGrowthConfig):
        return value
    if value is None:
        if default_dim is None:
            raise ValueError("extras_dim must be provided when extras_growth is None")
        return ExtrasGrowthConfig(extras_dim=max(0, int(default_dim)))
    if isinstance(value, dict):
        data = dict(value)
        dim = data.get("extras_dim", data.get("extras", default_dim))
        if dim is None:
            raise ValueError("extras_growth dict must include 'extras_dim' or provide a default_dim")

        def _opt_int(*keys: str) -> Optional[int]:
            for key in keys:
                if key in data and data[key] is not None:
                    try:
                        return int(data[key])
                    except Exception as exc:
                        raise ValueError(f"extras_growth[{key}] could not be coerced to int") from exc
            return None

        def _opt_float(*keys: str) -> Optional[float]:
            for key in keys:
                if key in data and data[key] is not None:
                    try:
                        return float(data[key])
                    except Exception as exc:
                        raise ValueError(f"extras_growth[{key}] could not be coerced to float") from exc
            return None

        def _opt_str(*keys: str) -> Optional[str]:
            for key in keys:
                if key in data and data[key] is not None:
                    return str(data[key])
            return None

        append = data.get("append_to_inputs", data.get("append", False))
        warm = _opt_int("warm_start_epochs", "warm_start")
        freeze_flag = data.get("freeze_until_plateau")
        if freeze_flag is None and "freeze_until" in data:
            freeze_flag = data.get("freeze_until")
        freeze = bool(freeze_flag) if freeze_flag is not None else False
        weight = _opt_float("loss_weight", "extras_loss_weight")
        mode = _opt_str("loss_mode", "extras_loss_mode")
        cycle = _opt_int("loss_cycle", "extras_loss_cycle")
        auto_expand_flag = data.get("auto_expand_on_load", data.get("auto_expand"))
        auto_expand = True if auto_expand_flag is None else bool(auto_expand_flag)
        dim_val = max(0, int(dim))
        return ExtrasGrowthConfig(
            extras_dim=dim_val,
            append_to_inputs=bool(append),
            warm_start_epochs=warm,
            freeze_until_plateau=freeze,
            loss_weight=weight,
            loss_mode=mode.lower().strip() if isinstance(mode, str) else mode,
            loss_cycle=cycle,
            auto_expand_on_load=auto_expand,
        )
    try:
        dim = int(value)
    except Exception as exc:
        raise ValueError("extras_growth must be an int, dict, or ExtrasGrowthConfig") from exc
    return ExtrasGrowthConfig(extras_dim=max(0, int(dim)))


def extras_growth_to_metadata(cfg: ExtrasGrowthConfig) -> Dict[str, Any]:
    """Serialize extras growth config into plain metadata."""

    data: Dict[str, Any] = {
        "extras_dim": int(cfg.extras_dim),
        "append_to_inputs": bool(cfg.append_to_inputs),
        "auto_expand_on_load": bool(cfg.auto_expand_on_load),
    }
    if cfg.warm_start_epochs is not None:
        data["warm_start_epochs"] = int(cfg.warm_start_epochs)
    if cfg.freeze_until_plateau:
        data["freeze_until_plateau"] = True
    if cfg.loss_weight is not None:
        data["loss_weight"] = float(cfg.loss_weight)
    if cfg.loss_mode is not None:
        data["loss_mode"] = str(cfg.loss_mode)
    if cfg.loss_cycle is not None:
        data["loss_cycle"] = int(cfg.loss_cycle)
    return data

def ensure_supervised_extras_config(value: Any) -> SupervisedExtrasConfig:
    """Coerce persisted metadata into :class:`SupervisedExtrasConfig`."""

    if isinstance(value, SupervisedExtrasConfig):
        return value
    if isinstance(value, dict):
        return SupervisedExtrasConfig(
            primary_dim=int(value.get("primary_dim", value.get("primary", 0))),
            extras_dim=int(value.get("extras_dim", value.get("extras", 0))),
            feature_dim=int(value.get("feature_dim", value.get("features", 0))),
            append_to_inputs=bool(value.get("append_to_inputs", value.get("append", False))),
            weight=float(value.get("weight", value.get("extras_loss_weight", 1.0))),
            mode=str(value.get("mode", value.get("extras_loss_mode", "joint"))),
            cycle=int(value.get("cycle", value.get("extras_loss_cycle", 1))),
        )
    raise ValueError("Unsupported supervised extras configuration format")





def _expand_linear_in_features(linear: nn.Linear, new_in: int) -> nn.Linear:
    old_in = int(linear.in_features)
    if new_in < old_in:
        raise ValueError("new input dimension must be >= current dimension")
    if new_in == old_in:
        return linear
    device = linear.weight.device
    dtype = linear.weight.dtype
    out_features = int(linear.out_features)
    bias = linear.bias is not None
    new_linear = nn.Linear(new_in, out_features, bias=bias).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_linear.weight.zero_()
        new_linear.weight[:, :old_in] = linear.weight.data
        if bias and linear.bias is not None:
            new_linear.bias.zero_()
            new_linear.bias[: linear.bias.shape[0]] = linear.bias.data
    return new_linear


def _expand_linear_out_features(linear: nn.Linear, new_out: int) -> nn.Linear:
    old_out = int(linear.out_features)
    if new_out < old_out:
        raise ValueError("new output dimension must be >= current dimension")
    if new_out == old_out:
        return linear
    device = linear.weight.device
    dtype = linear.weight.dtype
    in_features = int(linear.in_features)
    bias = linear.bias is not None
    new_linear = nn.Linear(in_features, new_out, bias=bias).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_linear.weight.zero_()
        new_linear.weight[:old_out, :in_features] = linear.weight.data
        if bias and linear.bias is not None:
            new_linear.bias.zero_()
            new_linear.bias[: old_out] = linear.bias.data
    return new_linear


def _expand_linear_in_out(linear: nn.Linear, new_in: int, new_out: int) -> nn.Linear:
    old_in = int(linear.in_features)
    old_out = int(linear.out_features)
    if new_in < old_in or new_out < old_out:
        raise ValueError("new dimensions must be >= current dimensions")
    device = linear.weight.device
    dtype = linear.weight.dtype
    bias = linear.bias is not None
    new_linear = nn.Linear(new_in, new_out, bias=bias).to(device=device, dtype=dtype)
    with torch.no_grad():
        new_linear.weight.zero_()
        new_linear.weight[:old_out, :old_in] = linear.weight.data
        if bias and linear.bias is not None:
            new_linear.bias.zero_()
            new_linear.bias[: old_out] = linear.bias.data
    return new_linear


def _expand_conv_module(
    conv: _ConvNd,
    *,
    new_in: Optional[int] = None,
    new_out: Optional[int] = None,
) -> _ConvNd:
    old_in = int(conv.in_channels)
    old_out = int(conv.out_channels)
    new_in = old_in if new_in is None else int(new_in)
    new_out = old_out if new_out is None else int(new_out)
    if new_in < old_in or new_out < old_out:
        raise ValueError('new_in and new_out must be >= current channel counts')
    if new_in == old_in and new_out == old_out:
        return conv
    if int(conv.groups) != 1:
        raise NotImplementedError(
            'Grouped convolutions are not yet supported for extras expansion.'
        )
    cls = type(conv)
    kwargs = {
        'kernel_size': conv.kernel_size,
        'stride': conv.stride,
        'padding': conv.padding,
        'dilation': conv.dilation,
        'groups': conv.groups,
        'bias': conv.bias is not None,
        'padding_mode': getattr(conv, 'padding_mode', 'zeros'),
    }
    new_conv = cls(new_in, new_out, **kwargs).to(
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:old_out, :old_in, ...] = conv.weight.data
        if conv.bias is not None:
            new_conv.bias.zero_()
            new_conv.bias[:old_out] = conv.bias.data
    return new_conv


def _expand_psann_conv_core(
    core: nn.Module,
    *,
    delta_inputs: int,
    delta_extras: int,
    new_total_out: int,
) -> None:
    has_body = len(getattr(core, 'body', [])) > 0
    if delta_inputs > 0 and has_body:
        block0 = core.body[0]
        if not hasattr(block0, 'conv'):
            raise NotImplementedError(
                "Unsupported convolutional block without 'conv' attribute."
            )
        block0.conv = _expand_conv_module(
            block0.conv,
            new_in=block0.conv.in_channels + delta_inputs,
        )
    if getattr(core, 'segmentation_head', False):
        head_in = core.head.in_channels + (delta_inputs if not has_body else 0)
        head_out = core.head.out_channels + delta_extras
        core.head = _expand_conv_module(core.head, new_in=head_in, new_out=head_out)
    else:
        fc_in = core.fc.in_features + (delta_inputs if not has_body else 0)
        fc_out = core.fc.out_features + delta_extras
        if has_body:
            core.fc = _expand_linear_out_features(core.fc, fc_out)
        else:
            core.fc = _expand_linear_in_out(core.fc, fc_in, fc_out)
    core.out_dim = int(new_total_out)


def _expand_residual_psann_conv2d_core(
    core: ResidualPSANNConv2dNet,
    *,
    delta_inputs: int,
    delta_extras: int,
    new_total_out: int,
) -> None:
    if delta_inputs > 0:
        core.in_proj = _expand_conv_module(
            core.in_proj,
            new_in=core.in_proj.in_channels + delta_inputs,
        )
    if core.segmentation_head:
        head_out = core.head.out_channels + delta_extras
        core.head = _expand_conv_module(core.head, new_out=head_out)
    else:
        core.fc = _expand_linear_out_features(core.fc, core.fc.out_features + delta_extras)
    core.out_dim = int(new_total_out)


def _bump_preserve_shape_metadata(estimator: Any, delta_channels: int) -> None:
    if delta_channels <= 0:
        return
    if not getattr(estimator, 'preserve_shape', False):
        return
    if hasattr(estimator, '_internal_input_shape_cf_'):
        shape_cf = list(estimator._internal_input_shape_cf_)
        if shape_cf:
            shape_cf[0] = int(shape_cf[0]) + delta_channels
            estimator._internal_input_shape_cf_ = tuple(shape_cf)
    if hasattr(estimator, 'input_shape_'):
        shape = list(estimator.input_shape_)
        if shape:
            idx = -1 if getattr(estimator, 'data_format', 'channels_first') == 'channels_last' else 0
            shape[idx] = int(shape[idx]) + delta_channels
            estimator.input_shape_ = tuple(shape)


def expand_extras_head(
    estimator: "PSANNRegressor",
    new_extras_dim: int,
    *,
    extras_growth: Any | None = None,
) -> "PSANNRegressor":
    """Clone a fitted estimator with an expanded extras head."""

    if new_extras_dim <= 0:
        raise ValueError("new_extras_dim must be positive")
    if not hasattr(estimator, "model_"):
        raise RuntimeError("Estimator must be fitted before expanding extras.")

    base_cfg = estimator.get_extras_growth()
    old_extras = int(base_cfg.extras_dim)
    if new_extras_dim <= old_extras:
        raise ValueError(
            f"new_extras_dim must be greater than the current extras_dim ({old_extras})."
        )

    target_cfg = (
        ensure_extras_growth_config(extras_growth, default_dim=new_extras_dim)
        if extras_growth is not None
        else replace(base_cfg, extras_dim=int(new_extras_dim))
    )

    clone = copy.deepcopy(estimator)
    clone._extras_growth_cfg = target_cfg
    clone.extras_growth = target_cfg
    clone.extras = int(target_cfg.extras_dim)

    if getattr(clone, "_supervised_extras_meta_", None) is not None:
        clone._supervised_extras_meta_ = replace(
            clone._supervised_extras_meta_, extras_dim=int(target_cfg.extras_dim)
        )
    if getattr(clone, "_hisso_cfg_", None) is not None:
        cfg = clone._hisso_cfg_
        try:
            clone._hisso_cfg_ = replace(cfg, extras_dim=int(target_cfg.extras_dim))
        except Exception:
            if isinstance(cfg, dict):
                cfg = dict(cfg)
                cfg["extras_dim"] = int(target_cfg.extras_dim)
                clone._hisso_cfg_ = cfg

    clone._extras_cache_ = None
    clone._hisso_extras_cache_ = None
    if hasattr(clone, "_stream_opt"):
        clone._stream_opt = None

    model = clone.model_
    preproc = None
    core = model
    if isinstance(model, WithPreprocessor):
        preproc = model.preproc
        core = model.core

    if preproc is not None and hasattr(preproc, "extras_dim"):
        preproc.extras_dim = int(target_cfg.extras_dim)

    param = next(core.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")
    dtype = param.dtype if param is not None else torch.float32

    if isinstance(core, (PSANNNet, ResidualPSANNNet)):
        old_total_out = int(core.head.out_features)
    elif isinstance(core, (PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet)):
        old_total_out = int(
            core.head.out_channels if getattr(core, "segmentation_head", False) else core.fc.out_features
        )
    elif isinstance(core, ResidualPSANNConv2dNet):
        old_total_out = int(core.head.out_channels if core.segmentation_head else core.fc.out_features)
    else:
        raise NotImplementedError(
            "expand_extras_head currently supports PSANNNet, ResidualPSANNNet, PSANNConvNd, and ResidualPSANNConv2dNet cores."
        )

    primary_dim = int(old_total_out - old_extras)
    if primary_dim < 0:
        raise RuntimeError("Detected inconsistent extras configuration on estimator.")

    delta_extras = int(target_cfg.extras_dim) - old_extras
    delta_inputs = delta_extras

    _bump_preserve_shape_metadata(clone, delta_inputs)

    if isinstance(core, PSANNNet):
        has_hidden = len(core.body) > 0
        if has_hidden and delta_inputs > 0:
            block0 = core.body[0]
            if not hasattr(block0, "linear"):
                raise NotImplementedError("Unsupported PSANNNet block structure.")
            block0.linear = _expand_linear_in_features(
                block0.linear, block0.linear.in_features + delta_inputs
            )
        if has_hidden:
            core.head = _expand_linear_out_features(
                core.head, core.head.out_features + delta_extras
            )
        else:
            new_in = core.head.in_features + delta_inputs
            new_out = core.head.out_features + delta_extras
            core.head = _expand_linear_in_out(core.head, new_in, new_out)
    elif isinstance(core, ResidualPSANNNet):
        if delta_inputs > 0:
            core.in_linear = _expand_linear_in_features(
                core.in_linear, core.in_linear.in_features + delta_inputs
            )
        core.head = _expand_linear_out_features(
            core.head, core.head.out_features + delta_extras
        )
        if hasattr(core, "output_dim"):
            core.output_dim = int(primary_dim + target_cfg.extras_dim)
    elif isinstance(core, (PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet)):
        _expand_psann_conv_core(
            core,
            delta_inputs=delta_inputs,
            delta_extras=delta_extras,
            new_total_out=int(primary_dim + target_cfg.extras_dim),
        )
    elif isinstance(core, ResidualPSANNConv2dNet):
        _expand_residual_psann_conv2d_core(
            core,
            delta_inputs=delta_inputs,
            delta_extras=delta_extras,
            new_total_out=int(primary_dim + target_cfg.extras_dim),
        )

    if hasattr(core, "output_dim") and not isinstance(core, ResidualPSANNNet):
        try:
            core.output_dim = int(primary_dim + target_cfg.extras_dim)
        except Exception:
            pass

    core.to(device=device, dtype=dtype)
    if isinstance(model, WithPreprocessor):
        model.core = core
        clone.model_ = model
    else:
        clone.model_ = core

    if hasattr(clone, "input_shape_") and isinstance(clone.input_shape_, tuple) and len(clone.input_shape_) == 1:
        clone.input_shape_ = (int(clone.input_shape_[0]) + delta_inputs,)

    if getattr(clone, "_scaler_kind_", None) == "standard" and getattr(clone, "_scaler_state_", None) is not None:
        st = clone._scaler_state_
        for key in ("mean", "M2"):
            arr = st.get(key)
            if arr is not None:
                st[key] = np.concatenate(
                    [np.asarray(arr, dtype=np.float32), np.zeros(delta_inputs, dtype=np.float32)]
                )
        clone._scaler_state_ = st
    elif getattr(clone, "_scaler_kind_", None) == "minmax" and getattr(clone, "_scaler_state_", None) is not None:
        st = clone._scaler_state_
        mn = np.asarray(st.get("min"), dtype=np.float32)
        mx = np.asarray(st.get("max"), dtype=np.float32)
        st["min"] = np.concatenate([mn, np.zeros(delta_inputs, dtype=np.float32)])
        st["max"] = np.concatenate([mx, np.ones(delta_inputs, dtype=np.float32)])
        clone._scaler_state_ = st
    elif getattr(clone, "_scaler_kind_", None) == "custom":
        clone._scaler_kind_ = None
        clone._scaler_state_ = None
        clone._scaler_spec_ = None
        if hasattr(clone, "_scaler_fitted_"):
            clone._scaler_fitted_ = False

    if hasattr(clone, "_hisso_trainer_"):
        clone._hisso_trainer_ = None
    if hasattr(clone, "_invalidate_training_state"):
        clone._invalidate_training_state()

    return clone

def _normalise_initial_extras(
    extras_dim: int,
    *,
    initial_extras: Optional[np.ndarray],
    cache: Optional[np.ndarray],
) -> np.ndarray:
    if extras_dim <= 0:
        raise ValueError("extras_dim must be positive for extras rollout")
    if initial_extras is not None:
        arr = np.asarray(initial_extras, dtype=np.float32).reshape(-1)
        if arr.shape[-1] != extras_dim:
            raise ValueError(
                f"initial extras length {arr.shape[-1]} does not match extras_dim={extras_dim}"
            )
        return arr.astype(np.float32, copy=False)
    if cache is not None:
        arr = np.asarray(cache, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] == extras_dim:
                return arr.astype(np.float32, copy=False)
        elif arr.ndim == 2 and arr.shape[1] == extras_dim:
            return arr[-1].astype(np.float32, copy=False)
    return np.zeros(extras_dim, dtype=np.float32)


def _apply_estimator_scaler(estimator: Any, arr: np.ndarray) -> np.ndarray:
    """Apply estimator scaler to a flattened 2D array (N, F)."""

    kind = getattr(estimator, "_scaler_kind_", None)
    if kind is None:
        return arr
    state = getattr(estimator, "_scaler_state_", {}) or {}
    if kind == "standard":
        mean = np.asarray(state.get("mean"), dtype=np.float32)
        var = np.asarray(state.get("M2"), dtype=np.float32)
        n = max(int(state.get("n", 1)), 1)
        if mean.size and var.size:
            std = np.sqrt(np.maximum(var / n, 1e-8)).astype(np.float32)
            return (arr - mean) / std
    elif kind == "minmax":
        mn = np.asarray(state.get("min"), dtype=np.float32)
        mx = np.asarray(state.get("max"), dtype=np.float32)
        if mn.size and mx.size:
            scale = np.where((mx - mn) > 1e-8, (mx - mn), 1.0)
            return (arr - mn) / scale
    elif kind == "custom" and hasattr(estimator.scaler, "transform"):
        return estimator.scaler.transform(arr)
    return arr


def rollout_supervised_extras(
    estimator: Any,
    X_obs: np.ndarray,
    *,
    config: SupervisedExtrasConfig,
    extras_cache: Optional[np.ndarray] = None,
    initial_extras: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out supervised extras over a time-series.

    Parameters
    ----------
    estimator: fitted PSANN estimator
    X_obs: array-like of base observations with shape (N, base_dim)
    config: supervised extras configuration (primary/extras dims, etc.)
    extras_cache: optional cached extras state from a previous rollout
    initial_extras: optional explicit initial extras vector; overrides cache

    Returns
    -------
    primary_pred: np.ndarray, shape (N, primary_dim)
    extras_seq: np.ndarray, shape (N + 1, extras_dim) with extras_seq[0] being the
        initial extras and extras_seq[t+1] the predicted extras at step t
    extras_cache: np.ndarray, cached extras sequence (equal to extras_seq)
    """

    X_arr = np.asarray(X_obs, dtype=np.float32)
    cfg = ensure_supervised_extras_config(config)
    K = int(cfg.extras_dim)
    if K <= 0:
        raise ValueError("Estimator was not configured with extras; nothing to roll out.")

    if cfg.append_to_inputs:
        expected_base = cfg.base_dim
        if X_arr.ndim != 2 or X_arr.shape[1] != expected_base:
            raise ValueError(
                f"Expected base features with shape (N, {expected_base}) but received {X_arr.shape}"
            )
    else:
        if X_arr.ndim != 2 or X_arr.shape[1] != cfg.feature_dim:
            raise ValueError(
                f"Expected features with shape (N, {cfg.feature_dim}) but received {X_arr.shape}"
            )

    init_extras = _normalise_initial_extras(K, initial_extras=initial_extras, cache=extras_cache)
    N = int(X_arr.shape[0])

    primary = np.zeros((N, cfg.primary_dim), dtype=np.float32)
    extras_seq = np.zeros((N + 1, K), dtype=np.float32)
    extras_seq[0] = init_extras

    model = estimator.model_
    if model is None:
        raise RuntimeError("Estimator has no trained model; call fit() first.")
    model.eval()
    device = estimator._device()

    extras_state = init_extras.astype(np.float32, copy=True)
    for t in range(N):
        if cfg.append_to_inputs:
            x_t = np.concatenate([X_arr[t], extras_state], axis=-1)
        else:
            x_t = X_arr[t]
        x_batch = x_t.reshape(1, -1)
        x_batch = _apply_estimator_scaler(estimator, x_batch)
        with torch.no_grad():
            torch_in = torch.from_numpy(x_batch).to(device)
            out = model(torch_in).cpu().numpy()[0]
        primary[t] = out[: cfg.primary_dim]
        extras_state = out[cfg.primary_dim : cfg.primary_dim + K]
        extras_seq[t + 1] = extras_state

    cache_arr = extras_seq.copy()
    return primary, extras_seq, cache_arr


__all__ = [
    "SupervisedExtrasConfig",
    "ensure_supervised_extras_config",
    "rollout_supervised_extras",
    "ExtrasGrowthConfig",
    "ensure_extras_growth_config",
    "extras_growth_to_metadata",
    "expand_extras_head",
]



