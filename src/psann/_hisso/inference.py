from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from .config import HISSOOptions, HISSOTrainerConfig
from .context import _call_context_extractor
from .reward import _align_context_for_reward, _compute_reward, _resolve_reward_kwarg

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


def _resolve_hisso_config(
    estimator: "PSANNRegressor",
    override: Optional[HISSOTrainerConfig],
) -> Optional[HISSOTrainerConfig]:
    if override is not None:
        return override
    return getattr(estimator, "_hisso_cfg_", None)


def _resolve_primary_transform(
    cfg: Optional[HISSOTrainerConfig],
    options: Optional[HISSOOptions],
) -> Optional[str]:
    if cfg is not None and cfg.primary_transform:
        return cfg.primary_transform
    if options is not None:
        return options.primary_transform
    return None


def _apply_primary_transform_numpy(values: np.ndarray, transform: Optional[str]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    squeeze = False
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        squeeze = True

    if transform is None:
        result = arr
    else:
        transform_lower = transform.lower()
        if transform_lower == "identity":
            result = arr
        elif transform_lower == "softmax":
            shifted = arr - arr.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            totals = exp.sum(axis=1, keepdims=True)
            np.clip(totals, a_min=np.finfo(np.float32).tiny, a_max=None, out=totals)
            result = exp / totals
        elif transform_lower == "tanh":
            result = np.tanh(arr)
        else:
            raise ValueError(f"Unsupported primary transform '{transform}'.")

    return result.squeeze(1) if squeeze else result


def hisso_infer_series(
    estimator: "PSANNRegressor",
    X_obs: np.ndarray,
    *,
    trainer_cfg: Optional[HISSOTrainerConfig] = None,
) -> np.ndarray:
    cfg = _resolve_hisso_config(estimator, trainer_cfg)
    if getattr(estimator, "stateful", False):
        preds = estimator.predict_sequence(X_obs, reset_state=True, return_sequence=True)
    else:
        preds = estimator.predict(X_obs)
    options = getattr(estimator, "_hisso_options_", None)
    return _apply_primary_transform_numpy(preds, _resolve_primary_transform(cfg, options))


def hisso_evaluate_reward(
    estimator: "PSANNRegressor",
    X_obs: np.ndarray,
    *,
    trainer_cfg: Optional[HISSOTrainerConfig] = None,
) -> float:
    options = getattr(estimator, "_hisso_options_", None)
    if options is not None:
        reward_fn = options.reward_fn
        context_extractor = options.context_extractor
    else:
        reward_fn = getattr(estimator, "_hisso_reward_fn_", None)
        context_extractor = getattr(estimator, "_hisso_context_extractor_", None)

    if reward_fn is None:
        return 0.0

    device = estimator._device()
    X_np = np.asarray(X_obs, dtype=np.float32)
    inputs_t = torch.from_numpy(X_np).to(device)

    cfg = _resolve_hisso_config(estimator, trainer_cfg)
    preds = estimator.predict(X_obs)
    primary_np = _apply_primary_transform_numpy(preds, _resolve_primary_transform(cfg, options))
    primary_t = torch.from_numpy(primary_np).to(device)

    context_t = _call_context_extractor(context_extractor, inputs_t)
    context_t = context_t.to(device=primary_t.device, dtype=primary_t.dtype)
    if context_t.ndim > 2:
        context_t = context_t.reshape(context_t.shape[0], -1)
    context_t = _align_context_for_reward(primary_t, context_t)

    transition_penalty = 0.0
    if cfg is not None:
        transition_penalty = cfg.resolved_transition_penalty()
    elif options is not None:
        transition_penalty = float(options.transition_penalty or 0.0)
    reward = _compute_reward(
        reward_fn,
        _resolve_reward_kwarg(reward_fn),
        primary_t,
        context_t,
        transition_penalty=transition_penalty,
    )
    if isinstance(reward, torch.Tensor):
        return float(reward.mean().detach().cpu().item())
    return float(reward)
