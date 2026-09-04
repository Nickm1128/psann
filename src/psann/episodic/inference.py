from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from .legacy_config import HISSOOptions, HISSOTrainerConfig
from .runtime import transform_actions
from .runtime_loop import HISSOTrainer

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


def _compat_runtime(
    estimator: "PSANNRegressor", cfg: HISSOTrainerConfig, options: Optional[HISSOOptions]
) -> HISSOTrainer:
    """Build the permissive legacy adapter around the one episodic runtime."""

    model_device = next(estimator.model_.parameters()).device
    return HISSOTrainer(
        estimator.model_,
        cfg=cfg,
        # A retained test compatibility seam can advertise CUDA while preserving
        # CPU model parameters.  Follow the model's actual placement here; normal
        # fitted CUDA models remain on CUDA.
        device=model_device,
        lr=0.0,
        reward_fn=(
            options.reward_fn
            if options is not None
            else getattr(estimator, "_hisso_reward_fn_", None)
        ),
        context_extractor=(
            options.context_extractor
            if options is not None
            else getattr(estimator, "_hisso_context_extractor_", None)
        ),
        input_noise_std=None,
        stateful=bool(getattr(estimator, "stateful", False)),
        state_reset=str(getattr(estimator, "state_reset", "batch")),
        strict=False,
    )


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
    return transform_actions(preds, _resolve_primary_transform(cfg, options) or "identity")


def hisso_evaluate_reward(
    estimator: "PSANNRegressor",
    X_obs: np.ndarray,
    *,
    trainer_cfg: Optional[HISSOTrainerConfig] = None,
) -> float:
    options = getattr(estimator, "_hisso_options_", None)
    if options is not None:
        reward_fn = options.reward_fn
    else:
        reward_fn = getattr(estimator, "_hisso_reward_fn_", None)

    if reward_fn is None:
        return 0.0

    cfg = _resolve_hisso_config(estimator, trainer_cfg)
    if cfg is None:
        raise RuntimeError("HISSO reward evaluation requires a fitted HISSO trainer configuration.")
    prepared, _, _ = estimator._prepare_inference_inputs(X_obs)
    return _compat_runtime(estimator, cfg, options).evaluate_prepared(prepared)
