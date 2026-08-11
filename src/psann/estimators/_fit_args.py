from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..hisso import HISSOOptions
from ..types import HISSOFitParams, NoiseSpec
from ._fit_types import NormalisedFitArgs, ValidationInput

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


def normalise_fit_args(
    estimator: "PSANNRegressor",
    X: np.ndarray,
    y: Optional[np.ndarray],
    *,
    context: Optional[np.ndarray] = None,
    validation_data: Optional[ValidationInput],
    noisy: Optional[NoiseSpec],
    verbose: int,
    lr_max: Optional[float],
    lr_min: Optional[float],
    scheduler: str = "none",
    scheduler_params: Optional[Mapping[str, Any]] = None,
    nonfinite_policy: str = "error",
    fallback_policy: str = "warn",
    callback_error_policy: str = "raise",
    deterministic: bool = False,
    metrics: Optional[Mapping[str, Callable[..., Any]]] = None,
    callbacks: Optional[Sequence[Callable[..., Any]]] = None,
    logger: Optional[Any] = None,
    resume_from: Optional[str | os.PathLike[str]] = None,
    checkpoint_dir: Optional[str | os.PathLike[str]] = None,
    checkpoint_every: int = 0,
    checkpoint_keep: int = 3,
    hisso: bool,
    hisso_kwargs: HISSOFitParams,
) -> NormalisedFitArgs:
    """Coerce inputs, targets, and validation tuples into canonical form."""

    del estimator

    def _require_finite_array(name: str, value: np.ndarray) -> None:
        if np.isnan(value).any():
            raise ValueError(
                f"{name} contains missing values (NaN); PSANN's training data boundary "
                "requires missing values to be imputed before fit."
            )
        if np.isinf(value).any():
            raise ValueError(
                f"{name} contains infinite values; PSANN's training data boundary "
                "requires finite inputs and targets."
            )

    validation_value: Optional[ValidationInput] = None
    if validation_data is not None:
        if not isinstance(validation_data, (tuple, list)):
            raise TypeError(
                "validation_data must be a tuple/list (X, y) or (X, y, context); "
                f"received {type(validation_data).__name__}."
            )
        val_tuple = tuple(validation_data)
        if len(val_tuple) == 2:
            X_val = np.asarray(val_tuple[0], dtype=np.float32)
            y_val = np.asarray(val_tuple[1], dtype=np.float32)
            if X_val.ndim < 2:
                raise ValueError("validation_data X must be at least 2D.")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    "validation_data X and y must contain the same number of samples; "
                    f"received {X_val.shape[0]} and {y_val.shape[0]}."
                )
            _require_finite_array("validation_data X", X_val)
            _require_finite_array("validation_data y", y_val)
            validation_value = (X_val, y_val)
        elif len(val_tuple) == 3:
            X_val = np.asarray(val_tuple[0], dtype=np.float32)
            y_val = np.asarray(val_tuple[1], dtype=np.float32)
            ctx_val = np.asarray(val_tuple[2], dtype=np.float32)
            if X_val.ndim < 2:
                raise ValueError("validation_data X must be at least 2D.")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    "validation_data X and y must contain the same number of samples; "
                    f"received {X_val.shape[0]} and {y_val.shape[0]}."
                )
            if ctx_val.ndim == 1:
                ctx_val = ctx_val.reshape(-1, 1)
            if ctx_val.shape[0] != X_val.shape[0]:
                raise ValueError(
                    f"validation context has {ctx_val.shape[0]} samples but X has {X_val.shape[0]}."
                )
            _require_finite_array("validation_data X", X_val)
            _require_finite_array("validation_data y", y_val)
            _require_finite_array("validation context", ctx_val)
            validation_value = (X_val, y_val, ctx_val)
        else:
            raise ValueError(
                f"validation_data must contain 2 or 3 elements; received {len(val_tuple)}."
            )

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32) if y is not None else None
    if X_arr.ndim < 2:
        raise ValueError(f"X must be at least 2D (batch, features...); got {X_arr.shape}.")
    if X_arr.shape[0] < 1:
        raise ValueError("X must contain at least one sample.")
    _require_finite_array("X", X_arr)
    if y_arr is not None:
        if y_arr.ndim < 1:
            raise ValueError("y must include a batch dimension.")
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples; "
                f"received {X_arr.shape[0]} and {y_arr.shape[0]}."
            )
        _require_finite_array("y", y_arr)

    context_arr: Optional[np.ndarray] = None
    if context is not None:
        ctx = np.asarray(context, dtype=np.float32)
        if ctx.ndim == 1:
            ctx = ctx.reshape(-1, 1)
        if ctx.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"context has {ctx.shape[0]} samples but X has {X_arr.shape[0]}; dimensions must match."
            )
        _require_finite_array("context", ctx)
        context_arr = ctx

    if not hisso and y_arr is None:
        raise ValueError("y must be provided when hisso=False")

    noise_cfg: Optional[NoiseSpec] = None
    if noisy is not None:
        if np.isscalar(noisy):
            noise_cfg = float(noisy)
            if not np.isfinite(noise_cfg) or noise_cfg < 0:
                raise ValueError("noisy must be finite and >= 0.")
        else:
            noise_cfg = np.asarray(noisy, dtype=np.float32)
            _require_finite_array("noisy", noise_cfg)
            if (noise_cfg < 0).any():
                raise ValueError("noisy values must be >= 0.")

    hisso_options: Optional[HISSOOptions] = None
    if hisso:
        hisso_options = HISSOOptions.from_kwargs(
            window=hisso_kwargs.get("hisso_window"),
            batch_episodes=hisso_kwargs.get("hisso_batch_episodes"),
            updates_per_epoch=hisso_kwargs.get("hisso_updates_per_epoch"),
            reward_fn=hisso_kwargs.get("hisso_reward_fn"),
            context_extractor=hisso_kwargs.get("hisso_context_extractor"),
            primary_transform=hisso_kwargs.get("hisso_primary_transform"),
            transition_penalty=hisso_kwargs.get("hisso_transition_penalty"),
            trans_cost=hisso_kwargs.get("hisso_trans_cost"),
            input_noise=noise_cfg,
            supervised=hisso_kwargs.get("hisso_supervised"),
        )

    if callbacks is None:
        callbacks_value: tuple[Callable[..., Any], ...] = ()
    elif callable(callbacks):
        callbacks_value = (callbacks,)
    else:
        callbacks_value = tuple(callbacks)

    return NormalisedFitArgs(
        X=X_arr,
        y=y_arr,
        context=context_arr,
        validation=validation_value,
        hisso=bool(hisso),
        hisso_options=hisso_options,
        noisy=noise_cfg,
        verbose=int(verbose),
        lr_max=float(lr_max) if lr_max is not None else None,
        lr_min=float(lr_min) if lr_min is not None else None,
        scheduler=str(scheduler).strip().lower(),
        scheduler_params=dict(scheduler_params or {}),
        nonfinite_policy=str(nonfinite_policy).strip().lower(),
        fallback_policy=str(fallback_policy).strip().lower(),
        callback_error_policy=str(callback_error_policy).strip().lower(),
        deterministic=bool(deterministic),
        metrics=dict(metrics or {}),
        callbacks=callbacks_value,
        logger=logger,
        resume_from=os.fspath(resume_from) if resume_from is not None else None,
        checkpoint_dir=os.fspath(checkpoint_dir) if checkpoint_dir is not None else None,
        checkpoint_every=int(checkpoint_every),
        checkpoint_keep=int(checkpoint_keep),
    )
