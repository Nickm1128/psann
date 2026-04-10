from __future__ import annotations

from typing import TYPE_CHECKING, Optional

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
    hisso: bool,
    hisso_kwargs: HISSOFitParams,
) -> NormalisedFitArgs:
    """Coerce inputs, targets, and validation tuples into canonical form."""

    del estimator

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
            validation_value = (X_val, y_val)
        elif len(val_tuple) == 3:
            X_val = np.asarray(val_tuple[0], dtype=np.float32)
            y_val = np.asarray(val_tuple[1], dtype=np.float32)
            ctx_val = np.asarray(val_tuple[2], dtype=np.float32)
            if ctx_val.ndim == 1:
                ctx_val = ctx_val.reshape(-1, 1)
            if ctx_val.shape[0] != X_val.shape[0]:
                raise ValueError(
                    f"validation context has {ctx_val.shape[0]} samples but X has {X_val.shape[0]}."
                )
            validation_value = (X_val, y_val, ctx_val)
        else:
            raise ValueError(
                f"validation_data must contain 2 or 3 elements; received {len(val_tuple)}."
            )

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32) if y is not None else None

    context_arr: Optional[np.ndarray] = None
    if context is not None:
        ctx = np.asarray(context, dtype=np.float32)
        if ctx.ndim == 1:
            ctx = ctx.reshape(-1, 1)
        if ctx.shape[0] != X_arr.shape[0]:
            raise ValueError(
                f"context has {ctx.shape[0]} samples but X has {X_arr.shape[0]}; dimensions must match."
            )
        context_arr = ctx

    if not hisso and y_arr is None:
        raise ValueError("y must be provided when hisso=False")

    noise_cfg: Optional[NoiseSpec] = None
    if noisy is not None:
        if np.isscalar(noisy):
            noise_cfg = float(noisy)
        else:
            noise_cfg = np.asarray(noisy, dtype=np.float32)

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
    )
