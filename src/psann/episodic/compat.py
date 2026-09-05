"""Small conversion helpers for retained legacy HISSO call sites.

The helpers intentionally translate values only; model preparation remains owned by
the estimator and the canonical strategy remains free of legacy option objects.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, cast

from .config import EpisodeScheduleConfig, HISSOConfig, SupervisedWarmStartConfig


def legacy_hisso_strategy(
    *,
    window: int | None,
    batch_episodes: int | None,
    updates_per_epoch: int | None,
    reward: object | None,
    context_extractor: object | None,
    primary_transform: str | None,
    transition_penalty: float | None,
    trans_cost: float | None,
    noisy: object | None,
    supervised: Mapping[str, object] | bool | None,
) -> HISSOConfig:
    """Map accepted flat HISSO inputs to the unambiguous canonical schedule."""

    if (
        transition_penalty is not None
        and trans_cost is not None
        and transition_penalty != trans_cost
    ):
        raise ValueError("hisso_transition_penalty conflicts with hisso_trans_cost.")
    if trans_cost is not None:
        warnings.warn(
            "hisso_trans_cost is deprecated; use hisso_transition_penalty.",
            DeprecationWarning,
            stacklevel=2,
        )
    if noisy is not None and not isinstance(noisy, (int, float)):
        warnings.warn(
            "Non-scalar noisy is retained only for the legacy HISSO route and is ignored.",
            RuntimeWarning,
            stacklevel=2,
        )
        noisy = None
    # The omitted legacy update count meant one episode per update, repeated
    # ``hisso_batch_episodes`` times.  Preserve that behavior precisely.
    if updates_per_epoch is None:
        schedule = EpisodeScheduleConfig(
            episode_length=64 if window is None else window,
            batch_episodes=1,
            updates_per_epoch=32 if batch_episodes is None else batch_episodes,
        )
    else:
        schedule = EpisodeScheduleConfig(
            episode_length=64 if window is None else window,
            batch_episodes=32 if batch_episodes is None else batch_episodes,
            updates_per_epoch=updates_per_epoch,
        )
    warm_start = None
    if supervised:
        raw = {} if isinstance(supervised, bool) else dict(supervised)
        raw.pop("y", None)
        raw.pop("targets", None)
        if "lsm_lr" in raw:
            if "preprocessor_lr" in raw and raw["preprocessor_lr"] != raw["lsm_lr"]:
                raise ValueError("hisso_supervised.lsm_lr conflicts with preprocessor_lr.")
            raw["preprocessor_lr"] = raw.pop("lsm_lr")
        raw.pop("verbose", None)
        warm_start = SupervisedWarmStartConfig(**cast(Any, raw))
    return HISSOConfig(
        schedule=schedule,
        reward="default" if reward is None else reward,
        context_extractor=context_extractor,
        primary_transform="identity" if primary_transform is None else primary_transform,
        transition_penalty=(
            transition_penalty if transition_penalty is not None else (trans_cost or 0.0)
        ),
        input_noise_std=noisy,
        warm_start=warm_start,
    )


__all__ = ["legacy_hisso_strategy"]
