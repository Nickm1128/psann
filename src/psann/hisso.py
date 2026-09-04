from __future__ import annotations

"""Public HISSO helpers and trainer types."""

import warnings
from typing import Any, cast

from .episodic.legacy_config import (
    HISSOOptions,
    HISSOTrainerConfig,
    HISSOWarmStartConfig,
    coerce_warmstart_config as _coerce_warmstart_config,
    ensure_hisso_trainer_config as _ensure_hisso_trainer_config,
)
from .episodic.inference import hisso_evaluate_reward as _hisso_evaluate_reward
from .episodic.inference import hisso_infer_series as _hisso_infer_series
from .episodic.runtime_loop import HISSOTrainer, run_hisso_training as _run_hisso_training
from .episodic.warmstart import run_hisso_supervised_warmstart as _run_hisso_supervised_warmstart


def _warn() -> None:
    warnings.warn(
        "psann.hisso is deprecated; use psann.episodic.EpisodicTrainer and HISSOConfig.",
        DeprecationWarning,
        stacklevel=3,
    )


def coerce_warmstart_config(*args: object, **kwargs: object):
    _warn()
    return cast(Any, _coerce_warmstart_config)(*args, **kwargs)


def ensure_hisso_trainer_config(*args: object, **kwargs: object):
    _warn()
    return cast(Any, _ensure_hisso_trainer_config)(*args, **kwargs)


def hisso_evaluate_reward(*args: object, **kwargs: object):
    _warn()
    return cast(Any, _hisso_evaluate_reward)(*args, **kwargs)


def hisso_infer_series(*args: object, **kwargs: object):
    _warn()
    return cast(Any, _hisso_infer_series)(*args, **kwargs)


def run_hisso_training(*args: object, **kwargs: object):
    _warn()
    return cast(Any, _run_hisso_training)(*args, **kwargs)


def run_hisso_supervised_warmstart(*args: object, **kwargs: object):
    _warn()
    return cast(Any, _run_hisso_supervised_warmstart)(*args, **kwargs)


__all__ = [
    "HISSOWarmStartConfig",
    "HISSOOptions",
    "HISSOTrainer",
    "HISSOTrainerConfig",
    "coerce_warmstart_config",
    "ensure_hisso_trainer_config",
    "hisso_evaluate_reward",
    "hisso_infer_series",
    "run_hisso_supervised_warmstart",
    "run_hisso_training",
]

for _cls in (HISSOWarmStartConfig, HISSOOptions, HISSOTrainerConfig, HISSOTrainer):
    _cls.__module__ = __name__
