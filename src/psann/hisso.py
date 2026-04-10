from __future__ import annotations

"""Public HISSO helpers and trainer types."""

from ._hisso.config import (
    HISSOOptions,
    HISSOTrainerConfig,
    HISSOWarmStartConfig,
    coerce_warmstart_config,
    ensure_hisso_trainer_config,
)
from ._hisso.inference import hisso_evaluate_reward, hisso_infer_series
from ._hisso.trainer import HISSOTrainer, run_hisso_training
from ._hisso.warmstart import run_hisso_supervised_warmstart

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
