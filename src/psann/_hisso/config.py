"""Deprecated 0.x façade for episodic legacy-configuration adapters."""

from ..episodic.legacy_config import (
    HISSOOptions,
    HISSOTrainerConfig,
    HISSOWarmStartConfig,
    coerce_warmstart_config,
    ensure_hisso_trainer_config,
)

__all__ = [
    "HISSOOptions",
    "HISSOTrainerConfig",
    "HISSOWarmStartConfig",
    "coerce_warmstart_config",
    "ensure_hisso_trainer_config",
]
