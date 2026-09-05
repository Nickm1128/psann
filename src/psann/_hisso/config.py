"""Deprecated legacy façade for episodic compatibility-configuration adapters."""

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
