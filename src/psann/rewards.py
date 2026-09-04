"""Deprecated 0.x facade for :mod:`psann.episodic.rewards`.

The registry has one owner. Keeping this module deliberately small avoids a
second registry whose names can silently diverge from canonical episodic runs.
"""

from __future__ import annotations

import warnings

from .episodic.rewards import (
    FINANCE_PORTFOLIO_STRATEGY,
    RewardStrategy,
    RewardStrategyBundle,
    _REGISTRY as _STRATEGY_REGISTRY,  # noqa: F401 - retained explicit legacy import
    get_reward_strategy as _get_reward_strategy,
    register_reward_strategy as _register_reward_strategy,
)


def register_reward_strategy(
    name: str, bundle: RewardStrategyBundle, *, overwrite: bool = False
) -> None:
    """Register through the canonical episodic registry (deprecated)."""

    warnings.warn(
        "psann.rewards is deprecated; use psann.episodic.register_reward_strategy.",
        DeprecationWarning,
        stacklevel=2,
    )
    _register_reward_strategy(name, bundle, overwrite=overwrite)


def get_reward_strategy(name: str) -> RewardStrategyBundle:
    """Read the canonical episodic registry (deprecated)."""

    warnings.warn(
        "psann.rewards is deprecated; use psann.episodic.get_reward_strategy.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _get_reward_strategy(name)


__all__ = [
    "FINANCE_PORTFOLIO_STRATEGY",
    "RewardStrategy",
    "RewardStrategyBundle",
    "get_reward_strategy",
    "register_reward_strategy",
]
