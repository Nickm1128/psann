from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .episodes import RewardStrategy, multiplicative_return_reward
from .metrics import portfolio_metrics

MetricsFn = Callable[[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor], Dict[str, float]]


@dataclass(frozen=True)
class RewardStrategyBundle:
    """Grouped reward configuration with optional evaluation metrics."""

    reward_fn: RewardStrategy
    metrics_fn: Optional[MetricsFn] = None
    description: str = ""


_STRATEGY_REGISTRY: Dict[str, RewardStrategyBundle] = {}


def register_reward_strategy(name: str, bundle: RewardStrategyBundle, *, overwrite: bool = False) -> None:
    """Register a named reward strategy bundle.

    Parameters
    ----------
    name : str
        Registry key; stored in lowercase.
    bundle : RewardStrategyBundle
        Reward + optional metrics definition to register.
    overwrite : bool, default False
        If ``True`` existing entries may be replaced; otherwise duplicates raise.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("Strategy name must be non-empty")
    if key in _STRATEGY_REGISTRY and not overwrite:
        raise ValueError(f"Reward strategy '{name}' is already registered")
    _STRATEGY_REGISTRY[key] = bundle


def get_reward_strategy(name: str) -> RewardStrategyBundle:
    """Fetch a previously registered reward strategy bundle."""
    key = name.strip().lower()
    if not key or key not in _STRATEGY_REGISTRY:
        raise KeyError(f"Unknown reward strategy '{name}'")
    return _STRATEGY_REGISTRY[key]


FINANCE_PORTFOLIO_STRATEGY = RewardStrategyBundle(
    reward_fn=multiplicative_return_reward,
    metrics_fn=portfolio_metrics,
    description=(
        "Portfolio-style multiplicative returns with optional transition penalties; "
        "pairs the reward with `portfolio_metrics` for evaluation."
    ),
)

register_reward_strategy("finance", FINANCE_PORTFOLIO_STRATEGY)
register_reward_strategy("portfolio", FINANCE_PORTFOLIO_STRATEGY)
