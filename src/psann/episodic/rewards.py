"""Canonical differentiable reward registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, cast

import torch


class RewardStrategy(Protocol):
    def __call__(
        self, actions: torch.Tensor, context: torch.Tensor, **kwargs: object
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class RewardStrategyBundle:
    reward_fn: RewardStrategy
    metrics_fn: Callable[..., object] | None = None
    description: str = ""


def default_reward(
    actions: torch.Tensor, _context: torch.Tensor, **_kwargs: object
) -> torch.Tensor:
    return -actions.pow(2).mean(dim=-1)


def multiplicative_return_reward(
    actions: torch.Tensor,
    context: torch.Tensor,
    *,
    transition_penalty: float = 0.0,
    trans_cost: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if trans_cost is not None:
        transition_penalty = trans_cost
    if actions.ndim != 3 or context.ndim != 3:
        raise ValueError("reward actions and context must both be rank-3.")
    if actions.shape != context.shape:
        raise ValueError(
            f"reward actions/context shape mismatch: {tuple(actions.shape)} vs {tuple(context.shape)}."
        )
    if actions.shape[1] < 2:
        raise ValueError("reward episode_length must be at least 2.")
    returns = context[:, 1:] / (context[:, :-1] + eps) - 1.0
    growth = (actions[:, :-1] * (1.0 + returns)).sum(dim=-1).clamp_min(eps)
    reward = torch.log(growth).sum(dim=-1)
    if transition_penalty:
        reward = reward - float(transition_penalty) * (actions[:, 1:] - actions[:, :-1]).abs().sum(
            dim=-1
        ).sum(dim=-1)
    return reward


_REGISTRY: dict[str, RewardStrategyBundle] = {}


def register_reward_strategy(
    name: str, bundle: RewardStrategyBundle, *, overwrite: bool = False
) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("reward registry name must be non-empty.")
    if not isinstance(bundle, RewardStrategyBundle):
        raise TypeError("reward bundle must be a RewardStrategyBundle.")
    key = name.strip().lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"reward strategy {name!r} is already registered.")
    _REGISTRY[key] = bundle


def get_reward_strategy(name: str) -> RewardStrategyBundle:
    if not isinstance(name, str) or name.strip().lower() not in _REGISTRY:
        raise KeyError(f"unknown reward strategy {name!r}.")
    return _REGISTRY[name.strip().lower()]


def registered_reward_name(value: object) -> str | None:
    """Return a stable registry discriminator for an identical bundle/callable."""

    for name, bundle in _REGISTRY.items():
        if value is bundle or value is bundle.reward_fn:
            return name
    return None


register_reward_strategy(
    "default",
    RewardStrategyBundle(cast(RewardStrategy, default_reward), description="Activation penalty"),
)
FINANCE_PORTFOLIO_STRATEGY = RewardStrategyBundle(
    cast(RewardStrategy, multiplicative_return_reward), description="Portfolio returns"
)
register_reward_strategy("finance", FINANCE_PORTFOLIO_STRATEGY)
register_reward_strategy("portfolio", FINANCE_PORTFOLIO_STRATEGY)


def resolve_reward(value: object) -> RewardStrategy:
    if isinstance(value, str):
        return get_reward_strategy(value).reward_fn
    if isinstance(value, RewardStrategyBundle):
        return value.reward_fn
    if callable(value):
        return value
    raise TypeError("strategy.reward must resolve to a callable.")


__all__ = [
    "FINANCE_PORTFOLIO_STRATEGY",
    "RewardStrategy",
    "RewardStrategyBundle",
    "default_reward",
    "get_reward_strategy",
    "multiplicative_return_reward",
    "register_reward_strategy",
    "registered_reward_name",
    "resolve_reward",
]
