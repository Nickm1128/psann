from __future__ import annotations

import inspect
import math
from typing import Any, Optional

import torch

from ..types import RewardFn


def _align_context_for_reward(actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Match context shape to the primary actions for reward computation."""

    if context.shape == actions.shape:
        return context

    if context.shape[:-1] != actions.shape[:-1]:
        raise ValueError(
            "HISSO reward expects actions/context to share batch/time dimensions; "
            f"got {tuple(actions.shape)} vs {tuple(context.shape)}."
        )

    target_dim = actions.shape[-1]
    ctx_dim = context.shape[-1]

    if target_dim == ctx_dim:
        return context

    if target_dim == 1:
        return context.mean(dim=-1, keepdim=True)

    if ctx_dim == 1:
        return context.expand(*context.shape[:-1], target_dim)

    if ctx_dim > target_dim:
        return context[..., :target_dim]

    repeats = math.ceil(target_dim / ctx_dim)
    expanded = context.repeat_interleave(repeats, dim=-1)
    return expanded[..., :target_dim]


def _resolve_reward_kwarg(reward_fn: RewardFn) -> Optional[str]:
    """Detect whether the reward accepts transition penalty keywords."""

    try:
        sig = inspect.signature(reward_fn)
    except (TypeError, ValueError):
        return None

    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return "transition_penalty"

    transition = params.get("transition_penalty")
    if transition is not None and transition.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return "transition_penalty"

    legacy = params.get("trans_cost")
    if legacy is not None and legacy.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return "trans_cost"

    return None


def _compute_reward(
    reward_fn: RewardFn,
    reward_kwarg: Optional[str],
    actions: torch.Tensor,
    context: torch.Tensor,
    *,
    transition_penalty: float,
) -> Any:
    """Call reward functions with compatibility for transition penalty aliases."""

    if reward_kwarg is None:
        return reward_fn(actions, context)
    penalty = float(transition_penalty or 0.0)
    if reward_kwarg == "transition_penalty":
        return reward_fn(actions, context, transition_penalty=penalty)
    if reward_kwarg == "trans_cost":
        return reward_fn(actions, context, trans_cost=penalty)
    return reward_fn(actions, context, **{reward_kwarg: penalty})


def _default_reward_fn(primary: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
    """Fallback reward that penalises large activations."""

    return -primary.pow(2).mean(dim=-1)
