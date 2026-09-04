"""Deprecated 0.x façade for canonical episodic reward dispatch."""

from ..episodic.reward import (
    _align_context_for_reward,
    _compute_reward,
    _default_reward_fn,
    _resolve_reward_kwarg,
)

__all__ = [
    "_align_context_for_reward",
    "_compute_reward",
    "_default_reward_fn",
    "_resolve_reward_kwarg",
]
