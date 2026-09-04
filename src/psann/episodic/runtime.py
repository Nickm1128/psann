"""Shared canonical episodic transform and evaluation primitives."""

from __future__ import annotations

import inspect
from typing import Callable, cast

import numpy as np
import torch


def transform_action_tensor(values: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply the canonical action transform without changing device or dtype.

    Every episodic lifecycle entry point goes through this primitive.  Keeping the
    tensor implementation here prevents the training loop, canonical wrapper, and
    retained compatibility helpers from growing subtly different softmax axes or
    numerical implementations.
    """

    tensor = values
    # A rank-one estimator result denotes one decision per sample, not one
    # vector of decisions across the complete sample batch.  Preserve that
    # sample axis while applying the final-width transform.
    column_output = tensor.ndim == 1
    if column_output:
        tensor = tensor[:, None]
    if transform == "identity":
        result = tensor
    elif transform == "softmax":
        result = torch.softmax(tensor, dim=-1)
    elif transform == "tanh":
        result = torch.tanh(tensor)
    elif transform in {"relu_norm", "relu-normalize", "sparse"}:
        positive = torch.relu(tensor) + 1e-8
        result = positive / positive.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(f"strategy.primary_transform {transform!r} is unsupported.")
    if column_output:
        result = result[:, 0]
    return result


def transform_actions(values: np.ndarray, transform: str) -> np.ndarray:
    """NumPy boundary for :func:`transform_action_tensor`."""

    tensor = torch.as_tensor(values, dtype=torch.float32)
    return transform_action_tensor(tensor, transform).detach().cpu().numpy()


def align_context(actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    if actions.shape[:-1] != context.shape[:-1]:
        raise ValueError(
            f"strategy.context_extractor batch/time mismatch: {tuple(actions.shape)} vs {tuple(context.shape)}."
        )
    if actions.shape[-1] == context.shape[-1]:
        return context
    if context.shape[-1] == 1:
        return context.expand(*context.shape[:-1], actions.shape[-1])
    if actions.shape[-1] == 1:
        return context.mean(dim=-1, keepdim=True)
    raise ValueError(
        f"strategy.context_extractor width mismatch: {tuple(actions.shape)} vs {tuple(context.shape)}."
    )


def call_reward(
    reward: object, actions: torch.Tensor, context: torch.Tensor, penalty: float
) -> torch.Tensor:
    callable_reward = cast(Callable[..., object], reward)
    signature = inspect.signature(callable_reward)
    kwargs: dict[str, object] = {}
    if "transition_penalty" in signature.parameters or any(
        p.kind == p.VAR_KEYWORD for p in signature.parameters.values()
    ):
        kwargs["transition_penalty"] = penalty
    elif "trans_cost" in signature.parameters:
        kwargs["trans_cost"] = penalty
    elif penalty:
        raise ValueError(
            "strategy.transition_penalty requires a reward accepting transition_penalty or trans_cost."
        )
    output = callable_reward(actions, context, **kwargs)
    if not isinstance(output, torch.Tensor):
        raise TypeError("strategy.reward must return a torch.Tensor.")
    return output


def validate_reward_penalty(reward: object, penalty: float) -> None:
    """Reject an unconsumable canonical penalty before model construction."""

    if not penalty:
        return
    callable_reward = cast(Callable[..., object], reward)
    signature = inspect.signature(callable_reward)
    if "transition_penalty" in signature.parameters or "trans_cost" in signature.parameters:
        return
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return
    raise ValueError(
        "strategy.transition_penalty requires a reward accepting transition_penalty or trans_cost."
    )


__all__ = [
    "align_context",
    "call_reward",
    "transform_action_tensor",
    "transform_actions",
    "validate_reward_penalty",
]
