"""Narrow public protocols used by the canonical episodic boundary."""

from __future__ import annotations

from typing import Protocol

import torch


class RewardStrategy(Protocol):
    """Maps aligned ``(episodes, steps, width)`` tensors to episode rewards."""

    def __call__(
        self, actions: torch.Tensor, context: torch.Tensor, **kwargs: object
    ) -> torch.Tensor: ...


class ContextExtractor(Protocol):
    """Derives tensor-native reward context from prepared episode inputs."""

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...


__all__ = ["ContextExtractor", "RewardStrategy"]
