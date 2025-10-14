"""
Synthetic datasets for quick PSANN experiments.
"""

from __future__ import annotations

from typing import Tuple

import torch


def make_context_rotating_moons(
    n: int, *, noise: float = 0.05, seed: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a rotating moons dataset with context-conditioned rotation."""
    if n <= 0:
        raise ValueError("n must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    theta = torch.rand(n, generator=generator) * torch.pi
    labels = (torch.rand(n, generator=generator) > 0.5).long()

    base0 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
    base1 = torch.stack([1 - torch.cos(theta), 1 - torch.sin(theta) - 0.5], dim=-1)
    base = torch.where(labels.unsqueeze(-1) == 0, base0, base1)

    contexts = torch.rand(n, 1, generator=generator) * 2 * torch.pi - torch.pi
    cos_a = torch.cos(contexts.squeeze(-1))
    sin_a = torch.sin(contexts.squeeze(-1))
    rotation = torch.stack(
        [cos_a, -sin_a, sin_a, cos_a],
        dim=-1,
    ).reshape(n, 2, 2)
    features = torch.bmm(rotation, base.unsqueeze(-1)).squeeze(-1)

    if noise > 0:
        features = features + noise * torch.randn(features.shape, generator=generator)

    return features, labels, contexts


def make_regime_switch_ts(
    T: int, *, regimes: int = 3, seed: int | None = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a 1D time-series with regime switches driven by context."""
    if T <= 0:
        raise ValueError("T must be positive.")
    if regimes <= 0:
        raise ValueError("regimes must be positive.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    contexts = []
    series = []
    state = torch.zeros(1)

    regime_functions = [
        lambda h: torch.sin(1.4 * h),
        lambda h: torch.tanh(0.8 * h) + 0.25 * h,
        lambda h: torch.sin(h) + 0.3 * torch.cos(0.5 * h),
    ]

    base_length = T // regimes
    extras = T % regimes

    for idx in range(regimes):
        steps = base_length + (1 if idx < extras else 0)
        func = regime_functions[idx % len(regime_functions)]
        context = torch.zeros(regimes)
        context[idx] = 1.0
        for _ in range(steps):
            contexts.append(context.clone())
            state = func(state) + 0.05 * torch.randn_like(state, generator=generator)
            series.append(state.clone())

    return torch.stack(series, dim=0).squeeze(-1), torch.stack(contexts, dim=0)
