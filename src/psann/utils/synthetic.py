"""
Synthetic datasets for quick PSANN experiments.
"""

from __future__ import annotations

from typing import Tuple
import math

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


def make_drift_series(
    T: int,
    *,
    drift: float = 0.001,
    frequency: float = 0.02,
    noise: float = 0.02,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a univariate regression series with gradually drifting amplitude."""

    if T < 2:
        raise ValueError("T must be at least 2 to form supervised pairs.")
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    steps = torch.arange(T + 1, dtype=torch.float32)
    amplitude = 1.0 + drift * steps
    phase = 2.0 * math.pi * frequency * steps
    base = amplitude * torch.sin(phase)
    noise_term = noise * torch.randn(T + 1, generator=generator)
    series = base + noise_term

    X = series[:-1].unsqueeze(-1)
    y = series[1:].unsqueeze(-1)
    return X, y


def make_shock_series(
    T: int,
    *,
    shock_prob: float = 0.05,
    shock_scale: float = 2.0,
    noise: float = 0.05,
    mean_revert: float = 0.85,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a regression series with sporadic shocks and mean reversion."""

    if T < 2:
        raise ValueError("T must be at least 2 to form supervised pairs.")
    if not (0.0 <= shock_prob <= 1.0):
        raise ValueError("shock_prob must lie in [0, 1].")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    values = torch.zeros(T + 1, dtype=torch.float32)
    carry = torch.zeros(1, dtype=torch.float32)
    for idx in range(T):
        if torch.rand(1, generator=generator).item() < shock_prob:
            shock = shock_scale * torch.randn(1, generator=generator)
        else:
            shock = torch.zeros(1, dtype=torch.float32)
        carry = mean_revert * carry + noise * torch.randn(1, generator=generator) + shock
        values[idx + 1] = values[idx] + carry

    X = values[:-1].unsqueeze(-1)
    y = values[1:].unsqueeze(-1)
    return X, y
