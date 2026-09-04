"""LM sampling, resolved through the public core component interface."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import cast

import torch
from torch import nn

from psann.architectures import ActivationConfig
from psann.architectures.components import build_activation

from .config import LMActivationInitializationConfig, LMArchitectureConfig


def sample_initial_values(
    config: ActivationConfig, initialization: LMActivationInitializationConfig | None, features: int
) -> dict[str, float | torch.Tensor]:
    values: dict[str, float | torch.Tensor] = {}
    for name in ("amplitude", "frequency", "decay"):
        mean = getattr(config, name + "_init")
        std = getattr(initialization, name + "_std", 0.0)
        rng = getattr(initialization, name + "_range", None)
        if std > 0:
            value = torch.randn(features, dtype=torch.float32) * std + mean
            values[name] = value.clamp_min(torch.finfo(value.dtype).eps)
        elif rng is not None:
            values[name] = random.uniform(*rng)
        else:
            values[name] = mean
    return values


def build_lm_activation(
    architecture: LMArchitectureConfig, features: int, *, block_index: int = 0
) -> nn.Module:
    config = architecture.activation
    initialization = architecture.activation_initialization
    if config.kind == "mixed":
        seed = None if config.mix_seed is None else config.mix_seed + block_index * 9973
        result = build_activation(replace(config, mix_seed=seed), features=features)
        # Mixed composition assigns child widths before constructing their parameters.
        # Replacing a scalar-initialized child consumes no extra global RNG draws.
        acts = cast(nn.ModuleDict, result.get_submodule("acts"))
        if initialization is not None and "psann" in acts:
            child = acts["psann"]
            child_features = cast(int, getattr(child, "out_features"))
            child_config = ActivationConfig(
                amplitude_init=config.amplitude_init,
                frequency_init=config.frequency_init,
                decay_init=config.decay_init,
                learnable=config.learnable,
                decay_mode=config.decay_mode,
                bounds=config.bounds,
            )
            acts["psann"] = build_activation(
                child_config,
                features=child_features,
                initial_values=sample_initial_values(child_config, initialization, child_features),
            )
        return result
    if config.kind == "psann":
        return build_activation(
            config,
            features=features,
            initial_values=sample_initial_values(config, initialization, features),
        )
    return build_activation(config, features=features)
