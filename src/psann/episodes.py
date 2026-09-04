"""Deprecated model-level episode facade backed by the canonical runtime."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn as nn

from .episodic.legacy_config import HISSOTrainerConfig
from .episodic.rewards import multiplicative_return_reward
from .episodic.runtime_loop import HISSOTrainer
from .utils import choose_device


class RewardStrategy(Protocol):
    def __call__(
        self, actions: torch.Tensor, context: torch.Tensor, **kwargs: object
    ) -> torch.Tensor: ...


def portfolio_log_return_reward(
    allocations: torch.Tensor, prices: torch.Tensor, *, trans_cost: float = 0.0, eps: float = 1e-8
) -> torch.Tensor:
    """Retained legacy spelling for the canonical finance reward."""

    return multiplicative_return_reward(allocations, prices, transition_penalty=trans_cost, eps=eps)


@dataclass
class EpisodeConfig:
    episode_length: int
    batch_episodes: int = 32
    allocation_transform: str = "softmax"
    trans_cost: float = 0.0
    transition_penalty: float | None = None
    spatial_pool: str = "mean"
    random_state: int | None = None

    def __post_init__(self) -> None:
        warnings.warn(
            "psann.episodes.EpisodeConfig is deprecated; use psann.episodic.HISSOConfig.",
            DeprecationWarning,
            stacklevel=2,
        )

    def resolved_transition_penalty(self) -> float:
        return float(
            self.trans_cost if self.transition_penalty is None else self.transition_penalty
        )


class EpisodeTrainer:
    """0.x adapter over :class:`psann.episodic.runtime_loop.HISSOTrainer`.

    It deliberately owns no sampler, transform, reward dispatch, state lifecycle, or
    optimizer step.  Historical model-level constructor choices are translated into
    the runtime's compatibility policy.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        reward_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = multiplicative_return_reward,
        ep_cfg: EpisodeConfig,
        device: torch.device | str = "auto",
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
        grad_clip: float | None = None,
        price_extractor: Callable[[torch.Tensor], torch.Tensor] | None = None,
        context_extractor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        warnings.warn(
            "psann.episodes.EpisodeTrainer is deprecated; use psann.episodic.EpisodicTrainer.",
            DeprecationWarning,
            stacklevel=2,
        )
        if price_extractor is not None and context_extractor is None:
            warnings.warn(
                "EpisodeTrainer(price_extractor=...) is deprecated; use context_extractor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.model = model
        self.cfg = ep_cfg
        self.device = choose_device(device)
        self.context_extractor = context_extractor or price_extractor
        self.price_extractor = price_extractor
        transform = ep_cfg.allocation_transform
        if transform in {"relu_norm", "relu-normalize", "sparse"}:
            transform = "relu_norm"
        runtime_cfg = HISSOTrainerConfig(
            episode_length=ep_cfg.episode_length,
            episodes_per_batch=ep_cfg.batch_episodes,
            primary_transform=transform,
            random_state=ep_cfg.random_state,
            transition_penalty=ep_cfg.resolved_transition_penalty(),
        )
        self._runtime = HISSOTrainer(
            model,
            cfg=runtime_cfg,
            device=self.device,
            lr=lr,
            reward_fn=reward_fn,
            context_extractor=self.context_extractor,
            input_noise_std=None,
            optimizer=optimizer,
            gradient_clip=grad_clip,
            strict=False,
        )
        self.opt = self._runtime.optimizer

    def train(self, X: np.ndarray, *, epochs: int = 100, verbose: int = 1) -> None:
        self._runtime.train(X, epochs=epochs, verbose=verbose, lr_max=None, lr_min=None)

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, *, n_batches: int = 16) -> float:
        data = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device)
        values: list[float] = []
        self.model.eval()
        for _ in range(n_batches):
            episodes, _ = self._runtime._sample_episode_batch(
                data,
                total_steps=data.shape[0],
                episode_length=min(self.cfg.episode_length, data.shape[0]),
                count=self.cfg.batch_episodes,
            )
            context = self._runtime._extract_context(episodes)
            outputs = self.model(
                episodes.reshape(episodes.shape[0] * episodes.shape[1], *episodes.shape[2:])
            )
            if outputs.ndim == 1:
                outputs = outputs[:, None]
            outputs = outputs.reshape(episodes.shape[0], episodes.shape[1], -1)
            values.append(
                float(
                    self._runtime._coerce_reward(
                        self._runtime._apply_primary_transform(outputs), context
                    ).mean()
                )
            )
        return float(np.mean(values))


def make_episode_trainer_from_estimator(
    est: object,
    *,
    ep_cfg: EpisodeConfig,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = multiplicative_return_reward,
    device: torch.device | str = "auto",
    lr: float = 1e-3,
) -> EpisodeTrainer:
    if not hasattr(est, "model_"):
        raise RuntimeError("Estimator not fitted; call fit() first or attach .model_ manually.")
    return EpisodeTrainer(est.model_, reward_fn=reward_fn, ep_cfg=ep_cfg, device=device, lr=lr)


__all__ = [
    "EpisodeConfig",
    "EpisodeTrainer",
    "RewardStrategy",
    "make_episode_trainer_from_estimator",
    "multiplicative_return_reward",
    "portfolio_log_return_reward",
]
