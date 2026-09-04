"""The canonical estimator-wrapping episodic trainer."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .config import HISSOConfig, normalize_strategy, replace_strategy
from .rewards import resolve_reward
from .runtime import align_context, call_reward, transform_actions


class EpisodicTrainer:
    def __init__(
        self, *, estimator: object, strategy: HISSOConfig | Mapping[str, object] | str = "hisso"
    ) -> None:
        self.estimator = estimator
        self.strategy = strategy

    def get_params(self, deep: bool = True) -> dict[str, object]:
        params: dict[str, object] = {"estimator": self.estimator, "strategy": self.strategy}
        if deep:
            resolved = normalize_strategy(self.strategy)
            for field in fields(HISSOConfig):
                params[f"strategy__{field.name}"] = getattr(resolved, field.name)
            for field in fields(resolved.schedule):
                params[f"strategy__schedule__{field.name}"] = getattr(resolved.schedule, field.name)
            if resolved.warm_start is not None:
                for field in fields(resolved.warm_start):
                    params[f"strategy__warm_start__{field.name}"] = getattr(
                        resolved.warm_start, field.name
                    )
        return params

    def set_params(self, **params: object) -> "EpisodicTrainer":
        strategy = normalize_strategy(self.strategy)
        estimator = self.estimator
        for name, value in params.items():
            if name == "estimator":
                estimator = value
            elif name == "strategy":
                strategy = normalize_strategy(value)  # type: ignore[arg-type]
            elif name.startswith("strategy__"):
                strategy = replace_strategy(strategy, name, value)
            else:
                raise ValueError(f"Unknown parameter {name!r}.")
        self.estimator, self.strategy = estimator, strategy
        for name in ("estimator_", "history_", "profile_"):
            self.__dict__.pop(name, None)
        return self

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        *,
        context: np.ndarray | None = None,
        verbose: int = 0,
    ) -> "EpisodicTrainer":
        strategy = normalize_strategy(self.strategy)
        if strategy.warm_start is not None and y is None:
            raise ValueError("strategy.warm_start requires fit targets y.")
        if not hasattr(self.estimator, "fit"):
            raise TypeError("estimator must provide fit.")
        reward = resolve_reward(strategy.reward)
        warm_start: dict[str, object] | None = None
        if strategy.warm_start is not None:
            # The retained estimator adapter accepts only legacy keys and does
            # not accept explicit ``None`` values for optional fields.
            warm_start = {
                field.name: value
                for field in fields(strategy.warm_start)
                if (value := getattr(strategy.warm_start, field.name)) is not None
            }
            warm_start["y"] = y
            if warm_start.get("preprocessor_lr") is not None:
                warm_start["lsm_lr"] = warm_start.pop("preprocessor_lr")
        self.estimator.fit(
            X,
            y,
            context=context,
            verbose=verbose,
            hisso=True,
            hisso_window=strategy.schedule.episode_length,
            hisso_batch_episodes=strategy.schedule.batch_episodes,
            hisso_updates_per_epoch=strategy.schedule.updates_per_epoch,
            hisso_reward_fn=reward,
            hisso_context_extractor=strategy.context_extractor,
            hisso_primary_transform=strategy.primary_transform,
            hisso_transition_penalty=strategy.transition_penalty,
            hisso_supervised=warm_start,
            noisy=strategy.input_noise_std,
        )
        self.estimator_ = self.estimator
        legacy = getattr(self.estimator, "_hisso_trainer_", None)
        self.history_ = list(getattr(legacy, "history", ()))
        self.profile_ = dict(getattr(legacy, "profile", {}))
        self.estimator._episodic_strategy_ = strategy
        self.estimator._episodic_history_ = self.history_
        self.estimator._episodic_profile_ = self.profile_
        return self

    def _fitted(self) -> object:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("EpisodicTrainer is not fitted.")
        return self.estimator_

    def predict(self, X: np.ndarray, *, context: np.ndarray | None = None) -> np.ndarray:
        estimator = self._fitted()
        values = estimator.predict(X, context=context)
        return transform_actions(
            np.asarray(values), normalize_strategy(self.strategy).primary_transform
        )

    def evaluate(self, X: np.ndarray, *, context: np.ndarray | None = None) -> float:
        strategy = normalize_strategy(self.strategy)
        actions = torch.as_tensor(self.predict(X, context=context), dtype=torch.float32)
        if actions.ndim == 1:
            actions = actions[:, None]
        data = torch.as_tensor(np.asarray(X), dtype=actions.dtype)
        if strategy.context_extractor is None:
            reward_context = data.reshape(data.shape[0], -1)
        else:
            reward_context = strategy.context_extractor(data)
            if not isinstance(reward_context, torch.Tensor):
                raise TypeError("strategy.context_extractor must return a torch.Tensor.")
        if reward_context.ndim == 1:
            reward_context = reward_context[:, None]
        reward_context = align_context(actions.unsqueeze(0), reward_context.unsqueeze(0))
        reward = call_reward(
            resolve_reward(strategy.reward),
            actions.unsqueeze(0),
            reward_context,
            strategy.transition_penalty,
        )
        return float(reward.mean().detach().cpu())

    def save(self, path: str | Path) -> None:
        self._fitted().save(str(path))

    @classmethod
    def load(cls, path: str | Path, *, map_location: str = "cpu") -> "EpisodicTrainer":
        from psann import PSANNRegressor

        estimator = PSANNRegressor.load(str(path), map_location=map_location)
        strategy = getattr(estimator, "_episodic_strategy_", None)
        if strategy is None:
            raise ValueError("Checkpoint fitted.episodic metadata is missing.")
        trainer = cls(estimator=estimator, strategy=strategy)
        trainer.estimator_ = estimator
        trainer.history_ = list(getattr(estimator, "_episodic_history_", ()))
        trainer.profile_ = dict(getattr(estimator, "_episodic_profile_", {}))
        return trainer


__all__ = ["EpisodicTrainer"]
