"""The canonical estimator-wrapping episodic trainer."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import numpy as np

from .config import HISSOConfig, normalize_strategy, replace_strategy
from .rewards import resolve_reward
from .runtime import transform_actions, validate_reward_penalty
from .runtime_loop import HISSOTrainer


class EpisodicTrainer:
    def __init__(
        self, *, estimator: object, strategy: HISSOConfig | Mapping[str, object] | str = "hisso"
    ) -> None:
        self.estimator: Any = estimator
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
        unknown = [
            name
            for name in params
            if name not in {"estimator", "strategy"} and not name.startswith("strategy__")
        ]
        if unknown:
            raise ValueError(f"Unknown parameter {unknown[0]!r}.")
        if "strategy" in params and any(name.startswith("strategy__") for name in params):
            raise ValueError("strategy conflicts with strategy__ nested updates.")
        estimator = params.get("estimator", self.estimator)
        strategy = normalize_strategy(params.get("strategy", self.strategy))  # type: ignore[arg-type]
        nested = [(name, value) for name, value in params.items() if name.startswith("strategy__")]
        nested.sort(key=lambda item: item[0].count("__"))
        for name, value in nested:
            strategy = replace_strategy(strategy, name, value)
        self.estimator, self.strategy = estimator, strategy
        invalidate = getattr(self.estimator, "_clear_architecture_runtime", None)
        if callable(invalidate):
            invalidate()
        else:
            for name in (
                "_episodic_strategy_",
                "_episodic_history_",
                "_episodic_profile_",
                "_hisso_trainer_",
                "_hisso_options_",
                "_hisso_cfg_",
                "_hisso_reward_fn_",
                "_hisso_context_extractor_",
            ):
                getattr(self.estimator, "__dict__", {}).pop(name, None)
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
        if np.asarray(X).shape[0] == 0:
            raise ValueError("Canonical episodic training requires non-empty inputs.")
        if strategy.warm_start is not None and y is None:
            raise ValueError("strategy.warm_start requires fit targets y.")
        if not hasattr(self.estimator, "fit"):
            raise TypeError("estimator must provide fit.")
        validate_reward_penalty(resolve_reward(strategy.reward), strategy.transition_penalty)
        # The estimator still owns construction/scaling/preprocessing, but the
        # typed request is consumed by its episodic stage directly. Do not
        # round canonical fields through the deprecated flat keyword adapter.
        setattr(self.estimator, "_episodic_strategy_request_", strategy)
        setattr(self.estimator, "_episodic_targets_request_", y)
        setattr(self.estimator, "_episodic_canonical_call_", True)
        try:
            self.estimator.fit(
                X,
                y,
                context=context,
                verbose=verbose,
                hisso=True,
            )
        finally:
            self.estimator.__dict__.pop("_episodic_canonical_call_", None)
            self.estimator.__dict__.pop("_episodic_strategy_request_", None)
            self.estimator.__dict__.pop("_episodic_targets_request_", None)
        self.estimator_ = self.estimator
        legacy = getattr(self.estimator, "_hisso_trainer_", None)
        self.history_ = list(getattr(legacy, "history", ()))
        self.profile_ = dict(getattr(legacy, "profile", {}))
        self.estimator._episodic_strategy_ = strategy
        self.estimator._episodic_history_ = self.history_
        self.estimator._episodic_profile_ = self.profile_
        return self

    def _fitted(self) -> Any:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("EpisodicTrainer is not fitted.")
        return self.estimator_

    def predict(self, X: np.ndarray, *, context: np.ndarray | None = None) -> np.ndarray:
        estimator = self._fitted()
        if getattr(estimator, "stateful", False) and hasattr(estimator, "predict_sequence"):
            # Episodic inference always starts clean and never commits training
            # state.  The sequence mixin provides that guarantee explicitly.
            values = estimator.predict_sequence(
                X, context=context, reset_state=True, return_sequence=True, update_state=False
            )
        else:
            values = estimator.predict(X, context=context)
        return transform_actions(
            np.asarray(values), normalize_strategy(self.strategy).primary_transform
        )

    def evaluate(self, X: np.ndarray, *, context: np.ndarray | None = None) -> float:
        strategy = normalize_strategy(self.strategy)
        estimator = self._fitted()
        prepared, _, _ = estimator._prepare_inference_inputs(X, context=context)
        runtime = getattr(estimator, "_hisso_trainer_", None)
        if runtime is None:
            runtime = HISSOTrainer(
                estimator.model_,
                cfg=estimator._hisso_cfg_,
                device=estimator._device(),
                lr=0.0,
                reward_fn=resolve_reward(strategy.reward),
                context_extractor=cast(Callable | None, strategy.context_extractor),
                input_noise_std=None,
                stateful=bool(estimator.stateful),
                state_reset=str(estimator.state_reset),
                strict=True,
            )
        return runtime.evaluate_prepared(prepared)

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
