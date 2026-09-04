"""Frozen canonical configuration for differentiable episodic training."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from numbers import Real
from typing import Any, Mapping, cast


def _integer(value: object, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    if value < minimum:
        raise ValueError(f"{path} must be at least {minimum}.")
    return value


def _finite(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a finite real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{path} must be positive.")
    if not positive and result < 0:
        raise ValueError(f"{path} must be non-negative.")
    return result


def _name(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string.")
    return value.strip().lower().replace("_", "-")


@dataclass(frozen=True)
class EpisodeScheduleConfig:
    episode_length: int = 64
    batch_episodes: int = 32
    updates_per_epoch: int = 1
    random_state: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_length",
            _integer(self.episode_length, "strategy.schedule.episode_length"),
        )
        object.__setattr__(
            self,
            "batch_episodes",
            _integer(self.batch_episodes, "strategy.schedule.batch_episodes"),
        )
        object.__setattr__(
            self,
            "updates_per_epoch",
            _integer(self.updates_per_epoch, "strategy.schedule.updates_per_epoch"),
        )
        if self.random_state is not None:
            object.__setattr__(
                self,
                "random_state",
                _integer(self.random_state, "strategy.schedule.random_state", minimum=0),
            )


@dataclass(frozen=True)
class SupervisedWarmStartConfig:
    epochs: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    preprocessor_lr: float | None = None
    weight_decay: float | None = None
    shuffle: bool | None = None

    def __post_init__(self) -> None:
        for field_name in ("epochs", "batch_size"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self, field_name, _integer(value, f"strategy.warm_start.{field_name}")
                )
        for field_name in ("lr", "preprocessor_lr"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _finite(value, f"strategy.warm_start.{field_name}", positive=True),
                )
        if self.weight_decay is not None:
            object.__setattr__(
                self, "weight_decay", _finite(self.weight_decay, "strategy.warm_start.weight_decay")
            )
        if self.shuffle is not None and not isinstance(self.shuffle, bool):
            raise TypeError("strategy.warm_start.shuffle must be a boolean or None.")


@dataclass(frozen=True)
class HISSOConfig:
    schedule: EpisodeScheduleConfig = EpisodeScheduleConfig()
    primary_transform: str = "identity"
    transition_penalty: float = 0.0
    reward: object = "default"
    context_extractor: object | None = None
    input_noise_std: float | None = None
    warm_start: SupervisedWarmStartConfig | None = None
    gradient_clip: float | None = 1.0
    mixed_precision: bool = False
    amp_dtype: str = "float16"

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, EpisodeScheduleConfig):
            raise TypeError("strategy.schedule must be an EpisodeScheduleConfig.")
        transform = _name(self.primary_transform, "strategy.primary_transform")
        if transform not in {"identity", "softmax", "tanh"}:
            raise ValueError("strategy.primary_transform must be identity, softmax, or tanh.")
        object.__setattr__(self, "primary_transform", transform)
        object.__setattr__(
            self,
            "transition_penalty",
            _finite(self.transition_penalty, "strategy.transition_penalty"),
        )
        if (
            not isinstance(self.reward, str)
            and not callable(self.reward)
            and not hasattr(self.reward, "reward_fn")
        ):
            raise TypeError(
                "strategy.reward must be a registered name, callable, or RewardStrategyBundle."
            )
        if isinstance(self.reward, str) and not self.reward.strip():
            raise ValueError("strategy.reward must be non-empty.")
        if self.context_extractor is not None and not callable(self.context_extractor):
            raise TypeError("strategy.context_extractor must be callable or None.")
        if self.input_noise_std is not None:
            object.__setattr__(
                self, "input_noise_std", _finite(self.input_noise_std, "strategy.input_noise_std")
            )
        if self.warm_start is not None and not isinstance(
            self.warm_start, SupervisedWarmStartConfig
        ):
            raise TypeError("strategy.warm_start must be a SupervisedWarmStartConfig or None.")
        if self.gradient_clip is not None:
            object.__setattr__(
                self,
                "gradient_clip",
                _finite(self.gradient_clip, "strategy.gradient_clip", positive=True),
            )
        if not isinstance(self.mixed_precision, bool):
            raise TypeError("strategy.mixed_precision must be a boolean.")
        dtype = _name(self.amp_dtype, "strategy.amp_dtype")
        if dtype not in {"float16", "bfloat16"}:
            raise ValueError("strategy.amp_dtype must be float16 or bfloat16.")
        object.__setattr__(self, "amp_dtype", dtype)


def _nested(value: object, cls: type[Any], path: str) -> Any:
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    raw = dict(value)
    unknown = set(raw) - {field.name for field in fields(cls)}
    if unknown:
        raise ValueError(f"{path}.{min(unknown)} is unknown.")
    return cls(**cast(Any, raw))


def normalize_strategy(value: HISSOConfig | Mapping[str, object] | str = "hisso") -> HISSOConfig:
    """Normalize the sole supported episodic strategy without mutating mappings."""

    if isinstance(value, HISSOConfig):
        return value
    if isinstance(value, str):
        if _name(value, "strategy") != "hisso":
            raise ValueError("strategy preset must be hisso.")
        return HISSOConfig()
    if not isinstance(value, Mapping):
        raise TypeError("strategy must be HISSOConfig, a tagged mapping, or the hisso preset.")
    raw = dict(value)
    kind = _name(raw.pop("kind", ""), "strategy.kind")
    if kind != "hisso":
        raise ValueError("strategy.kind must be hisso.")
    unknown = set(raw) - {field.name for field in fields(HISSOConfig)}
    if unknown:
        raise ValueError(f"strategy.{min(unknown)} is unknown.")
    if "schedule" in raw:
        raw["schedule"] = _nested(raw["schedule"], EpisodeScheduleConfig, "strategy.schedule")
    if "warm_start" in raw and raw["warm_start"] is not None:
        raw["warm_start"] = _nested(
            raw["warm_start"], SupervisedWarmStartConfig, "strategy.warm_start"
        )
    return HISSOConfig(**cast(Any, raw))


def strategy_to_mapping(value: HISSOConfig) -> dict[str, object]:
    if not isinstance(value, HISSOConfig):
        raise TypeError("strategy must be a HISSOConfig.")
    reward: object = value.reward
    if not isinstance(reward, str):
        reward = {"kind": "callable"}
    context: object = None if value.context_extractor is None else {"kind": "callable"}
    return {
        "kind": "hisso",
        "schedule": {
            field.name: getattr(value.schedule, field.name)
            for field in fields(EpisodeScheduleConfig)
        },
        "primary_transform": value.primary_transform,
        "transition_penalty": value.transition_penalty,
        "reward": reward,
        "context_extractor": context,
        "input_noise_std": value.input_noise_std,
        "warm_start": (
            None
            if value.warm_start is None
            else {
                field.name: getattr(value.warm_start, field.name)
                for field in fields(SupervisedWarmStartConfig)
            }
        ),
        "gradient_clip": value.gradient_clip,
        "mixed_precision": value.mixed_precision,
        "amp_dtype": value.amp_dtype,
    }


def replace_strategy(value: HISSOConfig, path: str, replacement: object) -> HISSOConfig:
    """Transactional deep replacement used by the sklearn-compatible wrapper."""

    parts = path.split("__")
    if parts[0] != "strategy":
        raise ValueError(f"Unknown parameter {path!r}.")
    strategy_fields = {field.name for field in fields(HISSOConfig)}
    schedule_fields = {field.name for field in fields(EpisodeScheduleConfig)}
    warm_start_fields = {field.name for field in fields(SupervisedWarmStartConfig)}
    if len(parts) == 2 and parts[1] in strategy_fields:
        return replace(value, **{parts[1]: replacement})
    if len(parts) == 3 and parts[1] == "schedule" and parts[2] in schedule_fields:
        return replace(value, schedule=replace(value.schedule, **{parts[2]: replacement}))
    if (
        len(parts) == 3
        and parts[1] == "warm_start"
        and value.warm_start is not None
        and parts[2] in warm_start_fields
    ):
        return replace(value, warm_start=replace(value.warm_start, **{parts[2]: replacement}))
    raise ValueError(f"Unknown parameter {path!r}.")


__all__ = [
    "EpisodeScheduleConfig",
    "HISSOConfig",
    "SupervisedWarmStartConfig",
    "normalize_strategy",
    "replace_strategy",
    "strategy_to_mapping",
]
