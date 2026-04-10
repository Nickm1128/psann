from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Union

import numpy as np

from ..types import ArrayLike, ContextExtractor, NoiseSpec, RewardFn
from .reward import _default_reward_fn


@dataclass
class HISSOWarmStartConfig:
    """Configuration for an optional supervised warm start prior to HISSO."""

    targets: ArrayLike
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    lsm_lr: Optional[float] = None
    shuffle: bool = True
    verbose: int = 0


@dataclass(frozen=True)
class HISSOOptions:
    """Canonical configuration for HISSO reward/context behaviour."""

    episode_length: int
    batch_episodes: int
    updates_per_epoch: Optional[int]
    transition_penalty: float
    primary_transform: str
    reward_fn: RewardFn
    context_extractor: Optional[ContextExtractor]
    input_noise_std: Optional[float]
    supervised: Optional[Mapping[str, Any] | bool]

    @classmethod
    def from_kwargs(
        cls,
        *,
        window: Optional[int],
        batch_episodes: Optional[int] = None,
        updates_per_epoch: Optional[int] = None,
        reward_fn: Optional[RewardFn],
        context_extractor: Optional[ContextExtractor],
        primary_transform: Optional[str],
        transition_penalty: Optional[float],
        trans_cost: Optional[float],
        input_noise: Optional[NoiseSpec],
        supervised: Optional[Mapping[str, Any] | bool],
    ) -> "HISSOOptions":
        episode_length = 64 if window is None else max(1, int(window))
        resolved_batch_episodes = max(1, int(batch_episodes)) if batch_episodes is not None else 32
        resolved_updates_per_epoch = (
            max(1, int(updates_per_epoch)) if updates_per_epoch is not None else None
        )

        penalty_raw = transition_penalty if transition_penalty is not None else trans_cost
        penalty = float(penalty_raw) if penalty_raw is not None else 0.0

        transform = (primary_transform or "softmax").lower()
        if transform not in {"identity", "softmax", "tanh"}:
            raise ValueError(
                f"Unsupported HISSO primary transform '{primary_transform}'. "
                "Expected one of {'identity', 'softmax', 'tanh'}."
            )

        noise_std = None
        if input_noise is not None:
            noise_arr = np.asarray(input_noise, dtype=np.float32)
            if noise_arr.ndim == 0:
                noise_std = float(noise_arr.item())
            else:
                warnings.warn(
                    "HISSO currently supports scalar input noise; ignoring non-scalar noise specification.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return cls(
            episode_length=episode_length,
            batch_episodes=resolved_batch_episodes,
            updates_per_epoch=resolved_updates_per_epoch,
            transition_penalty=penalty,
            primary_transform=transform,
            reward_fn=reward_fn or _default_reward_fn,
            context_extractor=context_extractor,
            input_noise_std=noise_std,
            supervised=supervised,
        )

    def with_updates(self, **changes: Any) -> "HISSOOptions":
        return replace(self, **changes)

    def to_trainer_config(
        self,
        *,
        primary_dim: int,
        random_state: Optional[int],
        episode_batch_size: Optional[int] = None,
        updates_per_epoch: Optional[int] = None,
    ) -> "HISSOTrainerConfig":
        resolved_episode_batch = (
            int(episode_batch_size) if episode_batch_size is not None else int(self.batch_episodes)
        )
        resolved_updates_per_epoch = (
            int(updates_per_epoch) if updates_per_epoch is not None else self.updates_per_epoch
        )
        explicit_vectorized = resolved_updates_per_epoch is not None
        if explicit_vectorized:
            total_episodes = max(1, resolved_episode_batch * int(resolved_updates_per_epoch))
            cfg_episode_batch_size = max(1, resolved_episode_batch)
        else:
            total_episodes = max(1, resolved_episode_batch)
            cfg_episode_batch_size = None
        return HISSOTrainerConfig(
            episode_length=int(self.episode_length),
            episodes_per_batch=int(total_episodes),
            episode_batch_size=cfg_episode_batch_size,
            updates_per_epoch=(
                int(resolved_updates_per_epoch) if resolved_updates_per_epoch is not None else None
            ),
            primary_dim=int(primary_dim),
            primary_transform=str(self.primary_transform),
            random_state=random_state,
            transition_penalty=float(self.transition_penalty),
        )


@dataclass
class HISSOTrainerConfig:
    """Lean HISSO trainer configuration for primary-output optimisation."""

    episode_length: int = 64
    episodes_per_batch: int = 32
    episode_batch_size: Optional[int] = None
    updates_per_epoch: Optional[int] = None
    primary_dim: int = 1
    primary_transform: str = "identity"
    random_state: Optional[int] = None
    transition_penalty: float = 0.0

    def resolved_transition_penalty(self) -> float:
        return float(self.transition_penalty or 0.0)

    def resolved_episode_batch_size(self) -> int:
        if self.episode_batch_size is None:
            return 1
        return max(1, int(self.episode_batch_size))

    def resolved_updates_per_epoch(self) -> int:
        if self.updates_per_epoch is not None:
            return max(1, int(self.updates_per_epoch))
        total_episodes = max(1, int(self.episodes_per_batch))
        batch_size = self.resolved_episode_batch_size()
        return max(1, int(np.ceil(total_episodes / batch_size)))

    def episodes_in_update(self, update_idx: int) -> int:
        batch_size = self.resolved_episode_batch_size()
        if self.updates_per_epoch is not None and self.episode_batch_size is not None:
            return batch_size
        total_episodes = max(1, int(self.episodes_per_batch))
        consumed = int(update_idx) * batch_size
        remaining = total_episodes - consumed
        if remaining <= 0:
            return 0
        return min(batch_size, remaining)


def coerce_warmstart_config(
    hisso_supervised: Optional[Mapping[str, Any] | bool],
    y_default: Optional[np.ndarray],
) -> Optional[HISSOWarmStartConfig]:
    """Normalise the ``hisso_supervised`` argument passed to ``fit``."""

    if not hisso_supervised:
        return None
    if isinstance(hisso_supervised, bool):
        cfg_map: dict[str, Any] = {}
    elif isinstance(hisso_supervised, Mapping):
        cfg_map = dict(hisso_supervised)
    else:  # pragma: no cover - defensive
        raise ValueError("hisso_supervised must be a dict of options or a boolean.")

    targets = cfg_map.pop("y", None)
    if targets is None:
        targets = cfg_map.pop("targets", None)
    if targets is None:
        if y_default is not None:
            targets = y_default
        else:
            raise ValueError(
                "hisso_supervised requires targets either via the mapping or the fit(...) y argument."
            )

    epochs = cfg_map.pop("epochs", None)
    batch_size = cfg_map.pop("batch_size", None)
    lr = cfg_map.pop("lr", None)
    weight_decay = cfg_map.pop("weight_decay", None)
    lsm_lr = cfg_map.pop("lsm_lr", None)
    shuffle = bool(cfg_map.pop("shuffle", True))
    verbose = int(cfg_map.pop("verbose", 0))

    if cfg_map:
        unknown = ", ".join(sorted(cfg_map.keys()))
        raise ValueError(f"Unsupported hisso_supervised options: {unknown}")

    return HISSOWarmStartConfig(
        targets=targets,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        lsm_lr=lsm_lr,
        shuffle=shuffle,
        verbose=verbose,
    )


def ensure_hisso_trainer_config(
    value: Union[HISSOTrainerConfig, Mapping[str, Any], Any],
) -> HISSOTrainerConfig:
    """Coerce persisted metadata into a HISSOTrainerConfig instance."""

    if isinstance(value, HISSOTrainerConfig):
        return value
    if isinstance(value, Mapping):
        episode_batch_size_raw = value.get("episode_batch_size", None)
        updates_per_epoch_raw = value.get("updates_per_epoch", None)
        return HISSOTrainerConfig(
            episode_length=int(value.get("episode_length", 64)),
            episodes_per_batch=int(
                value.get("episodes_per_batch", value.get("batch_episodes", 32))
            ),
            episode_batch_size=(
                int(episode_batch_size_raw) if episode_batch_size_raw is not None else None
            ),
            updates_per_epoch=(
                int(updates_per_epoch_raw) if updates_per_epoch_raw is not None else None
            ),
            primary_dim=int(value.get("primary_dim", 1)),
            primary_transform=str(value.get("primary_transform", "identity")),
            random_state=value.get("random_state", None),
            transition_penalty=float(value.get("transition_penalty", 0.0)),
        )
    raise TypeError(
        "Unsupported HISSO trainer configuration format; " f"received {type(value).__name__}."
    )
