from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Union

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .training import TrainingLoopConfig, run_training_loop
from .types import ArrayLike, ContextExtractor, NoiseSpec, RewardFn

if TYPE_CHECKING:
    from .sklearn import PSANNRegressor


# ---------------------------------------------------------------------------
# Warm-start configuration
# ---------------------------------------------------------------------------


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
        reward_fn: Optional[RewardFn],
        context_extractor: Optional[ContextExtractor],
        primary_transform: Optional[str],
        transition_penalty: Optional[float],
        trans_cost: Optional[float],
        input_noise: Optional[NoiseSpec],
        supervised: Optional[Mapping[str, Any] | bool],
    ) -> "HISSOOptions":
        episode_length = 64 if window is None else max(1, int(window))

        penalty_raw = transition_penalty if transition_penalty is not None else trans_cost
        penalty = float(penalty_raw) if penalty_raw is not None else 0.0

        transform = (primary_transform or "softmax").lower()
        if transform not in {"identity", "softmax", "tanh"}:
            raise ValueError(
                f"Unsupported HISSO primary transform '{primary_transform}'. "
                "Expected one of {'identity', 'softmax', 'tanh'}."
            )

        noise_std: Optional[float] = None
        if input_noise is not None:
            noise_arr = np.asarray(input_noise, dtype=np.float32)
            if noise_arr.ndim == 0:
                noise_std = float(noise_arr.item())
            else:
                warnings.warn(
                    "HISSO currently supports scalar input noise; ignoring non-scalar noise specification.",
                    RuntimeWarning,
                )

        resolved_reward = reward_fn or _default_reward_fn

        return cls(
            episode_length=episode_length,
            transition_penalty=penalty,
            primary_transform=transform,
            reward_fn=resolved_reward,
            context_extractor=context_extractor,
            input_noise_std=noise_std,
            supervised=supervised,
        )

    def with_updates(self, **changes: Any) -> "HISSOOptions":
        """Return a new ``HISSOOptions`` instance with selected fields replaced."""

        return replace(self, **changes)

    def to_trainer_config(
        self,
        *,
        primary_dim: int,
        random_state: Optional[int],
        episodes_per_batch: int = 32,
    ) -> "HISSOTrainerConfig":
        return HISSOTrainerConfig(
            episode_length=int(self.episode_length),
            episodes_per_batch=int(episodes_per_batch),
            primary_dim=int(primary_dim),
            primary_transform=str(self.primary_transform),
            random_state=random_state,
            transition_penalty=float(self.transition_penalty),
        )


def _coerce_context_output(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert arbitrary context outputs into a detached tensor on ``device``."""

    target_dtype = dtype if dtype is not None else torch.float32

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor.to(device)

    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float32)
        tensor = torch.from_numpy(array)
        return tensor.to(device=device, dtype=target_dtype)

    if isinstance(value, Mapping):
        # Prefer common finance-style keys before falling back to the first convertible value.
        for key in ("price_matrix", "prices", "returns", "context"):
            if key in value:
                try:
                    return _coerce_context_output(value[key], device=device, dtype=target_dtype)
                except TypeError:
                    pass
        for item in value.values():
            try:
                return _coerce_context_output(item, device=device, dtype=target_dtype)
            except TypeError:
                continue
        raise TypeError("context_extractor mapping did not contain tensor-compatible values.")

    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _coerce_context_output(item, device=device, dtype=target_dtype)
            except TypeError:
                continue
        raise TypeError("context_extractor sequence did not contain tensor-compatible values.")

    raise TypeError(f"Unsupported context_extractor output type '{type(value).__name__}'.")


def _call_context_extractor(
    extractor: Optional[ContextExtractor],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Invoke ``extractor`` with best-effort dtype/device handling for HISSO training."""

    if extractor is None:
        return inputs.detach()

    try:
        context = extractor(inputs)
    except TypeError as first_exc:
        if not isinstance(inputs, torch.Tensor):
            raise first_exc
        inputs_np = inputs.detach().cpu().numpy()
        try:
            context = extractor(inputs_np)
        except Exception as second_exc:  # pragma: no cover - defensive fallback
            raise first_exc from second_exc

    if isinstance(context, tuple):
        context = context[0]
    if context is None:
        raise TypeError("context_extractor returned None; expected tensor-like output.")

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("HISSO context extraction requires tensor inputs.")

    device = inputs.device
    dtype = inputs.dtype if inputs.dtype is not None else torch.float32
    return _coerce_context_output(context, device=device, dtype=dtype)


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


def run_hisso_supervised_warmstart(
    estimator: "PSANNRegressor",
    X_flat: np.ndarray,
    *,
    primary_dim: int,
    config: Optional[HISSOWarmStartConfig],
    lsm_module: Optional[torch.nn.Module],
) -> None:
    """Run a supervised warm start against primary targets before HISSO."""

    if config is None:
        return

    y_vec = np.asarray(config.targets, dtype=np.float32)
    if y_vec.ndim == 1:
        y_vec = y_vec.reshape(-1, 1)
    if y_vec.ndim != 2:
        raise ValueError("hisso_supervised['y'] must be 2D with shape (N, primary_dim).")
    if y_vec.shape[0] != X_flat.shape[0]:
        raise ValueError("hisso_supervised['y'] length must match X.")
    if y_vec.shape[1] != int(primary_dim):
        raise ValueError("hisso_supervised['y'] column count must equal primary_dim.")

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_flat.astype(np.float32)),
        torch.from_numpy(y_vec.astype(np.float32)),
    )

    shuffle = not (estimator.stateful and estimator.state_reset in ("epoch", "none"))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.batch_size or estimator.batch_size),
        shuffle=shuffle,
        num_workers=int(estimator.num_workers),
    )

    device = estimator._device()
    estimator.model_.to(device)

    optimizer = estimator._build_optimizer(estimator.model_)
    if config.lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = float(config.lr)
    if config.weight_decay is not None:
        for group in optimizer.param_groups:
            group["weight_decay"] = float(config.weight_decay)
    if lsm_module is not None and config.lsm_lr is not None:
        for group in optimizer.param_groups:
            if any(param in group["params"] for param in lsm_module.parameters()):
                group["lr"] = float(config.lsm_lr)

    loop_cfg = TrainingLoopConfig(
        epochs=int(config.epochs or estimator.epochs),
        patience=1,
        early_stopping=False,
        stateful=bool(estimator.stateful),
        state_reset=str(estimator.state_reset),
        verbose=int(config.verbose),
        lr_max=None,
        lr_min=None,
    )

    loss_fn = estimator._make_loss()
    run_training_loop(
        estimator.model_,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dataloader,
        device=device,
        cfg=loop_cfg,
    )
    estimator.model_.eval()


# ---------------------------------------------------------------------------
# Episodic HISSO trainer (primary-output only)
# ---------------------------------------------------------------------------


@dataclass
class HISSOTrainerConfig:
    """Lean HISSO trainer configuration for primary-output optimisation."""

    episode_length: int = 64
    episodes_per_batch: int = 32
    primary_dim: int = 1
    primary_transform: str = "identity"
    random_state: Optional[int] = None
    transition_penalty: float = 0.0

    def resolved_transition_penalty(self) -> float:
        return float(self.transition_penalty or 0.0)


class HISSOTrainer:
    """Simple episodic trainer that optimises the primary head via rewards."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        cfg: HISSOTrainerConfig,
        device: torch.device,
        lr: float,
        reward_fn: Optional[RewardFn],
        context_extractor: Optional[ContextExtractor],
        input_noise_std: Optional[float],
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.reward_fn = reward_fn or _default_reward_fn
        self.context_extractor = context_extractor
        self.input_noise_std = float(input_noise_std) if input_noise_std is not None else None
        self.primary_dim = int(cfg.primary_dim)
        self.history: list[dict[str, Any]] = []
        self.profile: dict[str, Any] = {
            "device": str(device),
            "epochs": 0,
            "total_time_s": 0.0,
            "episode_length": int(cfg.episode_length),
            "batch_episodes": int(cfg.episodes_per_batch),
        }
        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=float(lr),
        )
        self._rng = np.random.default_rng(cfg.random_state)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        *,
        epochs: int,
        verbose: int,
        lr_max: Optional[float],
        lr_min: Optional[float],
    ) -> None:
        """Optimise the underlying model against sampled HISSO episodes."""

        data = np.asarray(X_train, dtype=np.float32)
        if data.size == 0:
            raise ValueError("HISSO training requires non-empty inputs.")

        x_tensor = torch.from_numpy(data).to(self.device)
        episode_len = max(1, int(self.cfg.episode_length))
        total_steps = int(x_tensor.shape[0])
        episode_len = min(episode_len, total_steps)

        self.model.to(self.device)
        self.history.clear()

        for epoch_idx in range(max(1, int(epochs))):
            self.model.train()
            epoch_start = time.perf_counter()

            total_reward = 0.0
            episode_count = 0

            for start in self._episode_starts(total_steps, episode_len):
                end = start + episode_len
                episode = x_tensor[start:end]

                inputs = episode
                if self.input_noise_std:
                    noise = torch.randn_like(inputs) * float(self.input_noise_std)
                    inputs = inputs + noise

                context = self._extract_context(inputs)
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                if outputs.ndim > 2:
                    outputs = outputs.view(outputs.shape[0], -1)

                primary = self._apply_primary_transform(outputs)
                reward_tensor = self._coerce_reward(primary, context)
                loss = -reward_tensor.mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_reward += float(reward_tensor.detach().mean().item())
                episode_count += 1

            duration = time.perf_counter() - epoch_start

            avg_reward = total_reward / max(episode_count, 1)
            self.history.append(
                {
                    "epoch": epoch_idx + 1,
                    "reward": avg_reward,
                    "episodes": episode_count,
                }
            )
            self.profile["total_time_s"] += duration

        self.profile["epochs"] = len(self.history)
        self.model.eval()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _episode_starts(self, total_steps: int, episode_length: int) -> Iterable[int]:
        if total_steps <= episode_length:
            return [0]
        max_start = total_steps - episode_length
        count = max(1, int(self.cfg.episodes_per_batch))
        starts = self._rng.integers(0, max_start + 1, size=count, endpoint=False)
        starts_arr = np.atleast_1d(starts)
        return [int(val) for val in starts_arr]

    def _extract_context(self, inputs: torch.Tensor) -> torch.Tensor:
        return _call_context_extractor(self.context_extractor, inputs)

    def _apply_primary_transform(self, primary: torch.Tensor) -> torch.Tensor:
        transform = (self.cfg.primary_transform or "identity").lower()
        if transform == "identity":
            return primary
        if transform == "softmax":
            return torch.softmax(primary, dim=-1)
        if transform == "tanh":
            return torch.tanh(primary)
        raise ValueError(f"Unsupported primary_transform '{self.cfg.primary_transform}'.")

    def _coerce_reward(self, primary: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        actions = primary
        ctx = context
        if actions.ndim == 1:
            actions = actions.unsqueeze(0).unsqueeze(-1)
        elif actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0).unsqueeze(-1)
        elif ctx.ndim == 2:
            ctx = ctx.unsqueeze(0)
        ctx = ctx.to(device=actions.device, dtype=actions.dtype)
        if actions.shape != ctx.shape:
            raise ValueError(
                f"HISSO reward expects actions/context with matching shape; got {tuple(actions.shape)} vs {tuple(ctx.shape)}."
            )
        reward = self.reward_fn(actions, ctx)
        if isinstance(reward, torch.Tensor):
            reward_tensor = reward
        else:
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=primary.device)
        if reward_tensor.ndim == 0:
            reward_tensor = reward_tensor.reshape(1)
        return reward_tensor


def _default_reward_fn(primary: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
    """Fallback reward that penalises large activations."""

    return -primary.pow(2).mean(dim=-1)


def run_hisso_training(
    estimator: "PSANNRegressor",
    X_train_arr: np.ndarray,
    *,
    trainer_cfg: HISSOTrainerConfig,
    lr: float,
    device: torch.device,
    reward_fn: Optional[RewardFn] = None,
    context_extractor: Optional[ContextExtractor] = None,
    lr_max: Optional[float] = None,
    lr_min: Optional[float] = None,
    input_noise_std: Optional[NoiseSpec] = None,
    verbose: int = 0,
) -> HISSOTrainer:
    """Instantiate the lightweight HISSO trainer and execute one optimisation run."""

    device_t = device if isinstance(device, torch.device) else torch.device(device)
    context_fn = context_extractor
    trainer = HISSOTrainer(
        estimator.model_,
        cfg=trainer_cfg,
        device=device_t,
        lr=float(lr),
        reward_fn=reward_fn,
        context_extractor=context_fn,
        input_noise_std=input_noise_std,
    )
    trainer.train(
        X_train_arr,
        epochs=int(estimator.epochs),
        verbose=int(verbose),
        lr_max=lr_max,
        lr_min=lr_min,
    )
    estimator._hisso_reward_fn_ = trainer.reward_fn
    estimator._hisso_context_extractor_ = context_fn
    return trainer


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _resolve_hisso_config(
    estimator: "PSANNRegressor",
    override: Optional[HISSOTrainerConfig],
) -> Optional[HISSOTrainerConfig]:
    """Prefer the explicit config, otherwise fall back to the fitted trainer state."""

    if override is not None:
        return override
    return getattr(estimator, "_hisso_cfg_", None)


def _resolve_primary_transform(
    cfg: Optional[HISSOTrainerConfig],
    options: Optional[HISSOOptions],
) -> Optional[str]:
    if cfg is not None and cfg.primary_transform:
        return cfg.primary_transform
    if options is not None:
        return options.primary_transform
    return None


def _apply_primary_transform_numpy(values: np.ndarray, transform: Optional[str]) -> np.ndarray:
    """Apply the configured primary transform in numpy space."""

    arr = np.asarray(values, dtype=np.float32)
    squeeze = False
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
        squeeze = True

    if transform is None:
        result = arr
    else:
        transform_lower = transform.lower()
        if transform_lower == "identity":
            result = arr
        elif transform_lower == "softmax":
            shifted = arr - arr.max(axis=1, keepdims=True)
            exp = np.exp(shifted)
            totals = exp.sum(axis=1, keepdims=True)
            np.clip(totals, a_min=np.finfo(np.float32).tiny, a_max=None, out=totals)
            result = exp / totals
        elif transform_lower == "tanh":
            result = np.tanh(arr)
        else:
            raise ValueError(f"Unsupported primary transform '{transform}'.")

    return result.squeeze(1) if squeeze else result


def hisso_infer_series(
    estimator: "PSANNRegressor",
    X_obs: np.ndarray,
    *,
    trainer_cfg: Optional[HISSOTrainerConfig] = None,
) -> np.ndarray:
    """Run the fitted estimator on a sequence for inference."""

    cfg = _resolve_hisso_config(estimator, trainer_cfg)
    if getattr(estimator, "stateful", False):
        preds = estimator.predict_sequence(X_obs, reset_state=True, return_sequence=True)
    else:
        preds = estimator.predict(X_obs)
    options = getattr(estimator, "_hisso_options_", None)
    transform = _resolve_primary_transform(cfg, options)
    return _apply_primary_transform_numpy(preds, transform)


def hisso_evaluate_reward(
    estimator: "PSANNRegressor",
    X_obs: np.ndarray,
    *,
    trainer_cfg: Optional[HISSOTrainerConfig] = None,
) -> float:
    """Evaluate the configured reward function over observed inputs."""

    options = getattr(estimator, "_hisso_options_", None)
    reward_fn = None
    context_extractor = None
    if options is not None:
        reward_fn = options.reward_fn
        context_extractor = options.context_extractor
    else:
        reward_fn = getattr(estimator, "_hisso_reward_fn_", None)
        context_extractor = getattr(estimator, "_hisso_context_extractor_", None)

    if reward_fn is None:
        return 0.0

    device = estimator._device()
    X_np = np.asarray(X_obs, dtype=np.float32)
    inputs_t = torch.from_numpy(X_np).to(device)

    context_t = _call_context_extractor(context_extractor, inputs_t)

    cfg = _resolve_hisso_config(estimator, trainer_cfg)
    preds = estimator.predict(X_obs)
    transform = _resolve_primary_transform(cfg, options)
    primary_np = _apply_primary_transform_numpy(preds, transform)
    primary_t = torch.from_numpy(primary_np).to(device)

    reward = reward_fn(primary_t, context_t)
    if isinstance(reward, torch.Tensor):
        reward_val = float(reward.mean().detach().cpu().item())
    else:
        reward_val = float(reward)
    return reward_val


def ensure_hisso_trainer_config(
    value: Union[HISSOTrainerConfig, Mapping[str, Any], Any],
) -> HISSOTrainerConfig:
    """Coerce persisted metadata into a HISSOTrainerConfig instance."""

    if isinstance(value, HISSOTrainerConfig):
        return value
    if isinstance(value, Mapping):
        return HISSOTrainerConfig(
            episode_length=int(value.get("episode_length", 64)),
            episodes_per_batch=int(
                value.get("episodes_per_batch", value.get("batch_episodes", 32))
            ),
            primary_dim=int(value.get("primary_dim", 1)),
            primary_transform=str(value.get("primary_transform", "identity")),
            random_state=value.get("random_state", None),
            transition_penalty=float(value.get("transition_penalty", 0.0)),
        )
    raise ValueError("Unsupported HISSO trainer configuration format.")
