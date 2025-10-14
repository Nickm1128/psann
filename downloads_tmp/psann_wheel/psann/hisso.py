from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .training import TrainingLoopConfig, run_training_loop
from .extras import SupervisedExtrasConfig


@dataclass
class HISSOWarmStartConfig:
    """Configuration for optional supervised warm-start before HISSO."""

    targets: Any
    extras_targets: Optional[Any] = None
    extras_loss_weight: Optional[float] = None
    extras_loss_mode: Optional[str] = None
    extras_loss_cycle: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    lsm_lr: Optional[float] = None
    shuffle: bool = True
    verbose: int = 0


@dataclass
class HISSOTrainerConfig:
    """Canonical HISSO trainer configuration used by sklearn wrappers."""

    episode_length: int = 64
    episodes_per_batch: int = 32
    primary_dim: int = 1
    extras_dim: int = 0
    primary_transform: str = "softmax"
    extras_transform: str = "tanh"
    random_state: Optional[int] = None
    transition_cost: float = 0.0
    extras_supervision_weight: float = 0.0
    extras_supervision_mode: str = "joint"
    extras_supervision_cycle: int = 2

    def to_predictive_extras(self):
        from .augmented import PredictiveExtrasConfig

        return PredictiveExtrasConfig(
            episode_length=int(self.episode_length),
            batch_episodes=int(self.episodes_per_batch),
            primary_dim=int(self.primary_dim),
            extras_dim=int(self.extras_dim),
            primary_transform=str(self.primary_transform),
            extras_transform=str(self.extras_transform),
            random_state=self.random_state,
            trans_cost=float(self.transition_cost),
            extras_supervision_weight=float(self.extras_supervision_weight),
            extras_supervision_mode=str(self.extras_supervision_mode),
            extras_supervision_cycle=int(self.extras_supervision_cycle),
        )

    @classmethod
    def from_predictive_extras(cls, cfg: Any) -> "HISSOTrainerConfig":
        return cls(
            episode_length=int(cfg.episode_length),
            episodes_per_batch=int(getattr(cfg, "batch_episodes", getattr(cfg, "episodes_per_batch", 32))),
            primary_dim=int(cfg.primary_dim),
            extras_dim=int(cfg.extras_dim),
            primary_transform=str(cfg.primary_transform),
            extras_transform=str(cfg.extras_transform),
            random_state=getattr(cfg, "random_state", None),
            transition_cost=float(getattr(cfg, "trans_cost", getattr(cfg, "transition_cost", 0.0))),
            extras_supervision_weight=float(getattr(cfg, "extras_supervision_weight", 0.0)),
            extras_supervision_mode=str(getattr(cfg, "extras_supervision_mode", "joint")),
            extras_supervision_cycle=int(getattr(cfg, "extras_supervision_cycle", 2)),
        )


def coerce_warmstart_config(
    hisso_supervised: Optional[Dict[str, Any] | bool],
    y_default: Optional[np.ndarray],
) -> Optional[HISSOWarmStartConfig]:
    if not hisso_supervised:
        return None
    if isinstance(hisso_supervised, bool):
        cfg_map: Dict[str, Any] = {}
    elif isinstance(hisso_supervised, dict):
        cfg_map = dict(hisso_supervised)
    else:
        raise ValueError("hisso_supervised must be a dict of options or a boolean")

    targets = cfg_map.pop("y", None)
    if targets is None:
        targets = cfg_map.pop("targets", None)
    if targets is None:
        if y_default is not None:
            targets = y_default
        else:
            raise ValueError("hisso_supervised requires 'y' either in the dict or via the y argument to fit()")

    extras_targets = cfg_map.pop("extras_targets", None)
    extras_weight = cfg_map.pop("extras_loss_weight", cfg_map.pop("extras_weight", None))
    extras_mode = cfg_map.pop("extras_loss_mode", None)
    extras_cycle = cfg_map.pop("extras_loss_cycle", None)
    epochs = cfg_map.pop("epochs", None)
    batch_size = cfg_map.pop("batch_size", None)
    lr = cfg_map.pop("lr", None)
    weight_decay = cfg_map.pop("weight_decay", None)
    lsm_lr = cfg_map.pop("lsm_lr", None)
    shuffle = bool(cfg_map.pop("shuffle", True))
    verbose = int(cfg_map.pop("verbose", 0))

    return HISSOWarmStartConfig(
        targets=targets,
        extras_targets=extras_targets,
        extras_loss_weight=extras_weight,
        extras_loss_mode=extras_mode,
        extras_loss_cycle=extras_cycle,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        lsm_lr=lsm_lr,
        shuffle=shuffle,
        verbose=verbose,
    )


def run_hisso_supervised_warmstart(
    estimator: Any,
    X_flat: np.ndarray,
    *,
    primary_dim: int,
    extras_dim: int,
    config: Optional[HISSOWarmStartConfig],
    lsm_module: Optional[torch.nn.Module],
) -> None:
    if config is None:
        return

    y_vec = np.asarray(config.targets, dtype=np.float32)
    if y_vec.ndim == 1:
        y_vec = y_vec.reshape(-1, 1)
    if y_vec.ndim != 2:
        raise ValueError("hisso_supervised['y'] must be 2D with shape (N, primary_dim)")
    if y_vec.shape[0] != X_flat.shape[0]:
        raise ValueError("hisso_supervised['y'] must have the same number of samples as X")

    extras_targets = None if config.extras_targets is None else np.asarray(config.extras_targets, dtype=np.float32)
    if extras_targets is not None and extras_targets.ndim == 1:
        extras_targets = extras_targets.reshape(-1, 1)
    if extras_targets is not None:
        if extras_targets.shape[0] != X_flat.shape[0]:
            raise ValueError("hisso_supervised['extras_targets'] must align with X in the first dimension")
        if extras_dim <= 0:
            raise ValueError("extras_targets provided but estimator was initialised with extras=0")
        if extras_targets.shape[1] != int(extras_dim):
            raise ValueError("hisso_supervised['extras_targets'] shape mismatch with extras_dim")
    extras_info: Optional[Dict[str, Any]] = None
    primary_expected = int(primary_dim)

    extras_supervision_requested = (
        extras_targets is not None
        or config.extras_loss_weight is not None
        or config.extras_loss_mode is not None
        or config.extras_loss_cycle is not None
    )
    if extras_supervision_requested:
        extras_info = estimator._prepare_supervised_extras_targets(
            y_vec,
            extras_dim=extras_dim,
            extras_targets=extras_targets,
            extras_loss_weight=config.extras_loss_weight,
            extras_loss_mode=config.extras_loss_mode,
            extras_loss_cycle=config.extras_loss_cycle,
        )
    if extras_info is None and extras_dim > 0 and y_vec.shape[1] == primary_expected + extras_dim:
        primary_y = y_vec[:, :primary_expected]
        extras_arr = y_vec[:, primary_expected:]
        weight = float(config.extras_loss_weight if config.extras_loss_weight is not None else 1.0)
        mode = (config.extras_loss_mode or ("alternate" if weight > 0.0 else "joint")).strip().lower()
        if mode not in {"joint", "alternate"}:
            raise ValueError("extras_loss_mode must be 'joint' or 'alternate'")
        cycle = max(1, int(config.extras_loss_cycle) if config.extras_loss_cycle is not None else 2)
        targets_full = np.concatenate(
            [primary_y.astype(np.float32, copy=False), extras_arr.astype(np.float32, copy=False)],
            axis=1,
        )
        extras_info = {
            "targets": targets_full,
            "primary_dim": int(primary_y.shape[1]),
            "extras_dim": int(extras_dim),
            "weight": float(weight),
            "mode": mode,
            "cycle": int(cycle),
        }

    if extras_info is not None:
        feature_dim = X_flat.shape[1]
        append_flag = feature_dim == int(extras_info["primary_dim"]) + int(extras_info["extras_dim"])
        meta = SupervisedExtrasConfig(
            primary_dim=int(extras_info["primary_dim"]),
            extras_dim=int(extras_info["extras_dim"]),
            feature_dim=int(feature_dim),
            append_to_inputs=bool(append_flag),
            weight=float(extras_info["weight"]),
            mode=str(extras_info["mode"]),
            cycle=int(extras_info["cycle"]),
        )
        estimator._supervised_extras_meta_ = meta
        targets_np = np.asarray(extras_info["targets"], dtype=np.float32)
        primary_dim_use = int(meta.primary_dim)
        extras_dim_use = int(meta.extras_dim)
    else:
        estimator._supervised_extras_meta_ = None
        targets_np = y_vec.astype(np.float32, copy=False)
        primary_dim_use = int(primary_expected) if extras_dim == 0 else min(primary_expected, targets_np.shape[1])
        extras_dim_use = 0 if extras_dim <= 0 else max(0, targets_np.shape[1] - primary_dim_use)
    epochs = int(config.epochs) if config.epochs is not None else max(1, max(1, estimator.epochs // 5))
    batch_size = int(config.batch_size) if config.batch_size is not None else (estimator.batch_size if estimator.batch_size > 0 else 128)
    warm_lr = config.lr
    warm_weight_decay = config.weight_decay
    warm_lsm_lr = config.lsm_lr
    warm_shuffle = bool(config.shuffle)
    warm_verbose = int(config.verbose)

    inputs_np = np.asarray(X_flat, dtype=np.float32)
    ds = TensorDataset(
        torch.from_numpy(inputs_np),
        torch.from_numpy(targets_np.astype(np.float32, copy=False)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=warm_shuffle, num_workers=estimator.num_workers)

    if estimator.lsm_train and getattr(estimator.model_, "preproc", None) is not None and lsm_module is not None:
        params = [
            {"params": estimator.model_.core.parameters(), "lr": float(estimator.lr)},
            {
                "params": estimator.model_.preproc.parameters(),
                "lr": float(estimator.lsm_lr) if estimator.lsm_lr is not None else float(estimator.lr),
            },
        ]
        if warm_lr is not None:
            params[0]["lr"] = float(warm_lr)
            params[1]["lr"] = float(warm_lsm_lr if warm_lsm_lr is not None else warm_lr)
        elif warm_lsm_lr is not None:
            params[1]["lr"] = float(warm_lsm_lr)
        weight_decay = float(warm_weight_decay) if warm_weight_decay is not None else estimator.weight_decay
        opt_name = estimator.optimizer.lower()
        if opt_name == "adamw":
            opt = torch.optim.AdamW(params, weight_decay=weight_decay)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(params, momentum=0.9, weight_decay=weight_decay)
        else:
            opt = torch.optim.Adam(params, weight_decay=weight_decay)
    else:
        opt = estimator._make_optimizer(estimator.model_)
        if warm_lr is not None:
            for group in opt.param_groups:
                group["lr"] = float(warm_lr)
        if warm_weight_decay is not None:
            for group in opt.param_groups:
                group["weight_decay"] = float(warm_weight_decay)

    loss_fn = estimator._make_loss()
    if extras_info is not None:
        loss_fn = estimator._build_extras_loss(
            loss_fn,
            primary_dim=primary_dim_use,
            extras_dim=extras_dim_use,
            weight=float(extras_info["weight"]),
            mode=str(extras_info["mode"]),
            cycle=int(extras_info["cycle"]),
        )

    device = estimator._device()
    cfg_loop = TrainingLoopConfig(
        epochs=int(max(1, epochs)),
        patience=1,
        early_stopping=False,
        stateful=bool(estimator.stateful),
        state_reset=str(estimator.state_reset),
        verbose=warm_verbose,
        lr_max=None,
        lr_min=None,
    )

    run_training_loop(
        estimator.model_,
        optimizer=opt,
        loss_fn=loss_fn,
        train_loader=dl,
        device=device,
        cfg=cfg_loop,
    )
    estimator.model_.eval()
    estimator._supervised_extras_meta_ = None


def run_hisso_training(
    estimator: Any,
    X_train_arr: np.ndarray,
    *,
    trainer_cfg: HISSOTrainerConfig,
    lr: float,
    device: torch.device,
    reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    extras_cache: Optional[Union[np.ndarray, torch.Tensor]] = None,
    lr_max: Optional[float] = None,
    lr_min: Optional[float] = None,
    input_noise_std: Optional[float] = None,
    verbose: int = 0,
    trainer: Optional[Any] = None,
) -> Any:
    try:
        from .augmented import PredictiveExtrasTrainer
        from .episodes import portfolio_log_return_reward
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Predictive extras components not available") from exc

    predictive_cfg = trainer_cfg.to_predictive_extras()
    extras_mode = trainer_cfg.extras_supervision_mode.lower().strip()
    if extras_mode not in {"joint", "alternate"}:
        raise ValueError("hisso_extras_mode must be 'joint' or 'alternate'")

    default_ctx = context_extractor
    if default_ctx is None and getattr(estimator, "_scaler_kind_", None) is not None:

        def _ctx_inv(X_ep: torch.Tensor) -> torch.Tensor:
            return estimator._scaler_inverse_tensor(X_ep)

        default_ctx = _ctx_inv

    target_device = device if isinstance(device, torch.device) else torch.device(device)
    resolved_reward = (
        reward_fn
        if reward_fn is not None
        else (lambda alloc, ctx: portfolio_log_return_reward(alloc, ctx, trans_cost=trainer_cfg.transition_cost))
    )
    resolved_noise = float(input_noise_std) if input_noise_std is not None else None

    reuse_trainer = (
        isinstance(trainer, PredictiveExtrasTrainer)
        and trainer.model is estimator.model_
        and trainer.cfg.episode_length == predictive_cfg.episode_length
        and trainer.primary_dim == int(predictive_cfg.primary_dim)
        and trainer.extras_dim == int(predictive_cfg.extras_dim)
        and trainer.device == target_device
    )

    episode_len = int(predictive_cfg.episode_length)

    def _cache_if_ready(cache):
        if cache is None:
            return None
        length = None
        if hasattr(cache, "shape"):
            shape0 = getattr(cache, "shape")[0]
            try:
                length = int(shape0)
            except Exception:
                length = None
        if length is None:
            try:
                length = int(len(cache))
            except Exception:
                length = None
        if length is None:
            return None
        if length < episode_len + 1:
            return None
        return cache

    cache_ready = _cache_if_ready(extras_cache)

    if reuse_trainer:
        trainer.reward_fn = resolved_reward
        trainer.cfg = predictive_cfg
        trainer.primary_dim = int(predictive_cfg.primary_dim)
        trainer.extras_dim = int(predictive_cfg.extras_dim)
        trainer.context_extractor = default_ctx
        trainer.input_noise_std = resolved_noise
        trainer.device = target_device
        trainer.model = estimator.model_
        trainer.model.to(target_device)
        for group in trainer.opt.param_groups:
            group["lr"] = float(lr)
        trainer.history.clear()
        trainer.profile["epochs"] = 0
        trainer.profile["total_time_s"] = 0.0
        trainer.profile["episode_length"] = int(predictive_cfg.episode_length)
        trainer.profile["batch_episodes"] = int(predictive_cfg.batch_episodes)
        if cache_ready is None:
            trainer.extras_cache = None
        else:
            trainer.load_extras_cache(cache_ready)
    else:
        trainer = PredictiveExtrasTrainer(
            estimator.model_,
            reward_fn=resolved_reward,
            cfg=predictive_cfg,
            device=target_device,
            lr=float(lr),
            input_noise_std=resolved_noise,
            context_extractor=default_ctx,
            extras_cache=cache_ready,
        )

    trainer.train(
        X_train_arr,
        epochs=int(estimator.epochs),
        verbose=int(verbose),
        lr_max=(float(lr_max) if lr_max is not None else None),
        lr_min=(float(lr_min) if lr_min is not None else None),
    )
    return trainer


def _apply_estimator_scaler(estimator: Any, X_obs: np.ndarray) -> np.ndarray:
    if getattr(estimator, "_scaler_kind_", None) is None or estimator.preserve_shape:
        return X_obs
    X2d = X_obs.reshape(X_obs.shape[0], -1)
    kind = estimator._scaler_kind_
    st = getattr(estimator, "_scaler_state_", {})
    if kind == "standard":
        mean = st.get("mean")
        var = st.get("M2")
        n = max(st.get("n", 1), 1)
        if mean is not None and var is not None:
            std = np.sqrt(np.maximum(var / n, 1e-8)).astype(np.float32)
            X2d = (X2d - mean) / std
    elif kind == "minmax":
        mn = st.get("min")
        mx = st.get("max")
        if mn is not None and mx is not None:
            scale = np.where((mx - mn) > 1e-8, (mx - mn), 1.0)
            X2d = (X2d - mn) / scale
    elif kind == "custom" and hasattr(estimator.scaler, "transform"):
        X2d = estimator.scaler.transform(X2d)
    return X2d.reshape(X_obs.shape)


def hisso_infer_series(
    estimator: Any,
    X_obs: np.ndarray,
    *,
    trainer_cfg: HISSOTrainerConfig,
    extras_cache: Optional[np.ndarray],
    initial_extras: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    try:
        from .augmented import PredictiveExtrasTrainer
        from .episodes import portfolio_log_return_reward
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Predictive extras components not available") from exc

    predictive_cfg = trainer_cfg.to_predictive_extras()
    device = estimator._device()
    def _ctx(X_ep: torch.Tensor) -> torch.Tensor:
        return estimator._scaler_inverse_tensor(X_ep)

    trainer = PredictiveExtrasTrainer(
        estimator.model_,
        reward_fn=lambda alloc, ctx: portfolio_log_return_reward(alloc, ctx, trans_cost=trainer_cfg.transition_cost),
        cfg=predictive_cfg,
        device=device,
        context_extractor=_ctx,
        extras_cache=extras_cache,
    )
    X_in = np.asarray(X_obs, dtype=np.float32)
    X_in = _apply_estimator_scaler(estimator, X_in)
    prim, ex = trainer.infer_series(X_in, E0=initial_extras)
    new_cache = getattr(trainer, "extras_cache", None)
    return prim, ex, new_cache


def hisso_evaluate_reward(
    estimator: Any,
    X_obs: np.ndarray,
    *,
    trainer_cfg: HISSOTrainerConfig,
    extras_cache: Optional[np.ndarray],
    n_batches: int = 8,
) -> Tuple[float, Optional[np.ndarray]]:
    try:
        from .augmented import PredictiveExtrasTrainer
        from .episodes import portfolio_log_return_reward
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Predictive extras components not available") from exc

    predictive_cfg = trainer_cfg.to_predictive_extras()
    device = estimator._device()
    def _ctx(X_ep: torch.Tensor) -> torch.Tensor:
        return estimator._scaler_inverse_tensor(X_ep)

    trainer = PredictiveExtrasTrainer(
        estimator.model_,
        reward_fn=lambda alloc, ctx: portfolio_log_return_reward(alloc, ctx, trans_cost=trainer_cfg.transition_cost),
        cfg=predictive_cfg,
        device=device,
        context_extractor=_ctx,
        extras_cache=extras_cache,
    )
    X_in = np.asarray(X_obs, dtype=np.float32)
    X_in = _apply_estimator_scaler(estimator, X_in)
    val = float(trainer.evaluate_reward(X_in, n_batches=int(n_batches)))
    new_cache = getattr(trainer, "extras_cache", None)
    return val, new_cache


def ensure_hisso_trainer_config(value: Any) -> HISSOTrainerConfig:
    """Coerce persisted metadata into a HISSOTrainerConfig instance."""
    if isinstance(value, HISSOTrainerConfig):
        return value
    try:
        from .augmented import PredictiveExtrasConfig
        if isinstance(value, PredictiveExtrasConfig):
            return HISSOTrainerConfig.from_predictive_extras(value)
    except Exception:  # pragma: no cover - optional dependency
        pass
    if isinstance(value, dict):
        return HISSOTrainerConfig(
            episode_length=int(value.get("episode_length", 64)),
            episodes_per_batch=int(value.get("episodes_per_batch", value.get("batch_episodes", 32))),
            primary_dim=int(value.get("primary_dim", 1)),
            extras_dim=int(value.get("extras_dim", 0)),
            primary_transform=str(value.get("primary_transform", "softmax")),
            extras_transform=str(value.get("extras_transform", "tanh")),
            random_state=value.get("random_state", None),
            transition_cost=float(value.get("transition_cost", value.get("trans_cost", 0.0))),
            extras_supervision_weight=float(value.get("extras_supervision_weight", 0.0)),
            extras_supervision_mode=str(value.get("extras_supervision_mode", "joint")),
            extras_supervision_cycle=int(value.get("extras_supervision_cycle", 2)),
        )
    raise ValueError("Unsupported HISSO trainer configuration format")
