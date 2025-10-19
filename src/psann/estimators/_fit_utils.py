from __future__ import annotations

"""Lean training helpers for the sklearn-style estimators.

This module focuses on the primary-output pipeline now that the legacy extras
heads have been retired.  It exposes a small set of dataclasses and helper
functions that normalise fit arguments, prepare inputs/targets, build variant
models, and orchestrate supervised or HISSO training.
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..hisso import (
    HISSOOptions,
    HISSOTrainer,
    HISSOTrainerConfig,
    coerce_warmstart_config,
    run_hisso_supervised_warmstart,
    run_hisso_training,
)
from ..nn import WithPreprocessor
from ..training import TrainingLoopConfig, run_training_loop
from ..types import HISSOFitParams, NoiseSpec

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


# ---------------------------------------------------------------------------
# Core data carriers
# ---------------------------------------------------------------------------

ValidationPair = Tuple[np.ndarray, np.ndarray]
ValidationInput = ValidationPair


@dataclass
class NormalisedFitArgs:
    """Canonical view of the arguments supplied to ``fit``."""

    X: np.ndarray
    y: Optional[np.ndarray]
    validation: Optional[ValidationPair]
    hisso: bool
    hisso_options: Optional[HISSOOptions]
    noisy: Optional[NoiseSpec]
    verbose: int
    lr_max: Optional[float]
    lr_min: Optional[float]


@dataclass
class PreparedInputState:
    """Intermediate artefacts produced after scaler/shape handling."""

    X_flat: np.ndarray
    X_cf: Optional[np.ndarray]
    input_shape: Tuple[int, ...]
    internal_shape_cf: Optional[Tuple[int, ...]]
    scaler_transform: Optional[Callable[[np.ndarray], np.ndarray]]
    train_inputs: np.ndarray
    train_targets: Optional[np.ndarray]
    y_vector: Optional[np.ndarray]
    y_cf: Optional[np.ndarray]
    primary_dim: int
    output_dim: int


@dataclass
class ModelBuildRequest:
    """Bundle of information required to construct the estimator core."""

    estimator: "PSANNRegressor"
    prepared: PreparedInputState
    primary_dim: int
    lsm_module: Optional[nn.Module]
    lsm_output_dim: Optional[int]
    preserve_shape: bool


@dataclass
class HISSOTrainingPlan:
    """Precomputed artefacts required to launch HISSO training."""

    inputs: np.ndarray
    primary_dim: int
    trainer_config: HISSOTrainerConfig
    allow_full_window: bool
    options: HISSOOptions
    lsm_module: Optional[nn.Module]


class ModelFactory(Protocol):
    def __call__(self, request: ModelBuildRequest) -> nn.Module: ...


class PreprocFactory(Protocol):
    def __call__(self, request: ModelBuildRequest) -> Optional[nn.Module]: ...


class HISSOPlanFactory(Protocol):
    def __call__(
        self,
        estimator: "PSANNRegressor",
        request: ModelBuildRequest,
        *,
        fit_args: NormalisedFitArgs,
    ) -> Optional[HISSOTrainingPlan]: ...


@dataclass
class FitVariantHooks:
    """Declarative hooks that let estimator variants share the pipeline."""

    build_model: ModelFactory
    build_preprocessor: Optional[PreprocFactory] = None
    build_hisso_plan: Optional[HISSOPlanFactory] = None

    def wants_hisso(self) -> bool:
        return self.build_hisso_plan is not None


# ---------------------------------------------------------------------------
# Shared pipeline entrypoints
# ---------------------------------------------------------------------------


def normalise_fit_args(
    estimator: "PSANNRegressor",
    X: np.ndarray,
    y: Optional[np.ndarray],
    *,
    validation_data: Optional[ValidationInput],
    noisy: Optional[NoiseSpec],
    verbose: int,
    lr_max: Optional[float],
    lr_min: Optional[float],
    hisso: bool,
    hisso_kwargs: HISSOFitParams,
) -> NormalisedFitArgs:
    """Coerce inputs, targets, and validation tuples into canonical form."""

    val_pair: Optional[ValidationPair] = None
    if validation_data is not None:
        if not isinstance(validation_data, (tuple, list)):
            raise ValueError("validation_data must be a tuple (X, y)")
        val_tuple = tuple(validation_data)
        if len(val_tuple) != 2:
            raise ValueError("validation_data must be length 2")
        X_val = np.asarray(val_tuple[0], dtype=np.float32)
        y_val = np.asarray(val_tuple[1], dtype=np.float32)
        val_pair = (X_val, y_val)

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = None
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float32)

    if not hisso and y_arr is None:
        raise ValueError("y must be provided when hisso=False")

    noise_cfg: Optional[NoiseSpec] = None
    if noisy is not None:
        if np.isscalar(noisy):
            noise_cfg = float(noisy)
        else:
            noise_cfg = np.asarray(noisy, dtype=np.float32)

    hisso_options: Optional[HISSOOptions] = None
    if hisso:
        hisso_options = HISSOOptions.from_kwargs(
            window=hisso_kwargs.get("hisso_window"),
            reward_fn=hisso_kwargs.get("hisso_reward_fn"),
            context_extractor=hisso_kwargs.get("hisso_context_extractor"),
            primary_transform=hisso_kwargs.get("hisso_primary_transform"),
            transition_penalty=hisso_kwargs.get("hisso_transition_penalty"),
            trans_cost=hisso_kwargs.get("hisso_trans_cost"),
            input_noise=noise_cfg,
            supervised=hisso_kwargs.get("hisso_supervised"),
        )

    return NormalisedFitArgs(
        X=X_arr,
        y=y_arr,
        validation=val_pair,
        hisso=bool(hisso),
        hisso_options=hisso_options,
        noisy=noise_cfg,
        verbose=int(verbose),
        lr_max=float(lr_max) if lr_max is not None else None,
        lr_min=float(lr_min) if lr_min is not None else None,
    )


def prepare_inputs_and_scaler(
    estimator: "PSANNRegressor",
    fit_args: NormalisedFitArgs,
) -> Tuple[PreparedInputState, int]:
    """Apply scalers, reshape inputs, and derive primary dimensions."""

    estimator.input_shape_ = estimator._infer_input_shape(fit_args.X)

    if estimator.preserve_shape:
        return _prepare_preserve_shape_inputs(estimator, fit_args)

    return _prepare_flatten_inputs(estimator, fit_args)


def _prepare_flatten_inputs(
    estimator: "PSANNRegressor",
    fit_args: NormalisedFitArgs,
) -> Tuple[PreparedInputState, int]:
    X = fit_args.X
    y = fit_args.y

    X2d = estimator._flatten(X)
    scaler_transform = estimator._scaler_fit_update(X2d)
    if scaler_transform is not None:
        X_scaled = scaler_transform(X2d).reshape(X.shape[0], *estimator.input_shape_)
    else:
        X_scaled = X

    X_flat = estimator._flatten(X_scaled).astype(np.float32, copy=False)
    train_inputs = X_flat

    y_vec: Optional[np.ndarray] = None
    primary_dim: int
    if y is not None:
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_vec = y_arr.reshape(-1, 1)
        else:
            y_vec = y_arr.reshape(y_arr.shape[0], -1)
        primary_dim = int(y_vec.shape[1])
    else:
        if fit_args.hisso:
            if estimator.output_shape is not None:
                primary_dim = int(np.prod(estimator.output_shape))
            else:
                inferred = int(np.prod(estimator.input_shape_)) if estimator.input_shape_ else 1
                primary_dim = max(1, inferred)
        else:
            # Non-HISSO regressors require explicit targets
            primary_dim = 1

    prepared = PreparedInputState(
        X_flat=train_inputs,
        X_cf=None,
        input_shape=estimator.input_shape_,
        internal_shape_cf=None,
        scaler_transform=scaler_transform,
        train_inputs=train_inputs,
        train_targets=y_vec,
        y_vector=y_vec,
        y_cf=None,
        primary_dim=primary_dim,
        output_dim=primary_dim,
    )

    return prepared, primary_dim


def _prepare_preserve_shape_inputs(
    estimator: "PSANNRegressor",
    fit_args: NormalisedFitArgs,
) -> Tuple[PreparedInputState, int]:
    X = fit_args.X
    y = fit_args.y

    if X.ndim < 3:
        raise ValueError("preserve_shape=True requires inputs of shape (N, C, ...).")

    if estimator.data_format not in {"channels_first", "channels_last"}:
        raise ValueError("data_format must be 'channels_first' or 'channels_last'.")

    X_cf = np.moveaxis(X, -1, 1) if estimator.data_format == "channels_last" else X
    estimator._internal_input_shape_cf_ = tuple(X_cf.shape[1:])

    N, C = X_cf.shape[0], int(X_cf.shape[1])
    X2d = X_cf.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
    scaler_transform = estimator._scaler_fit_update(X2d)
    if scaler_transform is not None:
        X2d_scaled = scaler_transform(X2d)
        X_cf = X2d_scaled.reshape(N, -1, C).transpose(0, 2, 1).reshape(X_cf.shape)

    y_cf: Optional[np.ndarray] = None
    y_vec: Optional[np.ndarray] = None
    primary_dim: int

    if y is None:
        if not fit_args.hisso:
            raise ValueError("y must be provided when hisso=False (preserve_shape=True).")
        if estimator.output_shape is not None:
            primary_dim = int(np.prod(estimator.output_shape))
        else:
            primary_dim = int(np.prod(estimator._internal_input_shape_cf_))
    elif estimator.per_element:
        if estimator.output_shape is not None:
            n_targets = int(
                estimator.output_shape[0]
                if estimator.data_format == "channels_first"
                else estimator.output_shape[-1]
            )
        else:
            if estimator.data_format == "channels_first":
                n_targets = int(y.shape[1] if y.ndim == X_cf.ndim else 1)
            else:
                n_targets = int(y.shape[-1] if y.ndim == X.ndim else 1)
        if estimator.data_format == "channels_last":
            if y.ndim == X.ndim:
                y_cf = np.moveaxis(y, -1, 1)
            else:
                y_cf = y[:, None, ...]
        else:
            if y.ndim == X_cf.ndim:
                y_cf = y
            else:
                y_cf = y[:, None, ...]
        if y_cf is None:
            raise ValueError("Unable to align targets with per-element configuration.")
        y_cf = y_cf.astype(np.float32, copy=False)
        y_vec = y_cf.reshape(y_cf.shape[0], -1)
        primary_dim = int(n_targets)
    else:
        y_arr = np.asarray(y, dtype=np.float32)
        if y_arr.ndim == 1:
            y_vec = y_arr.reshape(-1, 1)
        else:
            y_vec = y_arr.reshape(y_arr.shape[0], -1)
        if estimator.output_shape is not None:
            expected = int(np.prod(estimator.output_shape))
            if y_vec.shape[1] != expected:
                raise ValueError(
                    f"y has {y_vec.shape[1]} targets; expected {expected} from output_shape."
                )
        primary_dim = int(y_vec.shape[1])

    X_flat = estimator._flatten(X).astype(np.float32, copy=False)
    use_cf_inputs = bool(
        estimator.per_element or getattr(estimator, "_use_channel_first_train_inputs_", False)
    )
    train_inputs = X_cf.astype(np.float32, copy=False) if use_cf_inputs else X_flat

    if use_cf_inputs and y_cf is not None:
        train_targets = y_cf
    elif y_vec is not None:
        train_targets = y_vec
    else:
        train_targets = None

    prepared = PreparedInputState(
        X_flat=X_flat,
        X_cf=X_cf.astype(np.float32, copy=False),
        input_shape=estimator.input_shape_,
        internal_shape_cf=estimator._internal_input_shape_cf_,
        scaler_transform=scaler_transform,
        train_inputs=train_inputs,
        train_targets=train_targets,
        y_vector=y_vec,
        y_cf=y_cf,
        primary_dim=primary_dim,
        output_dim=primary_dim,
    )

    return prepared, primary_dim


def build_model_from_hooks(
    hooks: FitVariantHooks,
    request: ModelBuildRequest,
) -> nn.Module:
    """Construct the model by delegating to the supplied hook(s)."""

    core = hooks.build_model(request)
    if not isinstance(core, nn.Module):
        raise TypeError("build_model hook must return an nn.Module instance.")
    if isinstance(core, WithPreprocessor):
        return core

    preproc: Optional[nn.Module] = request.lsm_module
    if hooks.build_preprocessor is not None:
        custom = hooks.build_preprocessor(request)
        if custom is not None:
            preproc = custom

    if preproc is None:
        return core

    return WithPreprocessor(preproc, core)


def build_hisso_training_plan(
    estimator: "PSANNRegressor",
    *,
    train_inputs: np.ndarray,
    primary_dim: int,
    fit_args: NormalisedFitArgs,
    options: HISSOOptions,
    lsm_module: Optional[nn.Module] = None,
) -> HISSOTrainingPlan:
    """Prepare HISSO trainer inputs without mutating estimator state."""

    if options is None:
        raise ValueError("HISSO options were not provided for HISSO planning.")

    inputs_arr = np.asarray(train_inputs, dtype=np.float32)

    trainer_cfg = options.to_trainer_config(
        primary_dim=int(primary_dim),
        random_state=estimator.random_state,
    )

    observed_window = int(inputs_arr.shape[0])
    if observed_window <= 0:
        raise ValueError("HISSO requires at least one timestep in X.")

    allow_full_window = observed_window >= int(trainer_cfg.episode_length)
    if not allow_full_window:
        adjusted_length = max(1, min(int(trainer_cfg.episode_length), observed_window))
        if adjusted_length != trainer_cfg.episode_length:
            trainer_cfg = replace(trainer_cfg, episode_length=adjusted_length)

    return HISSOTrainingPlan(
        inputs=inputs_arr,
        primary_dim=int(primary_dim),
        trainer_config=trainer_cfg,
        allow_full_window=allow_full_window,
        options=options,
        lsm_module=lsm_module,
    )


def maybe_run_hisso(
    hooks: FitVariantHooks,
    request: ModelBuildRequest,
    *,
    fit_args: NormalisedFitArgs,
) -> Optional[HISSOTrainer]:
    if not hooks.wants_hisso():
        return None
    plan = hooks.build_hisso_plan(
        request.estimator,
        request,
        fit_args=fit_args,
    )
    if plan is None:
        return None
    return run_hisso_stage(request.estimator, plan=plan, fit_args=fit_args)


def run_hisso_stage(
    estimator: "PSANNRegressor",
    *,
    plan: HISSOTrainingPlan,
    fit_args: NormalisedFitArgs,
) -> HISSOTrainer:
    """Execute HISSO training and update estimator state."""

    device = estimator._device()
    inputs_arr = plan.inputs

    warm_cfg = coerce_warmstart_config(plan.options.supervised, fit_args.y)
    if warm_cfg is not None:
        run_hisso_supervised_warmstart(
            estimator,
            inputs_arr,
            primary_dim=int(plan.primary_dim),
            config=warm_cfg,
            lsm_module=plan.lsm_module,
        )

    estimator._hisso_reward_fn_ = plan.options.reward_fn
    estimator._hisso_context_extractor_ = plan.options.context_extractor

    trainer = run_hisso_training(
        estimator,
        inputs_arr,
        trainer_cfg=plan.trainer_config,
        lr=float(estimator.lr),
        device=device,
        reward_fn=plan.options.reward_fn,
        context_extractor=plan.options.context_extractor,
        lr_max=float(fit_args.lr_max) if fit_args.lr_max is not None else None,
        lr_min=float(fit_args.lr_min) if fit_args.lr_min is not None else None,
        input_noise_std=plan.options.input_noise_std,
        verbose=int(fit_args.verbose),
    )

    estimator._hisso_options_ = plan.options
    estimator._hisso_trainer_ = trainer
    estimator._hisso_cfg_ = plan.trainer_config
    estimator._hisso_trained_ = True
    estimator.history_ = getattr(trainer, "history", [])
    # Backwards-compatible storage for legacy consumers.
    estimator._hisso_reward_fn_ = plan.options.reward_fn
    estimator._hisso_context_extractor_ = plan.options.context_extractor
    return trainer


def run_supervised_training(
    estimator: "PSANNRegressor",
    model: nn.Module,
    prepared: PreparedInputState,
    *,
    fit_args: NormalisedFitArgs,
) -> Dict[str, Any]:
    """Execute the optimiser/dataloader/loop flow shared by all estimators."""

    device = estimator._device()
    model.to(device)

    optimizer = _build_optimizer(estimator, model)
    estimator._optimizer_ = optimizer
    estimator._lr_scheduler_ = None

    loss_fn = estimator._make_loss()

    train_targets = prepared.train_targets
    if train_targets is None:
        if estimator.preserve_shape and prepared.y_cf is not None:
            train_targets = prepared.y_cf
        elif prepared.y_vector is not None:
            train_targets = prepared.y_vector
        else:
            raise ValueError("PreparedInputState did not contain training targets.")

    inputs_np = prepared.train_inputs.astype(np.float32, copy=False)
    targets_np = np.asarray(train_targets, dtype=np.float32)

    dataset = TensorDataset(torch.from_numpy(inputs_np), torch.from_numpy(targets_np))
    shuffle = not (estimator.stateful and estimator.state_reset in ("epoch", "none"))
    dataloader = DataLoader(
        dataset,
        batch_size=int(estimator.batch_size),
        shuffle=shuffle,
        num_workers=int(estimator.num_workers),
    )

    val_inputs_t, val_targets_t = _prepare_validation_tensors(
        estimator,
        prepared,
        fit_args.validation,
        device=device,
    )
    noise_std_t = _prepare_noise_tensor(estimator, prepared, fit_args.noisy, device)
    val_inputs = (
        _resolve_validation_inputs(estimator, model, val_inputs_t)
        if val_inputs_t is not None
        else None
    )

    cfg_loop = TrainingLoopConfig(
        epochs=int(estimator.epochs),
        patience=int(estimator.patience),
        early_stopping=bool(estimator.early_stopping),
        stateful=bool(estimator.stateful),
        state_reset=str(estimator.state_reset),
        verbose=int(fit_args.verbose),
        lr_max=None if fit_args.lr_max is None else float(fit_args.lr_max),
        lr_min=None if fit_args.lr_min is None else float(fit_args.lr_min),
    )

    gradient_hook = getattr(estimator, "gradient_hook", None)
    if not callable(gradient_hook):
        gradient_hook = None

    epoch_callback = getattr(estimator, "epoch_callback", None)
    if not callable(epoch_callback):
        epoch_callback = None

    history, best_state = run_training_loop(
        model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=dataloader,
        device=device,
        cfg=cfg_loop,
        noise_std=noise_std_t,
        val_inputs=val_inputs,
        val_targets=val_targets_t,
        gradient_hook=gradient_hook,
        epoch_callback=epoch_callback,
    )

    estimator.history_ = history
    if best_state is not None and estimator.early_stopping:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_state": best_state,
        "val_inputs": val_inputs,
        "val_targets": val_targets_t,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _build_optimizer(estimator: "PSANNRegressor", model: nn.Module) -> torch.optim.Optimizer:
    if estimator.lsm_train and isinstance(model, WithPreprocessor) and model.preproc is not None:
        params = [
            {"params": model.core.parameters(), "lr": float(estimator.lr)},
            {
                "params": model.preproc.parameters(),
                "lr": (
                    float(estimator.lsm_lr) if estimator.lsm_lr is not None else float(estimator.lr)
                ),
            },
        ]
        opt_name = str(estimator.optimizer).lower()
        if opt_name == "adamw":
            return torch.optim.AdamW(params, weight_decay=float(estimator.weight_decay))
        if opt_name == "sgd":
            return torch.optim.SGD(params, momentum=0.9)
        return torch.optim.Adam(params, weight_decay=float(estimator.weight_decay))
    return estimator._make_optimizer(model)


def _prepare_validation_tensors(
    estimator: "PSANNRegressor",
    prepared: PreparedInputState,
    validation: Optional[ValidationPair],
    *,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if validation is None:
        return None, None

    X_val = np.asarray(validation[0], dtype=np.float32)
    y_val = np.asarray(validation[1], dtype=np.float32)

    if estimator.preserve_shape:
        X_val_cf = np.moveaxis(X_val, -1, 1) if estimator.data_format == "channels_last" else X_val
        if prepared.internal_shape_cf is None:
            raise ValueError(
                "PreparedInputState missing channels-first shape for preserve_shape=True."
            )
        if X_val_cf.shape[1] != prepared.internal_shape_cf[0]:
            raise ValueError("validation_data channels mismatch.")
        X_val_t = torch.from_numpy(X_val_cf.astype(np.float32, copy=False)).to(device)
        if estimator.per_element:
            if estimator.data_format == "channels_last":
                if y_val.ndim == X_val.ndim:
                    y_val_cf = np.moveaxis(y_val, -1, 1)
                elif y_val.ndim == X_val.ndim - 1:
                    y_val_cf = y_val[:, None, ...]
                else:
                    raise ValueError(
                        "validation y must match X spatial dims with optional channel last."
                    )
            else:
                if y_val.ndim == X_val_cf.ndim:
                    y_val_cf = y_val
                elif y_val.ndim == X_val_cf.ndim - 1:
                    y_val_cf = y_val[:, None, ...]
                else:
                    raise ValueError(
                        "validation y must match X spatial dims, optional channel first."
                    )
            y_val_t = torch.from_numpy(y_val_cf.astype(np.float32, copy=False)).to(device)
        else:
            y_val_flat = y_val.reshape(y_val.shape[0], -1).astype(np.float32, copy=False)
            y_val_t = torch.from_numpy(y_val_flat).to(device)
        return X_val_t, y_val_t

    n_features = int(np.prod(estimator.input_shape_))
    if tuple(X_val.shape[1:]) != estimator.input_shape_:
        if int(np.prod(X_val.shape[1:])) != n_features:
            raise ValueError(
                f"validation_data X has shape {X_val.shape[1:]}, expected {estimator.input_shape_} "
                f"(prod must match {n_features})."
            )
    X_val_flat = estimator._flatten(X_val)
    X_val_t = torch.from_numpy(X_val_flat.astype(np.float32, copy=False)).to(device)

    y_val_flat = y_val.reshape(y_val.shape[0], -1).astype(np.float32, copy=False)
    y_val_t = torch.from_numpy(y_val_flat.astype(np.float32, copy=False)).to(device)
    return X_val_t, y_val_t


def _prepare_noise_tensor(
    estimator: "PSANNRegressor",
    prepared: PreparedInputState,
    noisy: Optional[NoiseSpec],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if noisy is None:
        return None

    if estimator.preserve_shape:
        if prepared.internal_shape_cf is None:
            raise ValueError(
                "PreparedInputState missing internal channels-first shape for noise construction."
            )
        internal_shape = prepared.internal_shape_cf
        if np.isscalar(noisy):
            std = np.full((1, *internal_shape), float(noisy), dtype=np.float32)
        else:
            arr = np.asarray(noisy, dtype=np.float32)
            if tuple(arr.shape) == internal_shape:
                std = arr.reshape(1, *internal_shape)
            elif (
                tuple(arr.shape) == estimator.input_shape_
                and estimator.data_format == "channels_last"
            ):
                std = np.moveaxis(arr, -1, 0).reshape(1, *internal_shape)
            elif arr.ndim == 1 and arr.size == int(np.prod(internal_shape)):
                std = arr.reshape(1, *internal_shape)
            else:
                raise ValueError(
                    f"noisy shape {arr.shape} not compatible with input shape {estimator.input_shape_}"
                )
        return torch.from_numpy(std.astype(np.float32, copy=False)).to(device)

    n_features = int(np.prod(estimator.input_shape_))
    if np.isscalar(noisy):
        std = np.full((1, n_features), float(noisy), dtype=np.float32)
    else:
        arr = np.asarray(noisy, dtype=np.float32)
        if arr.ndim == 1 and arr.size == n_features:
            std = arr.reshape(1, n_features)
        elif arr.ndim == 2 and arr.shape[1] == n_features:
            std = arr[:1]
        else:
            raise ValueError(
                f"noisy shape {arr.shape} not compatible with flattened feature dimension {n_features}"
            )
    return torch.from_numpy(std.astype(np.float32, copy=False)).to(device)


def _resolve_validation_inputs(
    estimator: "PSANNRegressor",
    model: nn.Module,
    inputs: torch.Tensor,
) -> torch.Tensor:
    val_inputs = inputs
    preproc = None
    if isinstance(model, WithPreprocessor) and model.preproc is not None:
        preproc = model.preproc
    elif hasattr(estimator, "lsm") and estimator.lsm is not None:
        preproc = estimator.lsm
    if preproc is None:
        return val_inputs

    if hasattr(preproc, "eval"):
        preproc.eval()
    if hasattr(preproc, "forward"):
        with torch.no_grad():
            val_inputs = preproc(inputs)
    return val_inputs
