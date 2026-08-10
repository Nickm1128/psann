from __future__ import annotations

"""Lean training helpers for the sklearn-style estimators."""

import hashlib
import logging
import random
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..hisso import (
    HISSOTrainer,
    coerce_warmstart_config,
    run_hisso_supervised_warmstart,
    run_hisso_training,
)
from ..nn import WithPreprocessor
from ..training import (
    TrainingLoopConfig,
    TrainingLoopResumeState,
    run_training_loop,
)
from ..training_checkpoint import (
    TrainingCheckpointError,
    TrainingCheckpointManager,
    capture_rng_state,
    load_training_checkpoint,
    restore_rng_state,
    restore_scaler_state,
)
from ..training_events import TrainingEvent
from ._fit_args import normalise_fit_args
from ._fit_contracts import validate_prediction_target_shape
from ._fit_inputs import prepare_inputs_and_scaler
from ._fit_types import (
    FitVariantHooks,
    HISSOTrainingPlan,
    ModelBuildRequest,
    NormalisedFitArgs,
    PreparedInputState,
    ValidationInput,
)
from ._fit_validation import (
    _prepare_noise_tensor,
    _prepare_validation_tensors,
    _resolve_validation_inputs,
)

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


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
    options,
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
        use_amp=bool(getattr(estimator, "_hisso_use_amp", False)),
        amp_dtype=getattr(estimator, "_hisso_amp_dtype", None),
    )

    estimator._hisso_options_ = plan.options
    estimator._hisso_trainer_ = trainer
    estimator._hisso_cfg_ = plan.trainer_config
    estimator._hisso_trained_ = True
    estimator.history_ = getattr(trainer, "history", [])
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
    estimator._ensure_model_device(device)
    model = estimator.model_

    validate_prediction_target_shape(model, prepared, device=device)

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
    context_np = None
    if prepared.train_context is not None:
        context_np = np.asarray(prepared.train_context, dtype=np.float32)
        if context_np.shape[0] != inputs_np.shape[0]:
            raise ValueError("Context array must align with training inputs along the batch axis.")

    inputs_t = torch.from_numpy(inputs_np)
    targets_t = torch.from_numpy(targets_np)
    if context_np is not None:
        context_t = torch.from_numpy(context_np.astype(np.float32, copy=False))
        dataset = TensorDataset(inputs_t, context_t, targets_t)
    else:
        dataset = TensorDataset(inputs_t, targets_t)
    shuffle = not (estimator.stateful and estimator.state_reset in ("epoch", "none"))
    data_loader_generator = torch.Generator()
    data_loader_generator.manual_seed(
        int(estimator.random_state)
        if estimator.random_state is not None
        else int(torch.initial_seed())
    )

    def _seed_worker(worker_id: int) -> None:
        del worker_id
        worker_seed = int(torch.initial_seed() % (2**32 - 1))
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=int(estimator.batch_size),
        shuffle=shuffle,
        num_workers=int(estimator.num_workers),
        generator=data_loader_generator,
        worker_init_fn=_seed_worker if int(estimator.num_workers) > 0 else None,
    )

    val_inputs_t, val_targets_t, val_context_t = _prepare_validation_tensors(
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
    for name, tensor in {
        "validation inputs": val_inputs,
        "validation targets": val_targets_t,
        "validation context": val_context_t,
        "noise standard deviation": noise_std_t,
    }.items():
        if tensor is not None and not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or infinity after preprocessing.")

    optimizer = _build_optimizer(estimator, model)
    estimator._optimizer_ = optimizer
    scheduler = _build_scheduler(estimator, optimizer, fit_args)
    estimator._lr_scheduler_ = scheduler
    loss_fn = estimator._make_loss()

    def _resolve_amp_dtype(value: Any) -> Optional[torch.dtype]:
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            aliases = {
                "bf16": "bfloat16",
                "bfloat16": "bfloat16",
                "fp16": "float16",
                "float16": "float16",
                "fp32": "float32",
                "float32": "float32",
            }
            key = aliases.get(key, key)
            return getattr(torch, key, None)
        return None

    use_amp = bool(getattr(estimator, "amp", False))
    amp_dtype = _resolve_amp_dtype(getattr(estimator, "amp_dtype", None))
    compile_model = bool(getattr(estimator, "compile", False))

    cfg_loop = TrainingLoopConfig(
        epochs=int(estimator.epochs),
        patience=int(estimator.patience),
        early_stopping=bool(estimator.early_stopping),
        stateful=bool(estimator.stateful),
        state_reset=str(estimator.state_reset),
        verbose=int(fit_args.verbose),
        lr_max=None if fit_args.lr_max is None else float(fit_args.lr_max),
        lr_min=None if fit_args.lr_min is None else float(fit_args.lr_min),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        compile_model=compile_model,
        compile_backend=str(getattr(estimator, "compile_backend", "inductor")),
        compile_mode=str(getattr(estimator, "compile_mode", "default")),
        compile_fullgraph=bool(getattr(estimator, "compile_fullgraph", False)),
        compile_dynamic=bool(getattr(estimator, "compile_dynamic", False)),
        nonfinite_policy=fit_args.nonfinite_policy,
        fallback_policy=fit_args.fallback_policy,
        callback_error_policy=fit_args.callback_error_policy,
        deterministic=fit_args.deterministic,
        seed=estimator.random_state,
    )

    gradient_hook = getattr(estimator, "gradient_hook", None)
    if not callable(gradient_hook):
        gradient_hook = None

    epoch_callback = getattr(estimator, "epoch_callback", None)
    if not callable(epoch_callback):
        epoch_callback = None

    estimator.training_events_ = []

    def _collect_event(event: TrainingEvent) -> None:
        estimator.training_events_.append(event.as_dict())
        if event.name == "train_start":
            estimator.training_metadata_ = dict(event.data)

    callbacks = (_collect_event, *fit_args.callbacks)
    logger = fit_args.logger
    if logger is None and fit_args.verbose:
        logger = logging.getLogger("psann.training")

    data_signature = _build_data_signature(prepared)
    resume_state = TrainingLoopResumeState()
    if fit_args.resume_from is not None:
        checkpoint = load_training_checkpoint(
            fit_args.resume_from,
            map_location=device,
        )
        _restore_training_checkpoint(
            estimator,
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_name=fit_args.scheduler,
            data_signature=data_signature,
            data_loader_generator=data_loader_generator,
            deterministic=fit_args.deterministic,
        )
        resume_state = TrainingLoopResumeState(
            start_epoch=int(checkpoint.get("epoch", 0)),
            global_step=int(checkpoint.get("global_step", 0)),
            best_metric=float(checkpoint.get("best_metric", float("inf"))),
            best_epoch=(
                int(checkpoint["best_epoch"]) if checkpoint.get("best_epoch") is not None else None
            ),
            patience_left=(
                int(checkpoint["patience_left"])
                if checkpoint.get("patience_left") is not None
                else None
            ),
            best_state=checkpoint.get("best_state"),
            history=[dict(entry) for entry in checkpoint.get("history", [])],
            amp_scaler_state=checkpoint.get("amp_scaler_state"),
        )

    checkpoint_manager = (
        TrainingCheckpointManager(
            fit_args.checkpoint_dir,
            periodic_every=fit_args.checkpoint_every,
            keep_periodic=fit_args.checkpoint_keep,
        )
        if fit_args.checkpoint_dir is not None
        else None
    )

    def _checkpoint_callback(
        loop_snapshot: Any,
        improved: bool,
    ) -> list[tuple[str, str]]:
        if checkpoint_manager is None:
            return []
        state = {
            "estimator_class": estimator.__class__.__name__,
            "model_signature": getattr(estimator, "_model_signature_", None),
            "data_signature": data_signature,
            "optimizer_name": str(estimator.optimizer).lower(),
            "scheduler_name": fit_args.scheduler,
            "deterministic": bool(fit_args.deterministic),
            "model_state": {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_kind": getattr(estimator, "_scaler_kind_", None),
            "scaler_state": getattr(estimator, "_scaler_state_", None),
            "scaler_spec": getattr(estimator, "_scaler_spec_", None),
            "target_scaler_kind": getattr(estimator, "_target_scaler_kind_", None),
            "target_scaler_state": getattr(estimator, "_target_scaler_state_", None),
            "target_scaler_spec": getattr(estimator, "_target_scaler_spec_", None),
            "rng_state": capture_rng_state(
                data_loader_generator=data_loader_generator,
            ),
            **dict(loop_snapshot),
        }
        paths = checkpoint_manager.save(
            state,
            epoch=int(loop_snapshot["epoch"]),
            improved=improved,
        )
        return [(kind, str(path)) for kind, path in paths]

    def _capture_amp_scaler(scaler: Optional[object]) -> None:
        estimator._amp_scaler_ = scaler

    metadata = {
        "train_input_shape": tuple(prepared.train_inputs.shape),
        "train_target_shape": tuple(targets_np.shape),
        "validation_input_shape": (tuple(val_inputs.shape) if val_inputs is not None else None),
        "validation_target_shape": (
            tuple(val_targets_t.shape) if val_targets_t is not None else None
        ),
        "scheduler": fit_args.scheduler,
        "fallback_policy": fit_args.fallback_policy,
        "nonfinite_policy": fit_args.nonfinite_policy,
        "preflight_fallbacks": list(getattr(estimator, "_fit_fallbacks_", [])),
    }

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
        val_context=val_context_t,
        gradient_hook=gradient_hook,
        epoch_callback=epoch_callback,
        metrics=fit_args.metrics,
        scheduler=scheduler,
        callbacks=callbacks,
        logger=logger,
        resume_state=resume_state,
        checkpoint_callback=(_checkpoint_callback if checkpoint_manager is not None else None),
        runtime_state_callback=_capture_amp_scaler,
        metadata=metadata,
        initial_fallbacks=getattr(estimator, "_fit_fallbacks_", []),
    )

    estimator.history_ = history
    if best_state is not None and estimator.early_stopping:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_state": best_state,
        "val_inputs": val_inputs,
        "val_targets": val_targets_t,
        "val_context": val_context_t,
    }


def _build_scheduler(
    estimator: "PSANNRegressor",
    optimizer: torch.optim.Optimizer,
    fit_args: NormalisedFitArgs,
) -> Optional[Any]:
    params = dict(fit_args.scheduler_params)
    if fit_args.scheduler == "none":
        return None
    if fit_args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(params.get("step_size", 1)),
            gamma=float(params.get("gamma", 0.1)),
        )
    if fit_args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(params.get("t_max", estimator.epochs)),
            eta_min=float(params.get("eta_min", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler {fit_args.scheduler!r}.")


def _array_digest(array: Optional[np.ndarray]) -> Optional[str]:
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(repr(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _build_data_signature(prepared: PreparedInputState) -> dict[str, Any]:
    return {
        "inputs_shape": tuple(prepared.train_inputs.shape),
        "targets_shape": (
            tuple(prepared.train_targets.shape) if prepared.train_targets is not None else None
        ),
        "context_shape": (
            tuple(prepared.train_context.shape) if prepared.train_context is not None else None
        ),
        "inputs_sha256": _array_digest(prepared.train_inputs),
        "targets_sha256": _array_digest(prepared.train_targets),
        "context_sha256": _array_digest(prepared.train_context),
    }


def _restore_training_checkpoint(
    estimator: "PSANNRegressor",
    checkpoint: Dict[str, Any],
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scheduler_name: str,
    data_signature: Dict[str, Any],
    data_loader_generator: torch.Generator,
    deterministic: bool,
) -> None:
    expected_class = estimator.__class__.__name__
    if checkpoint.get("estimator_class") != expected_class:
        raise TrainingCheckpointError(
            "Training checkpoint estimator mismatch: expected "
            f"{expected_class!r}, received {checkpoint.get('estimator_class')!r}."
        )
    if checkpoint.get("model_signature") != getattr(estimator, "_model_signature_", None):
        raise TrainingCheckpointError(
            "Training checkpoint model signature does not match the current estimator "
            "architecture or input/target shape."
        )
    if checkpoint.get("data_signature") != data_signature:
        raise TrainingCheckpointError(
            "Training checkpoint data signature does not match the supplied training "
            "inputs, targets, and context."
        )
    if checkpoint.get("optimizer_name") != str(estimator.optimizer).lower():
        raise TrainingCheckpointError(
            "Training checkpoint optimizer does not match the current estimator."
        )
    if checkpoint.get("scheduler_name") != scheduler_name:
        raise TrainingCheckpointError(
            "Training checkpoint scheduler does not match the current fit configuration."
        )
    if bool(checkpoint.get("deterministic", False)) != bool(deterministic):
        raise TrainingCheckpointError(
            "Training checkpoint deterministic mode does not match the current fit."
        )

    model_state = checkpoint.get("model_state")
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
        raise TrainingCheckpointError("Training checkpoint is missing model or optimizer state.")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler is None and scheduler_state is not None:
        raise TrainingCheckpointError(
            "Training checkpoint contains scheduler state but scheduler='none'."
        )
    if scheduler is not None:
        if scheduler_state is None:
            raise TrainingCheckpointError(
                "Training checkpoint is missing the configured scheduler state."
            )
        scheduler.load_state_dict(scheduler_state)

    estimator._scaler_kind_ = checkpoint.get("scaler_kind")
    estimator._scaler_state_ = restore_scaler_state(checkpoint.get("scaler_state"))
    estimator._scaler_spec_ = restore_scaler_state(checkpoint.get("scaler_spec"))
    estimator._target_scaler_kind_ = checkpoint.get("target_scaler_kind")
    estimator._target_scaler_state_ = restore_scaler_state(checkpoint.get("target_scaler_state"))
    estimator._target_scaler_spec_ = restore_scaler_state(checkpoint.get("target_scaler_spec"))
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        raise TrainingCheckpointError("Training checkpoint is missing RNG state.")
    restore_rng_state(
        rng_state,
        data_loader_generator=data_loader_generator,
    )


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
        if opt_name == "adam":
            return torch.optim.Adam(params, weight_decay=float(estimator.weight_decay))
        raise ValueError(f"Unknown optimizer {estimator.optimizer!r}. Supported: adam, adamw, sgd.")
    return estimator._make_optimizer(model)


__all__ = [
    "FitVariantHooks",
    "HISSOTrainingPlan",
    "ModelBuildRequest",
    "NormalisedFitArgs",
    "PreparedInputState",
    "ValidationInput",
    "_build_optimizer",
    "_prepare_noise_tensor",
    "_prepare_validation_tensors",
    "build_hisso_training_plan",
    "build_model_from_hooks",
    "maybe_run_hisso",
    "normalise_fit_args",
    "prepare_inputs_and_scaler",
    "run_hisso_stage",
    "run_hisso_supervised_warmstart",
    "run_supervised_training",
]
