from __future__ import annotations

"""Lean training helpers for the sklearn-style estimators."""

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
from ..training import TrainingLoopConfig, run_training_loop
from ._fit_args import normalise_fit_args
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
    dataloader = DataLoader(
        dataset,
        batch_size=int(estimator.batch_size),
        shuffle=shuffle,
        num_workers=int(estimator.num_workers),
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
        val_context=val_context_t,
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
        "val_context": val_context_t,
    }


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
