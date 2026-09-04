from __future__ import annotations

"""Lean training helpers for the sklearn-style estimators."""

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..episodic.legacy_config import HISSOWarmStartConfig, coerce_warmstart_config
from ..episodic.runtime_loop import HISSOTrainer, run_hisso_training
from ..episodic.warmstart import run_hisso_supervised_warmstart
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
    model_context: Optional[np.ndarray] = None,
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
        model_context=(
            None if model_context is None else np.asarray(model_context, dtype=np.float32)
        ),
    )


def maybe_run_hisso(
    hooks: FitVariantHooks,
    request: ModelBuildRequest,
    *,
    fit_args: NormalisedFitArgs,
) -> Optional[HISSOTrainer]:
    if not hooks.wants_hisso():
        return None
    assert hooks.build_hisso_plan is not None
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

    canonical = getattr(estimator, "_episodic_strategy_request_", None)
    if canonical is not None:
        from ..episodic import HISSOConfig, resolve_reward

        if not isinstance(canonical, HISSOConfig):
            raise TypeError("_episodic_strategy_request_ must be a HISSOConfig.")
        random_state = (
            canonical.schedule.random_state
            if canonical.schedule.random_state is not None
            else estimator.random_state
        )
        plan.trainer_config = plan.trainer_config.__class__(
            episode_length=canonical.schedule.episode_length,
            episodes_per_batch=canonical.schedule.batch_episodes
            * canonical.schedule.updates_per_epoch,
            episode_batch_size=canonical.schedule.batch_episodes,
            updates_per_epoch=canonical.schedule.updates_per_epoch,
            primary_dim=plan.primary_dim,
            primary_transform=canonical.primary_transform,
            random_state=random_state,
            transition_penalty=canonical.transition_penalty,
        )
        reward_fn = resolve_reward(canonical.reward)
        stateful_restricted = bool(estimator.stateful) and estimator.state_reset in {
            "epoch",
            "none",
        }
        if (
            canonical.warm_start is not None
            and canonical.warm_start.shuffle is True
            and stateful_restricted
        ):
            raise ValueError(
                "strategy.warm_start.shuffle=True is invalid for stateful epoch/none cadence."
            )
        warm_cfg = None
        if canonical.warm_start is not None:
            warm = canonical.warm_start
            warm_cfg = HISSOWarmStartConfig(
                targets=cast(np.ndarray, fit_args.y),
                epochs=warm.epochs,
                batch_size=warm.batch_size,
                lr=warm.lr,
                weight_decay=warm.weight_decay,
                lsm_lr=warm.preprocessor_lr,
                shuffle=(not stateful_restricted) if warm.shuffle is None else warm.shuffle,
            )
    else:
        reward_fn = cast(Any, plan.options.reward_fn)
        warm_cfg = coerce_warmstart_config(plan.options.supervised, fit_args.y)
    if warm_cfg is not None:
        run_hisso_supervised_warmstart(
            estimator,
            inputs_arr,
            primary_dim=int(plan.primary_dim),
            config=warm_cfg,
            lsm_module=plan.lsm_module,
        )

    context_extractor = (
        cast(Any, canonical.context_extractor)
        if canonical is not None
        else plan.options.context_extractor
    )
    estimator._hisso_reward_fn_ = reward_fn
    estimator._hisso_context_extractor_ = context_extractor

    # HISSO shares canonical parameter groups with supervised optimisation, but
    # retains its historical Adam algorithm.  Optimizer selection is an episodic
    # compatibility contract, not an estimator-level preprocessing policy.
    hisso_optimizer = _build_hisso_optimizer(estimator, estimator.model_)
    estimator._optimizer_ = hisso_optimizer
    trainer = run_hisso_training(
        estimator,
        inputs_arr,
        trainer_cfg=plan.trainer_config,
        lr=float(estimator.lr),
        optimizer=hisso_optimizer,
        device=device,
        reward_fn=reward_fn,
        context_extractor=context_extractor,
        lr_max=float(fit_args.lr_max) if fit_args.lr_max is not None else None,
        lr_min=float(fit_args.lr_min) if fit_args.lr_min is not None else None,
        input_noise_std=(
            canonical.input_noise_std if canonical is not None else plan.options.input_noise_std
        ),
        verbose=int(fit_args.verbose),
        use_amp=(
            canonical.mixed_precision
            if canonical is not None
            else bool(getattr(estimator, "_hisso_use_amp", False))
        ),
        amp_dtype=(
            getattr(torch, canonical.amp_dtype)
            if canonical is not None
            else getattr(estimator, "_hisso_amp_dtype", None)
        ),
        gradient_clip=(canonical.gradient_clip if canonical is not None else 1.0),
        strict=canonical is not None,
        model_context=plan.model_context,
        action_postprocessor=(
            estimator._inverse_fitted_target_scaler_tensor if canonical is not None else None
        ),
    )

    estimator._hisso_options_ = plan.options
    estimator._hisso_trainer_ = trainer
    estimator._hisso_cfg_ = plan.trainer_config
    estimator._hisso_trained_ = True
    estimator.history_ = getattr(trainer, "history", [])
    estimator._hisso_reward_fn_ = reward_fn
    estimator._hisso_context_extractor_ = context_extractor
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

    inputs_np: np.ndarray = prepared.train_inputs.astype(np.float32, copy=False)
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


def _build_optimizer_parameter_groups(
    estimator: "PSANNRegressor", model: nn.Module
) -> list[dict[str, object]]:
    """Return labeled canonical groups without choosing an optimizer algorithm."""

    training = getattr(getattr(estimator, "preprocessor", None), "training", None)
    joint_preprocessor = bool(getattr(training, "trainable", False))
    legacy_joint = training is None and bool(getattr(estimator, "lsm_train", False))
    if (
        (joint_preprocessor or legacy_joint)
        and isinstance(model, WithPreprocessor)
        and model.preproc is not None
    ):
        return [
            {
                "params": [p for p in model.core.parameters() if p.requires_grad],
                "lr": float(estimator.lr),
                "psann_parameter_group": "core",
            },
            {
                "params": [p for p in model.preproc.parameters() if p.requires_grad],
                "lr": (
                    float(training.lr)
                    if training is not None and training.lr is not None
                    else (
                        float(estimator.lsm_lr)
                        if getattr(estimator, "lsm_lr", None) is not None
                        else float(estimator.lr)
                    )
                ),
                "psann_parameter_group": "preprocessor",
            },
        ]
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    params = [
        {
            "params": trainable_params,
            "lr": float(estimator.lr),
            "psann_parameter_group": "core",
        }
    ]
    return params


def _build_optimizer(estimator: "PSANNRegressor", model: nn.Module) -> torch.optim.Optimizer:
    """Construct the supervised optimizer using the estimator-selected algorithm."""

    params = _build_optimizer_parameter_groups(estimator, model)
    if str(estimator.optimizer).lower() == "adamw":
        return torch.optim.AdamW(
            params,
            weight_decay=float(estimator.weight_decay),
        )
    if str(estimator.optimizer).lower() == "sgd":
        return torch.optim.SGD(params, momentum=0.9)
    return torch.optim.Adam(
        params,
        weight_decay=float(estimator.weight_decay),
    )


def _build_hisso_optimizer(estimator: "PSANNRegressor", model: nn.Module) -> torch.optim.Optimizer:
    """Construct the retained Adam HISSO optimizer over canonical groups."""

    return torch.optim.Adam(_build_optimizer_parameter_groups(estimator, model))


__all__ = [
    "FitVariantHooks",
    "HISSOTrainingPlan",
    "ModelBuildRequest",
    "NormalisedFitArgs",
    "PreparedInputState",
    "ValidationInput",
    "_build_optimizer",
    "_build_hisso_optimizer",
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
