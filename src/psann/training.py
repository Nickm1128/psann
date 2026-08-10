from __future__ import annotations

import inspect
import math
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from .training_events import TrainingEventCallback, TrainingEventDispatcher


@dataclass
class TrainingLoopConfig:
    epochs: int
    patience: int
    early_stopping: bool
    stateful: bool
    state_reset: str
    verbose: int
    lr_max: Optional[float]
    lr_min: Optional[float]
    use_amp: bool = False
    amp_dtype: Optional[torch.dtype] = None
    compile_model: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False
    nonfinite_policy: str = "error"
    fallback_policy: str = "warn"
    callback_error_policy: str = "raise"
    deterministic: bool = False
    seed: Optional[int] = None


@dataclass
class TrainingLoopResumeState:
    start_epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    best_epoch: Optional[int] = None
    patience_left: Optional[int] = None
    best_state: Optional[dict[str, torch.Tensor]] = None
    history: list[dict[str, Any]] = field(default_factory=list)
    amp_scaler_state: Optional[Mapping[str, Any]] = None


MetricCallable = Callable[[torch.Tensor, torch.Tensor], Any]
CheckpointCallback = Callable[
    [Mapping[str, Any], bool],
    Sequence[tuple[str, str]],
]
RuntimeStateCallback = Callable[[Optional[object]], None]


def _grad_norm(model_ref: torch.nn.Module) -> Optional[float]:
    total_sq = 0.0
    has_grad = False
    for param in model_ref.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        total_sq += float((grad.float() ** 2).sum().item())
        has_grad = True
    if not has_grad:
        return None
    return math.sqrt(total_sq)


def _metric_value(
    metric: MetricCallable,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    value = metric(prediction.detach(), target.detach())
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            raise ValueError("Training metric returned an empty tensor.")
        value = value.detach().float().mean().item()
    return float(value)


def _invoke_legacy_callback(
    callback: Callable[..., None],
    *args: Any,
    policy: str,
    name: str,
) -> None:
    try:
        callback(*args)
    except Exception as exc:
        if policy == "warn":
            warnings.warn(
                f"{name} failed and was ignored because callback_error_policy='warn': {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            return
        raise


def _handle_fallback(
    dispatcher: TrainingEventDispatcher,
    cfg: TrainingLoopConfig,
    *,
    component: str,
    requested: Any,
    effective: Any,
    reason: str,
) -> None:
    data = {
        "component": component,
        "requested": requested,
        "effective": effective,
        "reason": reason,
    }
    if cfg.fallback_policy == "error":
        raise RuntimeError(
            f"{component} fallback was required ({reason}) and fallback_policy='error'."
        )
    warnings.warn(
        f"{component} fallback: {reason}; using {effective!r}.",
        RuntimeWarning,
        stacklevel=3,
    )
    dispatcher.emit("fallback", data=data)


def _compile_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    cfg: TrainingLoopConfig,
    dispatcher: TrainingEventDispatcher,
) -> tuple[torch.nn.Module, torch.nn.Module, bool]:
    if not cfg.compile_model:
        return model, model, False
    if device.type != "cuda" or not torch.cuda.is_available():
        _handle_fallback(
            dispatcher,
            cfg,
            component="compile",
            requested=True,
            effective=False,
            reason="the current training device is not an available CUDA accelerator",
        )
        return model, model, False
    if not hasattr(torch, "compile"):
        _handle_fallback(
            dispatcher,
            cfg,
            component="compile",
            requested=True,
            effective=False,
            reason="torch.compile is unavailable in the installed PyTorch runtime",
        )
        return model, model, False

    compile_fn = getattr(torch, "compile")
    compile_kwargs: dict[str, object] = {}
    try:
        signature = inspect.signature(compile_fn)
        if "backend" in signature.parameters:
            compile_kwargs["backend"] = str(cfg.compile_backend)
        if "mode" in signature.parameters:
            compile_kwargs["mode"] = str(cfg.compile_mode)
        if "fullgraph" in signature.parameters:
            compile_kwargs["fullgraph"] = bool(cfg.compile_fullgraph)
        if "dynamic" in signature.parameters:
            compile_kwargs["dynamic"] = bool(cfg.compile_dynamic)
    except (TypeError, ValueError):
        compile_kwargs = {
            "backend": str(cfg.compile_backend),
            "mode": str(cfg.compile_mode),
            "fullgraph": bool(cfg.compile_fullgraph),
            "dynamic": bool(cfg.compile_dynamic),
        }

    try:
        train_model = compile_fn(model, **compile_kwargs)
    except Exception as exc:
        _handle_fallback(
            dispatcher,
            cfg,
            component="compile",
            requested=True,
            effective=False,
            reason=f"torch.compile failed: {exc}",
        )
        return model, model, False
    return train_model, getattr(train_model, "_orig_mod", model), True


def run_training_loop(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    train_loader: DataLoader,
    device: torch.device,
    cfg: TrainingLoopConfig,
    noise_std: Optional[torch.Tensor] = None,
    val_inputs: Optional[torch.Tensor] = None,
    val_targets: Optional[torch.Tensor] = None,
    val_context: Optional[torch.Tensor] = None,
    gradient_hook: Optional[Callable[[torch.nn.Module], None]] = None,
    epoch_callback: Optional[
        Callable[[int, float, Optional[float], bool, Optional[int]], None]
    ] = None,
    metrics: Optional[Mapping[str, MetricCallable]] = None,
    scheduler: Optional[Any] = None,
    callbacks: Optional[Sequence[TrainingEventCallback]] = None,
    logger: Optional[Any] = None,
    resume_state: Optional[TrainingLoopResumeState] = None,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    runtime_state_callback: Optional[RuntimeStateCallback] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    initial_fallbacks: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[list[dict], Optional[dict]]:
    """Run supervised optimisation with events, policies, metrics, and resume state."""

    dispatcher = TrainingEventDispatcher(
        callbacks,
        logger=logger,
        callback_error_policy=cfg.callback_error_policy,
    )
    metric_functions = dict(metrics or {})
    resume = resume_state or TrainingLoopResumeState()
    train_model = model
    state_model = model
    compile_effective = False

    try:
        for fallback in initial_fallbacks or ():
            dispatcher.emit("fallback", data=dict(fallback))
        train_model, state_model, compile_effective = _compile_model(
            model,
            device=device,
            cfg=cfg,
            dispatcher=dispatcher,
        )

        use_amp = bool(cfg.use_amp) and device.type == "cuda" and torch.cuda.is_available()
        amp_dtype = cfg.amp_dtype if cfg.amp_dtype is not None else torch.bfloat16
        if cfg.use_amp and not use_amp:
            _handle_fallback(
                dispatcher,
                cfg,
                component="amp",
                requested=True,
                effective=False,
                reason="AMP requires an available CUDA training device",
            )
        if (
            use_amp
            and amp_dtype == torch.bfloat16
            and not getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        ):
            _handle_fallback(
                dispatcher,
                cfg,
                component="amp_dtype",
                requested="bfloat16",
                effective="float16",
                reason="the active CUDA device does not report bfloat16 support",
            )
            amp_dtype = torch.float16

        def _amp_context() -> Any:
            if not use_amp:
                return nullcontext()
            try:
                return torch.autocast(device.type, dtype=amp_dtype)  # type: ignore[attr-defined]
            except (AttributeError, TypeError):
                return torch.cuda.amp.autocast(dtype=amp_dtype)  # type: ignore[attr-defined]

        scaler: Optional[object] = None
        if use_amp and amp_dtype == torch.float16:
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:
                scaler = torch.cuda.amp.GradScaler(enabled=True)
            if resume.amp_scaler_state is not None:
                scaler.load_state_dict(dict(resume.amp_scaler_state))
        if runtime_state_callback is not None:
            runtime_state_callback(scaler)

        best = float(resume.best_metric)
        patience = (
            int(resume.patience_left) if resume.patience_left is not None else int(cfg.patience)
        )
        best_state = resume.best_state
        best_epoch = resume.best_epoch
        history = [dict(entry) for entry in resume.history]
        global_step = int(resume.global_step)
        wall_start = time.perf_counter()
        prior_train_time = float(history[-1].get("train_time_s_total", 0.0)) if history else 0.0

        parameter_count = sum(parameter.numel() for parameter in state_model.parameters())
        trainable_parameter_count = sum(
            parameter.numel() for parameter in state_model.parameters() if parameter.requires_grad
        )
        start_data = {
            **dict(metadata or {}),
            "device": str(device),
            "dtype": str(next(state_model.parameters()).dtype),
            "seed": cfg.seed,
            "deterministic": bool(cfg.deterministic),
            "optimizer": optimizer.__class__.__name__,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "parameter_count": int(parameter_count),
            "trainable_parameter_count": int(trainable_parameter_count),
            "compile_requested": bool(cfg.compile_model),
            "compile_effective": bool(compile_effective),
            "amp_requested": bool(cfg.use_amp),
            "amp_effective": bool(use_amp),
            "amp_dtype": str(amp_dtype) if use_amp else None,
            "start_epoch": int(resume.start_epoch),
            "target_epochs": int(cfg.epochs),
        }
        dispatcher.emit("train_start", data=start_data)

        for epoch_index in range(int(resume.start_epoch), cfg.epochs):
            epoch_number = epoch_index + 1
            if cfg.lr_max is not None and cfg.lr_min is not None:
                if cfg.epochs <= 1:
                    lr_epoch = float(cfg.lr_min)
                else:
                    fraction = float(epoch_index) / float(max(cfg.epochs - 1, 1))
                    lr_epoch = (
                        float(cfg.lr_max) + (float(cfg.lr_min) - float(cfg.lr_max)) * fraction
                    )
                for group in optimizer.param_groups:
                    group["lr"] = lr_epoch

            dispatcher.emit(
                "epoch_start",
                epoch=epoch_number,
                step=global_step,
                data={"learning_rates": [float(group["lr"]) for group in optimizer.param_groups]},
            )

            if cfg.stateful and cfg.state_reset == "epoch" and hasattr(state_model, "reset_state"):
                state_model.reset_state()

            train_model.train()
            if train_model is not state_model:
                state_model.train()
            total = 0.0
            count = 0
            successful_steps = 0
            attempted_steps = 0
            skipped_steps = 0
            step_times: list[float] = []
            grad_norms: list[float] = []
            loss_nonfinite = 0
            grad_nonfinite = 0
            metric_totals = {name: 0.0 for name in metric_functions}
            metric_counts = {name: 0 for name in metric_functions}
            epoch_start = time.perf_counter()

            for batch in train_loader:
                attempted_steps += 1
                context_batch: Optional[torch.Tensor] = None
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 3:
                        inputs_batch, context_batch, targets_batch = batch
                    elif len(batch) == 2:
                        inputs_batch, targets_batch = batch
                    else:
                        raise ValueError(
                            "Unexpected batch tuple length encountered during training."
                        )
                else:
                    raise ValueError("Training batches must be tuple/list tensors.")

                if (
                    cfg.stateful
                    and cfg.state_reset == "batch"
                    and hasattr(state_model, "reset_state")
                ):
                    state_model.reset_state()
                inputs_batch = inputs_batch.to(device)
                if context_batch is not None:
                    context_batch = context_batch.to(device)
                targets_batch = targets_batch.to(device)
                if noise_std is not None:
                    inputs_batch = inputs_batch + torch.randn_like(inputs_batch) * noise_std
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_start = time.perf_counter()

                with _amp_context():
                    prediction = (
                        train_model(inputs_batch, context_batch)
                        if context_batch is not None
                        else train_model(inputs_batch)
                    )
                    loss = loss_fn(prediction, targets_batch)
                if loss.ndim != 0:
                    raise ValueError(
                        "Loss function returned a non-scalar tensor with shape "
                        f"{tuple(loss.shape)}; use loss_reduction='mean' or 'sum'."
                    )
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    loss_nonfinite += 1
                    dispatcher.emit(
                        "nonfinite_step",
                        epoch=epoch_number,
                        step=global_step,
                        data={"kind": "loss", "policy": cfg.nonfinite_policy},
                    )
                    if cfg.nonfinite_policy == "error":
                        raise FloatingPointError(
                            f"Non-finite loss detected at epoch {epoch_number}, "
                            f"attempted step {attempted_steps}."
                        )
                    if cfg.nonfinite_policy == "skip_step":
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        continue

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                grad_norm = _grad_norm(state_model)
                gradient_is_finite = grad_norm is None or math.isfinite(grad_norm)
                if grad_norm is not None:
                    grad_norms.append(grad_norm)
                if not gradient_is_finite:
                    grad_nonfinite += 1
                    dispatcher.emit(
                        "nonfinite_step",
                        epoch=epoch_number,
                        step=global_step,
                        data={"kind": "gradient", "policy": cfg.nonfinite_policy},
                    )
                    if cfg.nonfinite_policy == "error":
                        raise FloatingPointError(
                            f"Non-finite gradient detected at epoch {epoch_number}, "
                            f"attempted step {attempted_steps}."
                        )
                    if cfg.nonfinite_policy == "skip_step":
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        if scaler is not None:
                            scaler.update()
                        continue

                if gradient_hook is not None:
                    _invoke_legacy_callback(
                        gradient_hook,
                        state_model,
                        policy=cfg.callback_error_policy,
                        name="gradient_hook",
                    )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if hasattr(state_model, "commit_state_updates"):
                    state_model.commit_state_updates()

                batch_size = int(inputs_batch.shape[0])
                total += loss_value * batch_size
                count += batch_size
                successful_steps += 1
                global_step += 1
                for name, metric in metric_functions.items():
                    metric_value = _metric_value(metric, prediction, targets_batch)
                    metric_totals[name] += metric_value * batch_size
                    metric_counts[name] += batch_size
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_times.append(time.perf_counter() - step_start)

            if count == 0:
                raise RuntimeError(
                    f"Epoch {epoch_number} completed without an optimizer step; "
                    "all batches were skipped by the non-finite policy."
                )
            train_loss = total / count
            epoch_time = time.perf_counter() - epoch_start
            train_metrics = {
                f"train_{name}": metric_totals[name] / max(metric_counts[name], 1)
                for name in metric_functions
            }

            val_loss = None
            val_metrics: dict[str, float] = {}
            if val_inputs is not None and val_targets is not None:
                train_model.eval()
                if train_model is not state_model:
                    state_model.eval()
                with torch.no_grad(), _amp_context():
                    prediction_val = (
                        train_model(val_inputs, val_context)
                        if val_context is not None
                        else train_model(val_inputs)
                    )
                    val_loss_tensor = loss_fn(prediction_val, val_targets)
                    if val_loss_tensor.ndim != 0:
                        raise ValueError(
                            "Validation loss must be scalar; received shape "
                            f"{tuple(val_loss_tensor.shape)}."
                        )
                    val_loss = float(val_loss_tensor.item())
                    for name, metric in metric_functions.items():
                        val_metrics[f"val_{name}"] = _metric_value(
                            metric,
                            prediction_val,
                            val_targets,
                        )
                if not math.isfinite(val_loss):
                    dispatcher.emit(
                        "nonfinite_step",
                        epoch=epoch_number,
                        step=global_step,
                        data={"kind": "validation_loss", "policy": cfg.nonfinite_policy},
                    )
                    if cfg.nonfinite_policy == "error":
                        raise FloatingPointError(
                            f"Non-finite validation loss detected at epoch {epoch_number}."
                        )
                dispatcher.emit(
                    "validation_end",
                    epoch=epoch_number,
                    step=global_step,
                    data={"val_loss": val_loss, **val_metrics},
                )

            monitored_metric = val_loss if val_loss is not None else train_loss
            improved = bool(math.isfinite(monitored_metric) and monitored_metric + 1e-12 < best)
            if improved:
                best = monitored_metric
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in state_model.state_dict().items()
                }
                best_epoch = epoch_number
                if cfg.early_stopping:
                    patience = cfg.patience
            elif cfg.early_stopping:
                patience -= 1
            patience_left = patience if cfg.early_stopping else None

            if epoch_callback is not None:
                _invoke_legacy_callback(
                    epoch_callback,
                    epoch_index,
                    float(train_loss),
                    val_loss,
                    improved,
                    patience_left,
                    policy=cfg.callback_error_policy,
                    name="epoch_callback",
                )

            finite_grad_norms = [value for value in grad_norms if math.isfinite(value)]
            grad_norm_mean = (
                float(sum(finite_grad_norms) / len(finite_grad_norms))
                if finite_grad_norms
                else None
            )
            grad_norm_max = float(max(finite_grad_norms)) if finite_grad_norms else None
            step_time_mean = float(sum(step_times) / len(step_times)) if step_times else None
            learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
            epoch_record = {
                "epoch": int(epoch_number),
                "global_step": int(global_step),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
                **train_metrics,
                **val_metrics,
                "learning_rates": learning_rates,
                "epoch_time_s": float(epoch_time),
                "step_time_s_mean": step_time_mean,
                "steps": int(successful_steps),
                "attempted_steps": int(attempted_steps),
                "skipped_nonfinite_steps": int(skipped_steps),
                "samples": int(count),
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_max": grad_norm_max,
                "loss_nonfinite_steps": int(loss_nonfinite),
                "grad_nonfinite_steps": int(grad_nonfinite),
                "improved": bool(improved),
                "patience_left": (int(patience_left) if patience_left is not None else None),
                "best_metric": float(best) if math.isfinite(best) else None,
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
                "train_time_s_total": prior_train_time + float(time.perf_counter() - wall_start),
            }
            history.append(epoch_record)

            if scheduler is not None:
                scheduler.step()

            if checkpoint_callback is not None:
                scaler_state = scaler.state_dict() if scaler is not None else None
                snapshot = {
                    "epoch": int(epoch_number),
                    "global_step": int(global_step),
                    "best_metric": float(best),
                    "best_epoch": best_epoch,
                    "patience_left": patience_left,
                    "best_state": best_state,
                    "history": history,
                    "amp_scaler_state": scaler_state,
                }
                written = checkpoint_callback(snapshot, improved)
                for kind, path in written:
                    dispatcher.emit(
                        "checkpoint",
                        epoch=epoch_number,
                        step=global_step,
                        data={"kind": kind, "path": path},
                    )

            dispatcher.emit(
                "epoch_end",
                epoch=epoch_number,
                step=global_step,
                data=epoch_record,
            )

            if cfg.early_stopping and patience <= 0 and best_state is not None:
                state_model.load_state_dict(best_state)
                dispatcher.emit(
                    "early_stop",
                    epoch=epoch_number,
                    step=global_step,
                    data={"best_metric": best, "best_epoch": best_epoch},
                )
                break

        dispatcher.emit(
            "train_end",
            epoch=int(history[-1]["epoch"]) if history else int(resume.start_epoch),
            step=global_step,
            data={
                "epochs_completed": len(history),
                "best_metric": float(best) if math.isfinite(best) else None,
                "best_epoch": best_epoch,
            },
        )
        return history, best_state
    except Exception as exc:
        dispatcher.emit(
            "failure",
            data={"error_type": type(exc).__name__, "message": str(exc)},
            suppress_callback_errors=True,
        )
        raise


__all__ = [
    "CheckpointCallback",
    "MetricCallable",
    "TrainingLoopConfig",
    "TrainingLoopResumeState",
    "run_training_loop",
]
