from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader


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


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        return None


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
) -> Tuple[list[dict], Optional[dict]]:
    """Run the shared PSANN training loop."""

    train_model = model
    state_model = model
    if (
        cfg.compile_model
        and device.type == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch, "compile")
    ):
        compile_fn = getattr(torch, "compile")
        compile_kwargs: dict[str, object] = {}
        try:
            sig = inspect.signature(compile_fn)
            if "backend" in sig.parameters:
                compile_kwargs["backend"] = str(cfg.compile_backend)
            if "mode" in sig.parameters:
                compile_kwargs["mode"] = str(cfg.compile_mode)
            if "fullgraph" in sig.parameters:
                compile_kwargs["fullgraph"] = bool(cfg.compile_fullgraph)
            if "dynamic" in sig.parameters:
                compile_kwargs["dynamic"] = bool(cfg.compile_dynamic)
        except Exception:
            compile_kwargs = {
                "backend": str(cfg.compile_backend),
                "mode": str(cfg.compile_mode),
                "fullgraph": bool(cfg.compile_fullgraph),
                "dynamic": bool(cfg.compile_dynamic),
            }
        try:
            train_model = compile_fn(model, **compile_kwargs)
            state_model = getattr(train_model, "_orig_mod", model)
        except Exception as exc:
            train_model = model
            state_model = model
            if cfg.verbose:
                print(f"[warn] torch.compile failed; falling back to eager: {exc}")

    use_amp = bool(cfg.use_amp) and device.type == "cuda" and torch.cuda.is_available()
    amp_dtype = cfg.amp_dtype if cfg.amp_dtype is not None else torch.bfloat16

    amp_ctx: object
    if use_amp:
        try:
            amp_ctx = torch.autocast(device.type, dtype=amp_dtype)  # type: ignore[attr-defined]
        except Exception:
            amp_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)  # type: ignore[attr-defined]
    else:
        amp_ctx = _NullContext()

    scaler: Optional[object] = None
    if use_amp and amp_dtype == torch.float16:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    best = float("inf")
    patience = cfg.patience
    best_state: Optional[dict] = None
    best_epoch: Optional[int] = None
    history: list[dict] = []
    total_start = time.perf_counter()

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

    for epoch in range(cfg.epochs):
        if cfg.lr_max is not None and cfg.lr_min is not None:
            if cfg.epochs <= 1:
                lr_e = float(cfg.lr_min)
            else:
                frac = float(epoch) / float(max(cfg.epochs - 1, 1))
                lr_e = float(cfg.lr_max) + (float(cfg.lr_min) - float(cfg.lr_max)) * frac
            for group in optimizer.param_groups:
                group["lr"] = lr_e

        if cfg.stateful and cfg.state_reset == "epoch" and hasattr(state_model, "reset_state"):
            try:
                state_model.reset_state()
            except Exception:
                pass

        train_model.train()
        if train_model is not state_model:
            state_model.train()
        total = 0.0
        count = 0
        steps = 0
        step_times: list[float] = []
        grad_norms: list[float] = []
        loss_nonfinite = 0
        grad_nonfinite = 0
        epoch_start = time.perf_counter()
        for batch in train_loader:
            context_b: Optional[torch.Tensor] = None
            if isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    xb, context_b, yb = batch
                elif len(batch) == 2:
                    xb, yb = batch
                else:
                    raise ValueError("Unexpected batch tuple length encountered during training.")
            else:
                raise ValueError("Training batches must be tuple/list tensors.")
            if cfg.stateful and cfg.state_reset == "batch" and hasattr(state_model, "reset_state"):
                try:
                    state_model.reset_state()
                except Exception:
                    pass
            xb = xb.to(device)
            if context_b is not None:
                context_b = context_b.to(device)
            yb = yb.to(device)
            if noise_std is not None:
                xb = xb + torch.randn_like(xb) * noise_std
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            with amp_ctx:
                pred = (
                    train_model(xb, context_b) if context_b is not None else train_model(xb)
                )
                loss = loss_fn(pred, yb)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                loss_nonfinite += 1
            if scaler is not None:
                scaler.scale(loss).backward()
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                grad_norm = _grad_norm(state_model)
                if grad_norm is not None:
                    grad_norms.append(grad_norm)
                    if not math.isfinite(grad_norm):
                        grad_nonfinite += 1
                if gradient_hook is not None:
                    try:
                        gradient_hook(state_model)
                    except Exception:
                        pass
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = _grad_norm(state_model)
                if grad_norm is not None:
                    grad_norms.append(grad_norm)
                    if not math.isfinite(grad_norm):
                        grad_nonfinite += 1
                if gradient_hook is not None:
                    try:
                        gradient_hook(state_model)
                    except Exception:
                        pass
                optimizer.step()
            if hasattr(state_model, "commit_state_updates"):
                state_model.commit_state_updates()
            bs = xb.shape[0]
            total += loss_value * bs
            count += bs
            steps += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)
        train_loss = total / max(count, 1)
        epoch_time = time.perf_counter() - epoch_start

        val_loss = None
        if val_inputs is not None and val_targets is not None:
            train_model.eval()
            if train_model is not state_model:
                state_model.eval()
            with torch.no_grad():
                with amp_ctx:
                    pred_val = (
                        train_model(val_inputs, val_context)
                        if val_context is not None
                        else train_model(val_inputs)
                    )
                    val_loss = float(loss_fn(pred_val, val_targets).item())

        metric = val_loss if val_loss is not None else train_loss
        if cfg.verbose:
            if val_loss is not None:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{cfg.epochs} - loss: {train_loss:.6f}")

        improved = False
        patience_left: Optional[int] = None
        if cfg.early_stopping:
            if metric + 1e-12 < best:
                best = metric
                patience = cfg.patience
                best_state = {
                    k: v.detach().cpu().clone() for k, v in state_model.state_dict().items()
                }
                improved = True
                best_epoch = epoch + 1
            else:
                patience -= 1
            patience_left = patience

        if epoch_callback is not None:
            try:
                epoch_callback(epoch, float(train_loss), val_loss, improved, patience_left)
            except Exception:
                pass

        finite_grad_norms = [g for g in grad_norms if math.isfinite(g)]
        grad_norm_mean = (
            float(sum(finite_grad_norms) / len(finite_grad_norms)) if finite_grad_norms else None
        )
        grad_norm_max = float(max(finite_grad_norms)) if finite_grad_norms else None
        step_time_mean = float(sum(step_times) / len(step_times)) if step_times else None

        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss) if val_loss is not None else None,
                "epoch_time_s": float(epoch_time),
                "step_time_s_mean": step_time_mean,
                "steps": int(steps),
                "samples": int(count),
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_max": grad_norm_max,
                "loss_nonfinite_steps": int(loss_nonfinite),
                "grad_nonfinite_steps": int(grad_nonfinite),
                "improved": bool(improved),
                "patience_left": int(patience_left) if patience_left is not None else None,
                "best_metric": float(best) if math.isfinite(best) else None,
                "best_epoch": int(best_epoch) if best_epoch is not None else None,
            }
        )

        if cfg.early_stopping and patience <= 0 and best_state is not None:
            if cfg.verbose:
                print(f"Early stopping at epoch {epoch + 1} (best metric: {best:.6f})")
            state_model.load_state_dict(best_state)
            break

    total_time_s = time.perf_counter() - total_start
    if history:
        history[-1]["train_time_s_total"] = float(total_time_s)

    return history, best_state
