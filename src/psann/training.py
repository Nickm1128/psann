from __future__ import annotations

from dataclasses import dataclass
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
    gradient_hook: Optional[Callable[[torch.nn.Module], None]] = None,
    epoch_callback: Optional[
        Callable[[int, float, Optional[float], bool, Optional[int]], None]
    ] = None,
) -> Tuple[float, Optional[dict]]:
    """Run the shared PSANN training loop."""

    best = float("inf")
    patience = cfg.patience
    best_state: Optional[dict] = None

    for epoch in range(cfg.epochs):
        if cfg.lr_max is not None and cfg.lr_min is not None:
            if cfg.epochs <= 1:
                lr_e = float(cfg.lr_min)
            else:
                frac = float(epoch) / float(max(cfg.epochs - 1, 1))
                lr_e = float(cfg.lr_max) + (float(cfg.lr_min) - float(cfg.lr_max)) * frac
            for group in optimizer.param_groups:
                group["lr"] = lr_e

        if cfg.stateful and cfg.state_reset == "epoch" and hasattr(model, "reset_state"):
            try:
                model.reset_state()
            except Exception:
                pass

        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            if cfg.stateful and cfg.state_reset == "batch" and hasattr(model, "reset_state"):
                try:
                    model.reset_state()
                except Exception:
                    pass
            xb = xb.to(device)
            yb = yb.to(device)
            if noise_std is not None:
                xb = xb + torch.randn_like(xb) * noise_std
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if gradient_hook is not None:
                try:
                    gradient_hook(model)
                except Exception:
                    pass
            optimizer.step()
            if hasattr(model, "commit_state_updates"):
                model.commit_state_updates()
            bs = xb.shape[0]
            total += float(loss.item()) * bs
            count += bs
        train_loss = total / max(count, 1)

        val_loss = None
        if val_inputs is not None and val_targets is not None:
            model.eval()
            with torch.no_grad():
                pred_val = model(val_inputs)
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
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                improved = True
            else:
                patience -= 1
            patience_left = patience

        if epoch_callback is not None:
            try:
                epoch_callback(epoch, float(train_loss), val_loss, improved, patience_left)
            except Exception:
                pass

        if cfg.early_stopping and patience <= 0 and best_state is not None:
            if cfg.verbose:
                print(f"Early stopping at epoch {epoch + 1} (best metric: {best:.6f})")
            model.load_state_dict(best_state)
            break

    return train_loss, best_state
