"""Trainer for PSANN-LM (minimal CPU training loop).

This initial implementation supports a simple next-token LM objective
on CPU, with AdamW optimizer, optional gradient clipping, and basic
logging. Distributed/AMP/checkpointing can be added incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from ..config import TrainConfig
from ..data.dataset import collate_batch


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


class Trainer:
    """Minimal trainer with a CPU-only training loop."""

    def __init__(self, cfg: Optional[TrainConfig] = None) -> None:
        self.state = TrainState()
        self.cfg = cfg or TrainConfig()
        self.best_val_loss: float = float("inf")

    def _save_checkpoint(self, model: nn.Module, optim: torch.optim.Optimizer, tag: str) -> None:
        ckpt_dir = self.cfg.checkpoint_dir
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
        except Exception:
            pass
        payload = {
            "state": {"step": self.state.step, "epoch": self.state.epoch},
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(payload, path)

    def _compute_batch_size(self, max_length: int) -> int:
        btoks = int(self.cfg.batch_tokens)
        return max(1, btoks // max_length)

    def _build_scheduler(self, optim: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
        warmup = int(max(0, self.cfg.warmup_steps))

        def lr_lambda(step: int) -> float:
            # step is 0-indexed per PyTorch; use step+1 for human-friendly behavior
            s = step + 1
            if warmup > 0 and s <= warmup:
                return float(s) / float(max(1, warmup))
            if total_steps <= warmup:
                return 1.0
            # Cosine decay from 1.0 to 0.0 after warmup
            import math as _math

            progress = float(s - warmup) / float(max(1, total_steps - warmup))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + _math.cos(_math.pi * progress))

        return LambdaLR(optim, lr_lambda)

    @staticmethod
    def _grad_global_norm(model: nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            param_norm = float(p.grad.data.norm(2).item())
            total += param_norm * param_norm
        return float(total ** 0.5)

    def train(
        self,
        model: nn.Module,
        dataset,
        *,
        max_length: int,
        val_dataset: Optional[Any] = None,
    ) -> None:
        model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        batch_size = self._compute_batch_size(max_length)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

        optim = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=float(self.cfg.label_smoothing))

        # LR scheduler (cosine with warmup)
        # Estimate total optimizer steps (with grad accumulation)
        try:
            import math as _math

            steps_per_epoch = _math.ceil(len(dataset) / float(batch_size))
        except Exception:
            steps_per_epoch = len(dl)
        total_optimizer_steps = int(self.cfg.epochs) * max(1, steps_per_epoch) // max(1, int(self.cfg.grad_accum_steps))
        scheduler = self._build_scheduler(optim, total_optimizer_steps)

        micro = 0
        global_step = 0  # optimizer steps
        accum = max(1, int(self.cfg.grad_accum_steps))
        for epoch in range(self.cfg.epochs):
            self.state.epoch = epoch + 1
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                logits = model(input_ids)  # (B,T,V)
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), labels.view(B * T))
                loss = loss / float(accum)

                # Accumulate
                loss.backward()
                micro += 1

                # Logging on micro-steps if requested
                if (global_step + 1) % max(1, self.cfg.log_interval_steps) == 0 and micro == accum:
                    try:
                        import math as _math
                        ppl = float(_math.exp(loss.item() * accum))  # undo scaling for readability
                    except Exception:
                        ppl = float('nan')
                    lr = optim.param_groups[0]["lr"]
                    grad_norm = self._grad_global_norm(model)
                    toks = int(B * T * accum)
                    print(
                        f"epoch={epoch+1} step={global_step+1} loss={loss.item()*accum:.4f} "
                        f"ppl={ppl:.3f} lr={lr:.6g} grad_norm={grad_norm:.3f} toks/step~{toks}"
                    )

                if micro == accum:
                    # Optional grad clipping
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    micro = 0
                    global_step += 1
                    self.state.step = global_step

                    # Periodic checkpointing and optional validation
                    if global_step % max(1, self.cfg.save_interval_steps) == 0:
                        self._save_checkpoint(model, optim, tag=f"ckpt_step{global_step:06d}")
                        if val_dataset is not None:
                            vloss = self.validate(model, val_dataset)
                            if vloss < self.best_val_loss:
                                self.best_val_loss = float(vloss)
                                self._save_checkpoint(model, optim, tag="best")

        # Final save
        self._save_checkpoint(model, optim, tag="final")

    def validate(self, model: nn.Module, dataset) -> float:
        model.eval()
        device = next(model.parameters()).device
        dl = DataLoader(dataset, batch_size=max(1, self._compute_batch_size(dataset.cfg.max_length)), shuffle=False, collate_fn=collate_batch)  # type: ignore[attr-defined]
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                B, T, V = logits.shape
                loss = criterion(logits.view(B * T, V), labels.view(B * T))
                total_loss += float(loss.item()) * (B * T)
                total_tokens += int(B * T)
        model.train()
        return total_loss / max(1, total_tokens)
