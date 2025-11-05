"""Trainer for PSANN-LM with AMP and optional DDP.

Implements a next-token LM objective with AdamW, gradient accumulation,
optional gradient clipping, cosine LR with warmup, AMP (bf16/fp16), and
rank-aware checkpointing/logging. When running under torch.distributed
(`torchrun` or initialized process group) and `ddp` is enabled, wraps
the model in DistributedDataParallel and uses a DistributedSampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext

from ..config import TrainConfig
from ..data.dataset import collate_batch


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


class Trainer:
    """Trainer supporting AMP and optional DDP."""

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
        import math as _math
        model.train()

        # ---- Device selection ----
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- DDP bring-up (if requested/available) ----
        ddp_mode = str(getattr(self.cfg, "ddp", "auto")).lower()
        want_ddp = ddp_mode == "on"
        # Auto-enable DDP if WORLD_SIZE>1 or already initialized
        world_env = int(os.environ.get("WORLD_SIZE", "1"))
        is_dist_env = world_env > 1
        ddp_enabled = want_ddp or (ddp_mode == "auto" and is_dist_env)

        rank = 0
        world_size = 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        ddp = None
        if ddp_enabled and torch.distributed.is_available():
            import torch.distributed as dist
            if device.type == "cuda":
                try:
                    torch.cuda.set_device(local_rank)
                except Exception:
                    pass
                device = torch.device("cuda", local_rank)
            if not dist.is_initialized():
                backend = "nccl" if device.type == "cuda" else "gloo"
                dist.init_process_group(backend=backend)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            model.to(device)
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp = DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
            is_main = rank == 0
        else:
            model.to(device)
            ddp = model  # type: ignore[assignment]
            is_main = True

        # Enable model-level gradient checkpointing if requested and supported
        try:
            if bool(getattr(self.cfg, "grad_checkpoint", False)):
                if hasattr(model, "enable_gradient_checkpointing"):
                    model.enable_gradient_checkpointing(True)  # type: ignore[attr-defined]
                    print("[trainer] Gradient checkpointing: enabled via model.enable_gradient_checkpointing()")
                elif hasattr(model, "gradient_checkpointing"):
                    setattr(model, "gradient_checkpointing", True)
                    print("[trainer] Gradient checkpointing: enabled via model.gradient_checkpointing attr")
        except Exception:
            # non-fatal; proceed without checkpointing
            pass

        # ---- DataLoader (DistributedSampler if DDP) ----
        batch_size = self._compute_batch_size(max_length)
        sampler = None
        if ddp_enabled and torch.distributed.is_available():
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collate_batch,
            pin_memory=(device.type == "cuda"),
        )

        optim = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=float(self.cfg.label_smoothing))

        # LR scheduler (cosine with warmup)
        # Estimate total optimizer steps (with grad accumulation)
        try:
            steps_per_epoch = _math.ceil(len(dataset) / float(batch_size * max(1, world_size if ddp_enabled else 1)))
        except Exception:
            steps_per_epoch = len(dl)
        total_optimizer_steps = int(self.cfg.epochs) * max(1, steps_per_epoch) // max(1, int(self.cfg.grad_accum_steps))
        scheduler = self._build_scheduler(optim, total_optimizer_steps)

        # ---- AMP setup ----
        amp_mode = str(self.cfg.amp).lower()
        use_cuda_amp = device.type == "cuda" and amp_mode in {"bf16", "fp16"}
        amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
        autocast_ctx = (
            torch.cuda.amp.autocast(dtype=amp_dtype) if use_cuda_amp else nullcontext()
        )
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_mode == "fp16"))

        micro = 0
        global_step = 0  # optimizer steps
        accum = max(1, int(self.cfg.grad_accum_steps))
        for epoch in range(self.cfg.epochs):
            self.state.epoch = epoch + 1
            # Set epoch for distributed sampler to reshuffle deterministically
            if sampler is not None and hasattr(sampler, "set_epoch"):
                try:
                    sampler.set_epoch(epoch)
                except Exception:
                    pass
            for batch in dl:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                # Avoid gradient sync on accumulation micro-steps when using DDP
                no_sync_ctx = getattr(ddp, "no_sync", None)
                sync_ctx = nullcontext() if (micro + 1) == accum or no_sync_ctx is None else no_sync_ctx()
                with sync_ctx:
                    with autocast_ctx:
                        logits = ddp(input_ids)  # type: ignore[operator]
                        B, T, V = logits.shape
                        loss = criterion(logits.view(B * T, V), labels.view(B * T))
                        loss = loss / float(accum)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                micro += 1

                # Logging on micro-steps if requested
                if is_main and (global_step + 1) % max(1, self.cfg.log_interval_steps) == 0 and micro == accum:
                    try:
                        ppl = float(_math.exp(loss.detach().float().item() * accum))
                    except Exception:
                        ppl = float('nan')
                    lr = optim.param_groups[0]["lr"]
                    grad_norm = self._grad_global_norm(model)
                    toks = int(B * T * accum)
                    print(
                        f"rank={rank} epoch={epoch+1} step={global_step+1} loss={loss.detach().float().item()*accum:.4f} "
                        f"ppl={ppl:.3f} lr={lr:.6g} grad_norm={grad_norm:.3f} toks/step~{toks}"
                    )

                if micro == accum:
                    # Optional grad clipping
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    if scaler.is_enabled():
                        scaler.step(optim)
                        scaler.update()
                    else:
                        optim.step()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    micro = 0
                    global_step += 1
                    self.state.step = global_step

                    # Periodic checkpointing and optional validation
                    if is_main and global_step % max(1, self.cfg.save_interval_steps) == 0:
                        self._save_checkpoint(model, optim, tag=f"ckpt_step{global_step:06d}")
                        if val_dataset is not None:
                            vloss = self.validate(model, val_dataset)
                            if vloss < self.best_val_loss:
                                self.best_val_loss = float(vloss)
                                self._save_checkpoint(model, optim, tag="best")

        # Final save (main rank only)
        if is_main:
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
