"""Trainer for PSANN-LM with AMP, DDP/FSDP, and optional 8-bit optimizers.

Implements a next-token LM objective with AdamW (or optional 8-bit AdamW /
Adafactor), gradient accumulation, optional gradient clipping, cosine LR
with warmup, AMP (bf16/fp16), and rank-aware checkpointing/logging.

Distributed training:
  - DDP: Enabled when `ddp` is on/auto and world size > 1.
  - FSDP: Enabled when `fsdp` in TrainConfig is not 'off'. FSDP takes
    precedence over DDP when requested.

Data handling:
  - Supports `Dataset` and `IterableDataset`. For iterable datasets,
    `DistributedSampler` is not used, and scheduler falls back to
    `steps_per_epoch` (or a conservative default) if length is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Dict
import os
from functools import partial

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
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
        # Handle FSDP full-state extraction if applicable
        state_dict: Dict[str, Any]
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
            from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig  # type: ignore
            if isinstance(model, FSDP):  # type: ignore[arg-type]
                cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                    state_dict = model.state_dict()
            else:
                state_dict = model.state_dict()
        except Exception:
            # Fallback: best-effort local state
            state_dict = model.state_dict()
        payload = {
            "state": {"step": self.state.step, "epoch": self.state.epoch},
            "model": state_dict,
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

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        opt_name = str(getattr(self.cfg, "optimizer", "adamw")).lower()
        wd = float(self.cfg.weight_decay)
        lr = float(self.cfg.lr)
        betas = tuple(self.cfg.betas) if hasattr(self.cfg, "betas") else (0.9, 0.95)
        eps = float(getattr(self.cfg, "eps", 1e-8))
        if opt_name == "adamw8bit":
            try:
                import bitsandbytes as bnb  # type: ignore

                return bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)
            except Exception:
                print("[trainer] bitsandbytes not available; falling back to AdamW.")
        if opt_name == "adafactor":
            try:
                from transformers.optimization import Adafactor  # type: ignore

                return Adafactor(model.parameters(), lr=lr, weight_decay=wd, relative_step=False, scale_parameter=False)
            except Exception:
                print("[trainer] transformers.Adagactor not available; falling back to AdamW.")
        adamw_kwargs = dict(lr=lr, weight_decay=wd, betas=betas, eps=eps)
        if torch.cuda.is_available():
            adamw_kwargs["fused"] = True  # type: ignore[assignment]
        return torch.optim.AdamW(model.parameters(), **adamw_kwargs)

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

        # ---- Distributed bring-up (DDP/FSDP) ----
        ddp_mode = str(getattr(self.cfg, "ddp", "auto")).lower()
        fsdp_mode = str(getattr(self.cfg, "fsdp", "off")).lower()
        want_ddp = ddp_mode == "on"
        world_env = int(os.environ.get("WORLD_SIZE", "1"))
        is_dist_env = world_env > 1
        use_fsdp = fsdp_mode != "off"
        ddp_enabled = (want_ddp or (ddp_mode == "auto" and is_dist_env)) and not use_fsdp

        rank = 0
        world_size = 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        wrapped = model
        is_main = True

        if (ddp_enabled or use_fsdp) and torch.distributed.is_available():
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
            if use_fsdp:
                try:
                    from torch.distributed.fsdp import (
                        FullyShardedDataParallel as FSDP,
                        ShardingStrategy,
                    )  # type: ignore
                    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy  # type: ignore
                    auto_wrap = None
                    if str(getattr(self.cfg, "fsdp_auto_wrap_policy", "size")).lower() == "size":
                        min_params = int(getattr(self.cfg, "fsdp_min_params", 1_000_000))
                        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=min_params)
                    strategy = ShardingStrategy.FULL_SHARD if fsdp_mode == "full_shard" else ShardingStrategy.FULL_SHARD
                    wrapped = FSDP(
                        model,
                        auto_wrap_policy=auto_wrap,
                        sharding_strategy=strategy,
                        device_id=(local_rank if device.type == "cuda" else None),
                        use_orig_params=bool(getattr(self.cfg, "fsdp_use_orig_params", True)),
                        cpu_offload=None if not bool(getattr(self.cfg, "fsdp_cpu_offload", False)) else torch.distributed.fsdp.CPUOffload(offload_params=True),  # type: ignore
                    )
                except Exception as e:
                    print(f"[trainer] FSDP requested but not available ({e!s}); falling back to DDP/model-only.")
                    from torch.nn.parallel import DistributedDataParallel as DDP
                    wrapped = DDP(
                        model,
                        device_ids=[local_rank] if device.type == "cuda" else None,
                        output_device=local_rank if device.type == "cuda" else None,
                        find_unused_parameters=False,
                    )
            elif ddp_enabled:
                from torch.nn.parallel import DistributedDataParallel as DDP
                wrapped = DDP(
                    model,
                    device_ids=[local_rank] if device.type == "cuda" else None,
                    output_device=local_rank if device.type == "cuda" else None,
                    find_unused_parameters=False,
                )
            is_main = rank == 0
        else:
            model.to(device)
            wrapped = model
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
        if not isinstance(dataset, IterableDataset) and (ddp_enabled or use_fsdp) and torch.distributed.is_available():
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if isinstance(dataset, IterableDataset) else (sampler is None)),
            sampler=sampler,
            collate_fn=collate_batch,
            pin_memory=(device.type == "cuda"),
            num_workers=int(getattr(self.cfg, "dataloader_num_workers", 0)),
            prefetch_factor=(
                int(getattr(self.cfg, "dataloader_prefetch_factor", 2))
                if int(getattr(self.cfg, "dataloader_num_workers", 0)) > 0
                else None
            ),
            persistent_workers=(
                bool(getattr(self.cfg, "dataloader_persistent_workers", False))
                if int(getattr(self.cfg, "dataloader_num_workers", 0)) > 0
                else False
            ),
        )

        optim = self._build_optimizer(model)
        criterion = nn.CrossEntropyLoss(label_smoothing=float(self.cfg.label_smoothing))

        # LR scheduler (cosine with warmup)
        # Estimate total optimizer steps (with grad accumulation)
        steps_per_epoch_cfg = getattr(self.cfg, "steps_per_epoch", None)
        if steps_per_epoch_cfg is not None:
            steps_per_epoch = max(1, int(steps_per_epoch_cfg))
        else:
            try:
                steps_per_epoch = _math.ceil(len(dataset) / float(batch_size * max(1, world_size if (ddp_enabled or use_fsdp) else 1)))
            except Exception:
                try:
                    steps_per_epoch = len(dl)
                except Exception:
                    steps_per_epoch = 1000  # conservative default when unknown
        total_optimizer_steps = int(self.cfg.epochs) * max(1, steps_per_epoch) // max(1, int(self.cfg.grad_accum_steps))
        scheduler = self._build_scheduler(optim, total_optimizer_steps)

        # ---- AMP setup ----
        amp_mode = str(self.cfg.amp).lower()
        use_cuda_amp = device.type == "cuda" and amp_mode in {"bf16", "fp16"}
        amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=amp_dtype) if use_cuda_amp else nullcontext()
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(device.type == "cuda" and amp_mode == "fp16")
        )

        micro = 0
        global_step = 0  # optimizer steps
        accum = max(1, int(self.cfg.grad_accum_steps))
        for epoch in range(self.cfg.epochs):
            self.state.epoch = epoch + 1
            steps_this_epoch = 0
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
                no_sync_ctx = getattr(wrapped, "no_sync", None)
                sync_ctx = nullcontext() if (micro + 1) == accum or no_sync_ctx is None else no_sync_ctx()
                with sync_ctx:
                    with autocast_ctx:
                        logits = wrapped(input_ids)  # type: ignore[operator]
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
                    steps_this_epoch += 1

                    # Periodic checkpointing and optional validation
                    if is_main and global_step % max(1, self.cfg.save_interval_steps) == 0:
                        # Save via wrapper to support FSDP full state dict
                        self._save_checkpoint(wrapped, optim, tag=f"ckpt_step{global_step:06d}")
                        if val_dataset is not None:
                            vloss = self.validate(model, val_dataset)
                            if vloss < self.best_val_loss:
                                self.best_val_loss = float(vloss)
                                self._save_checkpoint(wrapped, optim, tag="best")

                    if steps_this_epoch >= max(1, steps_per_epoch):
                        break
            if steps_this_epoch >= max(1, steps_per_epoch):
                continue

        # Final save (main rank only)
        if is_main:
            self._save_checkpoint(wrapped, optim, tag="final")

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
