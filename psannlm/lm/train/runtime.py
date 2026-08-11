"""Checkpoint, optimizer, scheduler, cache, and validation helpers for Trainer."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from psann.utils.hf_cache import cleanup_hf_cache

from ..config import TrainConfig
from ..data.dataset import collate_batch


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0


class TrainerRuntimeMixin:
    """Non-loop runtime responsibilities shared by the LM trainer."""

    cfg: TrainConfig
    state: TrainState
    _last_cache_cleanup: float
    _last_cache_warn: float

    def _save_checkpoint(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        tag: str,
        *,
        data_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        ckpt_dir = self.cfg.checkpoint_dir
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
        except Exception:
            pass

        state_dict: Dict[str, Any]
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
            from torch.distributed.fsdp.api import (  # type: ignore
                FullStateDictConfig,
                StateDictType,
            )

            if isinstance(model, FSDP):  # type: ignore[arg-type]
                cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                    state_dict = model.state_dict()
            else:
                state_dict = model.state_dict()
        except Exception:
            state_dict = model.state_dict()

        payload = {
            "state": {"step": self.state.step, "epoch": self.state.epoch},
            "model": state_dict,
            "optim": optim.state_dict(),
            "cfg": self.cfg.__dict__,
        }
        if data_state:
            payload["data_state"] = dict(data_state)
        path = os.path.join(ckpt_dir, f"{tag}.pt")
        torch.save(payload, path)

    def _compute_batch_size(self, max_length: int) -> int:
        return max(1, int(self.cfg.batch_tokens) // max_length)

    def _build_scheduler(self, optim: torch.optim.Optimizer, total_steps: int) -> LambdaLR:
        warmup = int(max(0, self.cfg.warmup_steps))

        def lr_lambda(step: int) -> float:
            step_number = step + 1
            if warmup > 0 and step_number <= warmup:
                return float(step_number) / float(max(1, warmup))
            if total_steps <= warmup:
                return 1.0

            import math

            progress = float(step_number - warmup) / float(max(1, total_steps - warmup))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optim, lr_lambda)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        opt_name = str(getattr(self.cfg, "optimizer", "adamw")).lower()
        weight_decay = float(self.cfg.weight_decay)
        learning_rate = float(self.cfg.lr)
        betas = tuple(self.cfg.betas) if hasattr(self.cfg, "betas") else (0.9, 0.95)
        epsilon = float(getattr(self.cfg, "eps", 1e-8))
        if opt_name == "adamw8bit":
            try:
                import bitsandbytes as bnb  # type: ignore

                return bnb.optim.AdamW8bit(
                    model.parameters(),
                    lr=learning_rate,
                    betas=betas,
                    eps=epsilon,
                    weight_decay=weight_decay,
                )
            except Exception:
                print("[trainer] bitsandbytes not available; falling back to AdamW.")
        if opt_name == "adafactor":
            try:
                from transformers.optimization import Adafactor  # type: ignore

                return Adafactor(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    relative_step=False,
                    scale_parameter=False,
                )
            except Exception:
                print("[trainer] transformers.Adafactor not available; falling back to AdamW.")

        adamw_kwargs = {
            "lr": learning_rate,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": epsilon,
        }
        if torch.cuda.is_available():
            adamw_kwargs["fused"] = True
        return torch.optim.AdamW(model.parameters(), **adamw_kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _grad_global_norm(model: nn.Module) -> float:
        total = 0.0
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            parameter_norm = float(parameter.grad.data.norm(2).item())
            total += parameter_norm * parameter_norm
        return float(total**0.5)

    def _maybe_cleanup_cache(self) -> None:
        limit_gb = getattr(self.cfg, "hf_cache_limit_gb", None)
        if limit_gb is None or limit_gb <= 0:
            return
        now = time.time()
        if now - self._last_cache_cleanup < 60.0:
            return
        self._last_cache_cleanup = now
        max_bytes = int(limit_gb * (1024**3))
        try:
            freed, total = cleanup_hf_cache(max_bytes)
        except Exception as exc:
            if now - self._last_cache_warn > 300.0:
                print(f"[trainer] HF cache cleanup failed: {exc}")
                self._last_cache_warn = now
            return
        if freed > 0:
            freed_gb = freed / (1024**3)
            total_gb = total / (1024**3)
            print(
                f"[trainer] HF cache cleanup freed {freed_gb:.2f} GB "
                f"(cache now ~{total_gb:.2f} GB)"
            )

    def validate(self, model: nn.Module, dataset: Any) -> float:
        model.eval()
        device = next(model.parameters()).device
        batch_size = max(1, self._compute_batch_size(dataset.cfg.max_length))
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        )
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids)
                batch_count, sequence_length, vocabulary_size = logits.shape
                loss = criterion(
                    logits.view(batch_count * sequence_length, vocabulary_size),
                    labels.view(batch_count * sequence_length),
                )
                token_count = batch_count * sequence_length
                total_loss += float(loss.item()) * token_count
                total_tokens += int(token_count)
        model.train()
        return total_loss / max(1, total_tokens)
