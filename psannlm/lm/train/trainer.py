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
import time
from functools import partial
from collections import deque

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext

from ..config import TrainConfig
from ..data.dataset import collate_batch
from ...utils.hf_cache import cleanup_hf_cache


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
        self._last_cache_cleanup: float = 0.0
        self._last_cache_warn: float = 0.0

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
        if data_state:
            payload["data_state"] = dict(data_state)
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

                return bnb.optim.AdamW8bit(
                    model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd
                )
            except Exception:
                print("[trainer] bitsandbytes not available; falling back to AdamW.")
        if opt_name == "adafactor":
            try:
                from transformers.optimization import Adafactor  # type: ignore

                return Adafactor(
                    model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                    relative_step=False,
                    scale_parameter=False,
                )
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
                f"[trainer] HF cache cleanup freed {freed_gb:.2f} GB (cache now ~{total_gb:.2f} GB)"
            )

    def train(
        self,
        model: nn.Module,
        dataset,
        *,
        max_length: int,
        val_dataset: Optional[Any] = None,
        data_loader: Optional[DataLoader] = None,
        resume_checkpoint: Optional[str] = None,
        teacher_model: Optional[nn.Module] = None,
    ) -> None:
        import math as _math
        from torch.nn import functional as F

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
                    strategy = (
                        ShardingStrategy.FULL_SHARD
                        if fsdp_mode == "full_shard"
                        else ShardingStrategy.FULL_SHARD
                    )
                    wrapped = FSDP(
                        model,
                        auto_wrap_policy=auto_wrap,
                        sharding_strategy=strategy,
                        device_id=(local_rank if device.type == "cuda" else None),
                        use_orig_params=bool(getattr(self.cfg, "fsdp_use_orig_params", True)),
                        cpu_offload=None if not bool(getattr(self.cfg, "fsdp_cpu_offload", False)) else torch.distributed.fsdp.CPUOffload(offload_params=True),  # type: ignore
                    )
                except Exception as e:
                    print(
                        f"[trainer] FSDP requested but not available ({e!s}); falling back to DDP/model-only."
                    )
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
                    print(
                        "[trainer] Gradient checkpointing: enabled via model.enable_gradient_checkpointing()"
                    )
                elif hasattr(model, "gradient_checkpointing"):
                    setattr(model, "gradient_checkpointing", True)
                    print(
                        "[trainer] Gradient checkpointing: enabled via model.gradient_checkpointing attr"
                    )
        except Exception:
            # non-fatal; proceed without checkpointing
            pass

        # ---- Optional distillation teacher ----
        distill_alpha = float(getattr(self.cfg, "distill_alpha", 0.0))
        distill_temperature = float(getattr(self.cfg, "distill_temperature", 1.0))
        distill_enabled = teacher_model is not None and distill_alpha > 0.0
        teacher = teacher_model
        if distill_enabled and teacher is not None:
            try:
                teacher.to(device)
            except Exception:
                pass
            try:
                teacher.eval()
            except Exception:
                pass
            try:
                for p in teacher.parameters():
                    p.requires_grad_(False)
            except Exception:
                pass
            if is_main:
                print(
                    "[trainer] Distillation: enabled "
                    f"(alpha={distill_alpha:g}, T={distill_temperature:g})"
                )

        # ---- DataLoader (DistributedSampler if DDP) ----
        batch_size = self._compute_batch_size(max_length)
        sampler = None
        if data_loader is not None:
            dl = data_loader
        else:
            if (
                not isinstance(dataset, IterableDataset)
                and (ddp_enabled or use_fsdp)
                and torch.distributed.is_available()
            ):
                from torch.utils.data.distributed import DistributedSampler

                sampler = DistributedSampler(
                    dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
                )
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
        # Estimate total optimizer steps (scheduler steps).
        # Note: In this trainer, `global_step` increments once per *optimizer* step
        # (after `grad_accum_steps` micro-batches), and `scheduler.step()` is called
        # at the same cadence.
        steps_per_epoch_cfg = getattr(self.cfg, "steps_per_epoch", None)
        if steps_per_epoch_cfg is not None:
            steps_per_epoch = max(1, int(steps_per_epoch_cfg))
        else:
            try:
                steps_per_epoch = _math.ceil(
                    len(dataset)
                    / float(batch_size * max(1, world_size if (ddp_enabled or use_fsdp) else 1))
                )
            except Exception:
                try:
                    steps_per_epoch = len(dl)
                except Exception:
                    steps_per_epoch = 1000  # conservative default when unknown
        accum = max(1, int(self.cfg.grad_accum_steps))
        if steps_per_epoch_cfg is not None:
            # When provided explicitly (e.g., streaming runs), interpret steps_per_epoch
            # as optimizer steps (matching checkpoint/log `step=` counters).
            total_optimizer_steps = int(self.cfg.epochs) * max(1, steps_per_epoch)
        else:
            # When derived from dataset/loader length, `steps_per_epoch` is in dataloader
            # batches; convert to optimizer steps under gradient accumulation.
            total_optimizer_steps = int(self.cfg.epochs) * max(
                1, int(_math.ceil(float(steps_per_epoch) / float(accum)))
            )
        scheduler = self._build_scheduler(optim, total_optimizer_steps)
        steps_per_epoch_optimizer = (
            int(steps_per_epoch)
            if steps_per_epoch_cfg is not None
            else max(1, int(_math.ceil(float(steps_per_epoch) / float(accum))))
        )

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

        # Optional resume: load model/optimizer state and position counters
        start_epoch = 0
        start_step = 0
        if resume_checkpoint:
            try:
                payload = torch.load(resume_checkpoint, map_location="cpu")
                state_dict = payload.get("model", payload)
                wrapped.load_state_dict(state_dict)
                if isinstance(payload, dict) and "optim" in payload:
                    try:
                        optim.load_state_dict(payload["optim"])
                    except Exception as exc:  # pragma: no cover - best effort
                        print(f"[trainer] Warning: could not load optimizer state: {exc}")
                ckpt_state = payload.get("state", {}) if isinstance(payload, dict) else {}
                start_step = int(ckpt_state.get("step", 0) or 0)
                start_epoch = max(0, int(ckpt_state.get("epoch", 1) or 1) - 1)
                self.state.step = start_step
                self.state.epoch = start_epoch
                try:
                    scheduler.last_epoch = max(start_step - 1, -1)
                except Exception:
                    pass
                print(
                    f"[trainer] Resumed from {resume_checkpoint} "
                    f"(step={start_step}, epoch={start_epoch + 1})"
                )
            except Exception as exc:
                print(f"[trainer] Warning: failed to resume from {resume_checkpoint}: {exc}")

        # Optional torch.compile. Keep it opt-in and avoid DDP/FSDP for now.
        # Note: compile happens after resume loading so we don't have to deal with
        # load_state_dict/optimizer edge-cases on compiled wrappers.
        compiled_enabled = False
        if bool(getattr(self.cfg, "torch_compile", False)):
            if ddp_enabled or use_fsdp:
                if is_main:
                    print("[trainer] torch.compile requested but DDP/FSDP is enabled; skipping.")
            elif not hasattr(torch, "compile"):
                if is_main:
                    print("[trainer] torch.compile requested but torch.compile is unavailable; skipping.")
            else:
                mode = str(getattr(self.cfg, "torch_compile_mode", "default") or "default").strip()
                fullgraph = bool(getattr(self.cfg, "torch_compile_fullgraph", False))
                dynamic = bool(getattr(self.cfg, "torch_compile_dynamic", False))
                try:
                    t0_compile = time.time()
                    wrapped = torch.compile(
                        wrapped, mode=mode, fullgraph=fullgraph, dynamic=dynamic
                    )
                    if is_main:
                        dt = time.time() - t0_compile
                        print(
                            f"[trainer] torch.compile enabled (mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}) "
                            f"in {dt:.1f}s"
                        )
                    compiled_enabled = True
                except Exception as exc:
                    if is_main:
                        print(f"[trainer] torch.compile failed ({exc}); continuing without compile.")
                    compiled_enabled = False

        # Optional: free unused cached memory after init/compile so other processes
        # have VRAM headroom. This trades some performance for lower reserved VRAM.
        if (
            device.type == "cuda"
            and bool(getattr(self.cfg, "cuda_empty_cache_after_init", False))
            and hasattr(torch.cuda, "empty_cache")
        ):
            try:
                torch.cuda.empty_cache()
                if is_main:
                    print("[trainer] torch.cuda.empty_cache() after init/compile")
            except Exception:
                pass

        micro = 0
        global_step = start_step  # optimizer steps
        log_interval = max(1, int(self.cfg.log_interval_steps))
        ppl_window = deque(maxlen=log_interval)
        eval_interval = max(0, int(getattr(self.cfg, "eval_interval_steps", 0) or 0))
        eval_max_batches = max(0, int(getattr(self.cfg, "eval_max_batches", 0) or 0))
        empty_cache_interval = max(
            0, int(getattr(self.cfg, "cuda_empty_cache_interval_steps", 0) or 0)
        )
        data_state = {
            "seq_len": int(max_length),
            "micro_batch": int(batch_size),
            "grad_accum_steps": int(accum),
            "sequences_per_step": int(batch_size * accum),
            "tokens_per_step": int(batch_size * max_length * accum),
        }
        last_eval_step = -1

        def _eval_once(step: int) -> float:
            nonlocal last_eval_step
            if val_dataset is None or step == last_eval_step:
                return float("nan")
            last_eval_step = step
            limit_batches = eval_max_batches if eval_max_batches > 0 else 128
            vdl = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_batch,
                pin_memory=(device.type == "cuda"),
                num_workers=0,
            )
            wrapped.eval()
            criterion_eval = nn.CrossEntropyLoss()
            vloss_sum = 0.0
            vtoks = 0
            t0 = time.time()
            with torch.no_grad():
                for i, vbatch in enumerate(vdl):
                    if i >= limit_batches:
                        break
                    vinput = vbatch["input_ids"].to(device)
                    vlabels = vbatch["labels"].to(device)
                    with autocast_ctx:
                        vlogits = wrapped(vinput)  # type: ignore[operator]
                        bsz, seqlen, vocab = vlogits.shape
                        vloss = criterion_eval(
                            vlogits.view(bsz * seqlen, vocab), vlabels.view(bsz * seqlen)
                        )
                    vloss_sum += float(vloss.item()) * float(bsz * seqlen)
                    vtoks += int(bsz * seqlen)
            wrapped.train()
            avg_loss = vloss_sum / max(1, vtoks)
            try:
                vppl = float(_math.exp(avg_loss))
            except Exception:
                vppl = float("nan")
            dt = time.time() - t0
            print(
                f"[eval] step={step} tokens={vtoks} loss={avg_loss:.4f} ppl={vppl:.3f} dt={dt:.1f}s"
            )
            return float(avg_loss)

        micro_total_loss_sum = 0.0
        micro_hard_loss_sum = 0.0
        micro_kd_loss_sum = 0.0
        micro_loss_count = 0
        for epoch in range(start_epoch, self.cfg.epochs):
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
                sync_ctx = (
                    nullcontext() if (micro + 1) == accum or no_sync_ctx is None else no_sync_ctx()
                )
                for attempt in range(2):
                    try:
                        with sync_ctx:
                            with autocast_ctx:
                                logits = wrapped(input_ids)  # type: ignore[operator]
                                B, T, V = logits.shape
                                hard_loss = criterion(
                                    logits.view(B * T, V), labels.view(B * T)
                                )
                                kd_loss = None
                                total_loss = hard_loss
                                if distill_enabled and teacher is not None:
                                    with torch.no_grad():
                                        teacher_logits = teacher(input_ids)
                                    if teacher_logits.shape != logits.shape:
                                        raise ValueError(
                                            "Teacher/student logits shape mismatch: "
                                            f"teacher={tuple(teacher_logits.shape)} "
                                            f"student={tuple(logits.shape)}"
                                        )
                                    temp = float(distill_temperature)
                                    if temp <= 0:
                                        raise ValueError(
                                            f"distill_temperature must be > 0; got {temp}"
                                        )
                                    student_log_probs = F.log_softmax(
                                        logits / temp, dim=-1
                                    )
                                    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
                                    kd_loss = (
                                        F.kl_div(
                                            student_log_probs.view(B * T, V),
                                            teacher_probs.view(B * T, V),
                                            reduction="batchmean",
                                        )
                                        * (temp * temp)
                                    )
                                    alpha = float(distill_alpha)
                                    total_loss = (1.0 - alpha) * hard_loss + alpha * kd_loss
                                loss = total_loss / float(accum)
                            if scaler.is_enabled():
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        # Track per-micro loss for rolling averages (unscale back to per-step units).
                        try:
                            micro_total_loss_sum += float(
                                loss.detach().float().item()
                            ) * float(accum)
                            micro_hard_loss_sum += float(hard_loss.detach().float().item())
                            if kd_loss is not None:
                                micro_kd_loss_sum += float(kd_loss.detach().float().item())
                            micro_loss_count += 1
                        except Exception:
                            pass
                        break
                    except BaseException as exc:
                        msg = str(exc).lower()
                        is_oom = ("out of memory" in msg) or isinstance(
                            exc, getattr(torch, "OutOfMemoryError", RuntimeError)
                        )
                        if not (device.type == "cuda" and is_oom):
                            raise

                        if is_main:
                            print(
                                f"[trainer] CUDA OOM during backward (step={global_step+1} micro={micro+1}/{accum})."
                            )

                        # Best-effort cleanup.
                        try:
                            optim.zero_grad(set_to_none=True)
                        except Exception:
                            pass
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                        # If torch.compile was enabled, fall back to eager and retry once.
                        if compiled_enabled and attempt == 0:
                            if is_main:
                                print("[trainer] Disabling torch.compile and retrying in eager mode.")
                            try:
                                import torch._dynamo as dynamo  # type: ignore

                                dynamo.reset()
                            except Exception:
                                pass
                            wrapped = model
                            compiled_enabled = False
                            micro = 0
                            micro_total_loss_sum = 0.0
                            micro_hard_loss_sum = 0.0
                            micro_kd_loss_sum = 0.0
                            micro_loss_count = 0
                            continue

                        raise
                micro += 1

                # Logging on micro-steps if requested
                if micro == accum:
                    # Compute end-of-step metrics (pre-optimizer step) so logging uses a consistent
                    # per-optimizer-step loss/ppl even under gradient accumulation.
                    step_loss = float("nan")
                    step_total_loss = float("nan")
                    step_kd_loss = float("nan")
                    if micro_loss_count > 0:
                        step_loss = micro_hard_loss_sum / float(micro_loss_count)
                        step_total_loss = micro_total_loss_sum / float(micro_loss_count)
                        if distill_enabled:
                            step_kd_loss = micro_kd_loss_sum / float(micro_loss_count)
                    try:
                        step_ppl = float(_math.exp(step_loss))
                    except Exception:
                        step_ppl = float("nan")
                    ppl_window.append(step_ppl)
                    try:
                        ppl_avg = float(sum(ppl_window) / float(len(ppl_window))) if ppl_window else float("nan")
                    except Exception:
                        ppl_avg = float("nan")
                    micro_total_loss_sum = 0.0
                    micro_hard_loss_sum = 0.0
                    micro_kd_loss_sum = 0.0
                    micro_loss_count = 0

                    if is_main and (global_step + 1) % log_interval == 0:
                        lr = optim.param_groups[0]["lr"]
                        grad_norm = self._grad_global_norm(model)
                        toks = int(B * T * accum)
                        msg = (
                            f"rank={rank} epoch={epoch+1} step={global_step+1} "
                            f"loss={step_loss:.4f} ppl={step_ppl:.3f} lr={lr:.6g} "
                            f"grad_norm={grad_norm:.3f} toks/step~{toks} ppl_avg{log_interval}={ppl_avg:.3f}"
                        )
                        if distill_enabled:
                            msg += (
                                f" loss_total={step_total_loss:.4f} kd_loss={step_kd_loss:.4f}"
                            )
                        print(msg)
                        if bool(getattr(self.cfg, "log_gpu_mem", False)) and device.type == "cuda":
                            try:
                                alloc = torch.cuda.memory_allocated(device) / float(1024**3)
                                reserved = torch.cuda.memory_reserved(device) / float(1024**3)
                                max_alloc = torch.cuda.max_memory_allocated(device) / float(1024**3)
                                print(
                                    f"[gpu-mem] rank={rank} step={global_step+1} "
                                    f"alloc_gb={alloc:.3f} reserved_gb={reserved:.3f} max_alloc_gb={max_alloc:.3f}"
                                )
                            except Exception:
                                pass

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
                    self._maybe_cleanup_cache()

                    if empty_cache_interval and device.type == "cuda":
                        if global_step % max(1, empty_cache_interval) == 0:
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass

                    if eval_interval and is_main and val_dataset is not None:
                        if global_step % max(1, eval_interval) == 0:
                            _eval_once(global_step)

                    # Periodic checkpointing and optional validation
                    if is_main and global_step % max(1, self.cfg.save_interval_steps) == 0:
                        # Save via wrapper to support FSDP full state dict
                        self._save_checkpoint(
                            wrapped,
                            optim,
                            tag=f"ckpt_step{global_step:06d}",
                            data_state=data_state,
                        )
                        if val_dataset is not None:
                            vloss = _eval_once(global_step)
                            if vloss < self.best_val_loss:
                                self.best_val_loss = float(vloss)
                                self._save_checkpoint(
                                    wrapped, optim, tag="best", data_state=data_state
                                )

                    if steps_per_epoch_cfg is not None:
                        if global_step >= steps_per_epoch_optimizer:
                            break
                    elif steps_this_epoch >= steps_per_epoch_optimizer:
                        break
            if steps_per_epoch_cfg is not None:
                if global_step >= steps_per_epoch_optimizer:
                    break
            elif steps_this_epoch >= steps_per_epoch_optimizer:
                continue

        # If we were asked to run a fixed number of optimizer steps (streaming-style),
        # but the dataloader ended early, warn loudly. This typically means the
        # underlying dataset (after filtering) was smaller than expected.
        if steps_per_epoch_cfg is not None and global_step < steps_per_epoch_optimizer:
            if is_main:
                print(
                    "[trainer] WARNING: DataLoader exhausted early "
                    f"(step={global_step} < max_steps={steps_per_epoch_optimizer}). "
                    "This usually means your dataset/config/filtering ran out of samples. "
                    "Relax filters or use a larger dataset/config to reach the intended token budget."
                )

        # Final save (main rank only)
        if is_main:
            self._save_checkpoint(wrapped, optim, tag="final", data_state=data_state)

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
