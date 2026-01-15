#!/usr/bin/env python
"""Two-phase training pipeline for a ~300M PSANN-LM chat model.

Phase 1: Pre-train on WikiText-103-raw-v1.
Phase 2: Instruction-tune on OpenAssistant/oasst1 (200k-400k prompt/response pairs).

The script prints frequent progress updates, periodically saves checkpoints,
and writes the final psannLM checkpoint + tokenizer for downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.optim import AdamW

from datasets import load_dataset
from transformers import AutoTokenizer

from psannlm.lm import psannLM
from psannlm.lm.models.sine import SineConfig
from psannlm.lm.models.registry import get_base


# --------------------------------------------------------------------------- #
# Utility helpers


def log_progress(msg: str) -> None:
    print(f"[psann-chat] {msg}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Dataset streaming + batching utilities (adapted from bench harness)


class TextStream:
    """Re-iterable text iterator built from a factory function."""

    def __init__(self, iterator_fn: Callable[[], Iterable[str]], *, name: str) -> None:
        self.iterator_fn = iterator_fn
        self.name = name
        self._epoch = 0

    def __iter__(self) -> Iterator[str]:
        self._epoch += 1
        log_progress(f"[{self.name}] starting epoch {self._epoch}")
        for text in self.iterator_fn():
            if text:
                yield text


class SequenceBatcher:
    """Packs contiguous token streams into fixed-length batches."""

    def __init__(
        self,
        stream: TextStream,
        tokenizer,
        *,
        seq_len: int,
        micro_batch_size: int,
    ) -> None:
        from collections import deque

        self.stream = stream
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.window = self.seq_len + 1
        self.micro_batch = max(1, int(micro_batch_size))
        self._buffer = deque()
        self._iter = iter(self.stream)
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

    def _encode(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.bos_id is not None:
            ids = [self.bos_id] + ids
        if self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def _ensure_tokens(self) -> None:
        while len(self._buffer) < self.window:
            try:
                text = next(self._iter)
            except StopIteration:
                self._iter = iter(self.stream)
                continue
            ids = self._encode(text)
            if len(ids) < 2:
                continue
            self._buffer.extend(ids)

    def _next_sequence(self) -> Tuple[List[int], List[int]]:
        self._ensure_tokens()
        chunk = [self._buffer.popleft() for _ in range(self.window)]
        if not self._buffer:
            self._buffer.append(chunk[-1])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs: List[List[int]] = []
        labels: List[List[int]] = []
        for _ in range(self.micro_batch):
            x, y = self._next_sequence()
            inputs.append(x)
            labels.append(y)
        return (
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# --------------------------------------------------------------------------- #
# Dataset builders


def build_wikitext_stream(split: str = "train", *, streaming: bool = True) -> TextStream:
    log_progress(f"Loading WikiText-103 split='{split}' streaming={streaming}")
    dataset = load_dataset(
        "iohadrubin/wikitext-103-raw-v1",
        split=split,
        streaming=streaming,
    )

    def iterator() -> Iterable[str]:
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if text:
                yield text

    return TextStream(iterator, name="wikitext103")


def build_oasst_pair_stream(max_pairs: int) -> TextStream:
    """Build chat-style prompt/response pairs from OpenAssistant/oasst1."""
    log_progress("Loading OpenAssistant/oasst1 (full dataset)")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    message_by_id: Dict[str, Dict[str, str]] = {}
    for row in dataset:
        mid = row.get("message_id")
        if mid:
            message_by_id[mid] = row

    log_progress("Constructing user/assistant pairs")
    assistant_rows = [row for row in dataset if row.get("role") == "assistant"]
    random.shuffle(assistant_rows)
    max_pairs = int(max_pairs)

    def iterator() -> Iterable[str]:
        produced = 0
        for row in assistant_rows:
            parent_id = row.get("parent_id")
            if not parent_id or parent_id not in message_by_id:
                continue
            parent = message_by_id[parent_id]
            user_text = str(parent.get("text", "")).strip()
            assistant_text = str(row.get("text", "")).strip()
            if not user_text or not assistant_text:
                continue
            convo = f"User: {user_text}\nAssistant: {assistant_text}"
            yield convo
            produced += 1
            if max_pairs > 0 and produced >= max_pairs:
                break

    return TextStream(iterator, name="oasst1")


# --------------------------------------------------------------------------- #
# Model sizing helpers (reused from bench harness)


def count_model_params(base: str, vocab_size: int, **cfg: int) -> int:
    factory = get_base(base)
    model = factory(vocab_size=vocab_size, **cfg)
    total = sum(p.numel() for p in model.parameters())
    del model
    return int(total)


@dataclass
class LandingConfig:
    label: str
    target_params: int
    landed_params: int
    error_pct: float
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int


def land_config(
    label: str,
    target: int,
    *,
    vocab_size: int,
    base: str,
    width_choices: Sequence[int],
    layer_choices: Sequence[int],
    max_heads: int,
) -> LandingConfig:
    best: Optional[LandingConfig] = None
    for d_model in width_choices:
        n_heads = min(max_heads, max(4, d_model // 64))
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        d_mlp = 4 * d_model
        for n_layers in layer_choices:
            landed = count_model_params(
                base,
                vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_mlp=d_mlp,
                dropout=0.0,
                positional_encoding="rope",
                wave_interleave=True,
                wave_kernel_size=3,
                wave_dilation_growth=1,
            )
            error = abs(landed - target) / target
            cand = LandingConfig(
                label=label,
                target_params=target,
                landed_params=landed,
                error_pct=error,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_mlp=d_mlp,
            )
            if best is None or error < best.error_pct:
                best = cand
    if best is None:
        raise RuntimeError("Failed to land configuration.")
    log_progress(
        f"Landed {label}: params={best.landed_params/1e6:.1f}M "
        f"(target={best.target_params/1e6:.1f}M) "
        f"d_model={best.d_model} n_layers={best.n_layers} n_heads={best.n_heads}"
    )
    return best


# --------------------------------------------------------------------------- #
# Training core


def build_scheduler(
    optimizer: AdamW, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(0, warmup_steps)

    def lr_lambda(step: int) -> float:
        s = step + 1
        if warmup > 0 and s <= warmup:
            return float(s) / float(max(1, warmup))
        if total_steps <= warmup:
            return 1.0
        progress = float(s - warmup) / float(max(1, total_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class PhaseResult:
    phase: str
    steps_completed: int
    tokens_total: int
    avg_tokens_sec: float
    wall_clock: float
    final_loss: float
    peak_mem_gb: float


def train_phase(
    phase: str,
    model: nn.Module,
    batcher: SequenceBatcher,
    *,
    tokenizer,
    steps: int,
    tokens_per_step: int,
    seq_len: int,
    grad_accum: int,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    warmup_steps: int,
    amp_mode: str,
    grad_clip: float,
    log_interval: int,
    compile_model: bool,
) -> PhaseResult:
    log_progress(f"[{phase}] configuring optimizer (lr={lr})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if compile_model:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            log_progress(f"[{phase}] torch.compile enabled")
        except Exception as exc:  # pragma: no cover
            log_progress(f"[{phase}] torch.compile failed ({exc}); continuing without it.")

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=1e-8,
        weight_decay=weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = build_scheduler(optimizer, steps, warmup_steps)
    amp_enabled = amp_mode in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_mode == "fp16" and device.type == "cuda"))

    micro_batch = max(1, tokens_per_step // (seq_len * grad_accum))
    if micro_batch <= 0:
        raise ValueError("micro_batch computed as 0; adjust tokens_per_step or grad_accum.")
    log_progress(
        f"[{phase}] micro_batch={micro_batch} grad_accum={grad_accum} tokens/step={tokens_per_step}"
    )

    tokens_total = 0
    throughput_trace: List[float] = []
    loss_trace: List[float] = []
    step_times: List[float] = []

    model.train()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    start_time = time.perf_counter()
    for step in range(1, steps + 1):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        cumulative_loss = 0.0
        step_tokens = 0
        for _ in range(grad_accum):
            inputs, labels = batcher.next_batch()
            step_tokens += inputs.numel()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype if amp_enabled else torch.float32,
                enabled=amp_enabled,
            ):
                logits = model(inputs)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size),
                    labels.view(-1),
                )
                loss = loss / grad_accum
            if torch.isnan(loss):
                raise RuntimeError("NaN loss encountered.")
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            cumulative_loss += float(loss.item())

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        tokens_total += step_tokens
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        tokens_per_sec = step_tokens / max(step_time, 1e-6)
        throughput_trace.append(tokens_per_sec)
        loss_trace.append(cumulative_loss)

        if step % log_interval == 0 or step == 1:
            log_progress(
                f"[{phase}] step {step}/{steps} loss={cumulative_loss:.4f} "
                f"tokens/sec={tokens_per_sec:,.0f} lr={optimizer.param_groups[0]['lr']:.2e}"
            )

    wall_clock = time.perf_counter() - start_time
    peak_mem_gb = (
        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    )
    avg_tokens_sec = sum(throughput_trace) / max(1, len(throughput_trace))
    final_loss = loss_trace[-1] if loss_trace else float("nan")

    log_progress(
        f"[{phase}] completed {steps} steps in {wall_clock/3600:.2f}h | "
        f"avg tokens/sec={avg_tokens_sec:,.0f} peak_mem={peak_mem_gb:.1f}GB"
    )
    return PhaseResult(
        phase=phase,
        steps_completed=steps,
        tokens_total=tokens_total,
        avg_tokens_sec=avg_tokens_sec,
        wall_clock=wall_clock,
        final_loss=final_loss,
        peak_mem_gb=peak_mem_gb,
    )


# --------------------------------------------------------------------------- #
# Argument parsing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PSANN-LM chat model (pretrain + SFT).")
    parser.add_argument("--save-dir", type=str, default="runs/psannlm_chat")
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--pretrain-steps", type=int, default=20000)
    parser.add_argument("--sft-steps", type=int, default=4000)
    parser.add_argument("--pretrain-grad-accum", type=int, default=8)
    parser.add_argument("--sft-grad-accum", type=int, default=16)
    parser.add_argument("--lr-pretrain", type=float, default=3e-4)
    parser.add_argument("--lr-sft", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--sft-max-pairs", type=int, default=300000)
    parser.add_argument("--target-params", type=int, default=300_000_000)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Main orchestration


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    log_progress(f"Artifacts will be saved under {run_dir}")

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
    vocab_size = tokenizer.vocab_size
    log_progress(f"Tokenizer '{args.tokenizer}' vocab size: {vocab_size}")

    # Landing configuration for ~300M params
    landing = land_config(
        "chat300M",
        args.target_params,
        vocab_size=vocab_size,
        base="waveresnet",
        width_choices=[1408, 1536, 1664, 1792, 1920],
        layer_choices=list(range(10, 22, 2)),
        max_heads=32,
    )

    # Model instantiation via psannLM wrapper (for easy saving)
    sine_cfg = SineConfig()
    lm = psannLM(
        base="waveresnet",
        d_model=landing.d_model,
        n_layers=landing.n_layers,
        n_heads=landing.n_heads,
        d_mlp=landing.d_mlp,
        vocab_size=vocab_size,
        sine_params={
            "amp_init": sine_cfg.amp_init,
            "freq_init": sine_cfg.freq_init,
            "damp_init": sine_cfg.damp_init,
            "trainable": sine_cfg.trainable,
        },
        positional_encoding="rope",
    )
    model = lm._ensure_model(vocab_size)

    # Data streams
    wikitext_stream = build_wikitext_stream()
    oasst_stream = build_oasst_pair_stream(args.sft_max_pairs)

    micro_pretrain = max(1, args.tokens_per_step // (args.seq_len * args.pretrain_grad_accum))
    pretrain_batcher = SequenceBatcher(
        wikitext_stream,
        tokenizer,
        seq_len=args.seq_len,
        micro_batch_size=micro_pretrain,
    )
    micro_sft = max(1, args.tokens_per_step // (args.seq_len * args.sft_grad_accum))
    sft_batcher = SequenceBatcher(
        oasst_stream,
        tokenizer,
        seq_len=args.seq_len,
        micro_batch_size=micro_sft,
    )

    # Phase 1: Pretrain
    pretrain_result = train_phase(
        "pretrain",
        model,
        pretrain_batcher,
        tokenizer=tokenizer,
        steps=args.pretrain_steps,
        tokens_per_step=args.tokens_per_step,
        seq_len=args.seq_len,
        grad_accum=args.pretrain_grad_accum,
        lr=args.lr_pretrain,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_steps=args.warmup_steps,
        amp_mode=args.amp,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        compile_model=args.compile,
    )
    lm.save(str(run_dir / "checkpoint_pretrain.pt"))
    tokenizer.save_pretrained(run_dir / "tokenizer_pretrain")
    log_progress("Saved pretrain checkpoint.")

    # Phase 2: SFT
    sft_result = train_phase(
        "sft",
        model,
        sft_batcher,
        tokenizer=tokenizer,
        steps=args.sft_steps,
        tokens_per_step=args.tokens_per_step,
        seq_len=args.seq_len,
        grad_accum=args.sft_grad_accum,
        lr=args.lr_sft,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        warmup_steps=max(100, args.warmup_steps // 10),
        amp_mode=args.amp,
        grad_clip=args.grad_clip,
        log_interval=max(10, args.log_interval // 2),
        compile_model=args.compile,
    )

    # Final save
    final_path = run_dir / "psannlm_chat_final.pt"
    lm.save(str(final_path))
    tokenizer.save_pretrained(run_dir / "tokenizer_final")

    summary = {
        "landing": {
            "params_m": landing.landed_params / 1e6,
            "d_model": landing.d_model,
            "n_layers": landing.n_layers,
            "n_heads": landing.n_heads,
            "d_mlp": landing.d_mlp,
        },
        "pretrain": vars(pretrain_result),
        "sft": vars(sft_result),
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
        "tokens_per_step": args.tokens_per_step,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_progress(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
