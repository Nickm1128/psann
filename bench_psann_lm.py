#!/usr/bin/env python
"""PSANN-LM scaling + throughput benchmark harness.

This script prepares Hugging Face text datasets, runs a parameter-count sweep
over PSANN-LM configs, and records throughput / wall-clock / loss metrics for
each model size. It is designed to be copied to a GPU worker (e.g., RunPod) and
invoked via:

    python bench_psann_lm.py --sizes 15M,50M,125M,250M,500M \
        --tokens-per-step 32768 --steps 2000 --dataset wikitext-103 \
        --stream --seed 1337 --save-dir runs/psannlm_bench --grad-accum 4 \
        --compile --bf16

Artifacts per run (size x seed):
  - step_metrics.jsonl (per-step JSON log)
  - metrics.json (summary payload)
  - loss_curve.csv (step vs loss snapshot)

Global artifacts (under save-dir/<timestamp>/):
  - params_landing.json (landed config + parameter counts)
  - summary.csv / summary.json (one row per size with aggregated stats)
  - throughput_vs_size.png, wallclock_vs_size.png, loss_vs_tokens.png
  - optional participation_ratio.png when diagnostics enabled
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    from datasets import load_dataset  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'datasets'. Install via `pip install datasets`."
    ) from exc

try:
    from transformers import AutoTokenizer  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency 'transformers'. Install via `pip install transformers`."
    ) from exc

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from psannlm.lm.models.registry import get_base
from psannlm.lm.models.sine import SineConfig
from psann.utils.diagnostics import participation_ratio


DATASET_ALIASES = {
    "wikitext-103": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext103": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext_103": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext_2": ("wikitext", "wikitext-2-raw-v1"),
}


def log_progress(message: str) -> None:
    """Emit a flushed progress line for long-running RunPod jobs."""
    print(f"[bench] {message}", flush=True)


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
    dropout: float
    positional_encoding: str
    wave_interleave: bool
    wave_kernel_size: int
    wave_dilation_growth: int


@dataclass
class RunResult:
    label: str
    seed: int
    target_params: int
    landed_params: int
    steps_completed: int
    tokens_per_step: int
    tokens_total: int
    avg_tokens_sec: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    peak_mem_gb: float
    final_train_loss: float
    eval_ppl: Optional[float]
    wall_clock_s: float
    metrics_path: Path
    stability: Dict[str, bool]
    throughput_trace: List[float]
    loss_trace: List[Tuple[int, float]]
    participation_ratio: Optional[float]


def parse_size_targets(arg: str) -> List[Tuple[str, int]]:
    if not arg:
        raise ValueError("At least one size must be provided.")
    log_progress(f"parse_size_targets -> raw='{arg}'")
    tokens = [tok.strip() for tok in arg.split(",") if tok.strip()]
    results: List[Tuple[str, int]] = []
    for tok in tokens:
        label = tok.upper().replace(" ", "")
        mult = 1.0
        numeric = tok
        if label.endswith("M"):
            mult = 1_000_000
            numeric = label[:-1]
        elif label.endswith("B"):
            mult = 1_000_000_000
            numeric = label[:-1]
        try:
            value = float(numeric)
        except ValueError as exc:
            raise ValueError(f"Could not parse size token '{tok}'.") from exc
        params = int(value * mult)
        if params <= 0:
            raise ValueError(f"Size token '{tok}' resolved to non-positive params.")
        results.append((label, params))
    log_progress(f"parse_size_targets -> landed={results}")
    return results


def seed_everything(seed: int) -> None:
    log_progress(f"Seeding RNGs with seed={seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dep
        pass


def resolve_dataset(dataset: str, explicit_hub_id: Optional[str], dataset_name: Optional[str]) -> Tuple[str, Optional[str]]:
    log_progress(
        f"resolve_dataset -> dataset={dataset} explicit_hub_id={explicit_hub_id} dataset_name={dataset_name}"
    )
    if explicit_hub_id:
        return explicit_hub_id, dataset_name
    key = dataset.strip().lower()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    if "/" in dataset:
        return dataset, dataset_name
    return dataset, dataset_name


def _detect_text_field(dataset) -> str:
    sample = None
    if hasattr(dataset, "take"):
        iterator = dataset.take(1)
        sample = next(iter(iterator), None)
    if sample is None:
        try:
            sample = dataset[0]
        except Exception:
            pass
    if sample is None:
        raise RuntimeError("Unable to inspect dataset schema for text field detection.")
    row = dict(sample)
    if "text" in row and isinstance(row["text"], str):
        field = "text"
        log_progress(f"Detected default text field '{field}'.")
        return field
    for key in ("content", "article", "document", "body"):
        if key in row and isinstance(row[key], str):
            log_progress(f"Detected fallback text field '{key}'.")
            return key
    for key, value in row.items():
        if isinstance(value, str):
            log_progress(f"Detected inferred text field '{key}'.")
            return key
    raise RuntimeError("Could not find any string field in dataset sample.")


def _normalize_row_text(row: dict, text_field: str) -> Optional[str]:
    if text_field in row and isinstance(row[text_field], str):
        text = row[text_field].strip()
        if text:
            return text
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


class TextStream:
    """Re-iterable text iterator with optional shuffling."""

    def __init__(
        self,
        dataset,
        text_field: str,
        *,
        streaming: bool,
        shuffle: bool,
        seed: int,
        shuffle_buffer: int,
    ) -> None:
        self.dataset = dataset
        self.text_field = text_field
        self.streaming = streaming
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = max(1_000, int(shuffle_buffer)) if shuffle else 0
        self._epoch = 0
        log_progress(
            f"TextStream init -> streaming={self.streaming} shuffle={self.shuffle} buffer={self.shuffle_buffer}"
        )

    def __iter__(self) -> Iterator[str]:
        while True:
            ds = self.dataset
            if self.shuffle and hasattr(ds, "shuffle"):
                shuffle_seed = self.seed + self._epoch
                if self.streaming:
                    ds = ds.shuffle(seed=shuffle_seed, buffer_size=self.shuffle_buffer)
                else:
                    ds = ds.shuffle(seed=shuffle_seed)
            self._epoch += 1
            for row in ds:
                text = _normalize_row_text(dict(row), self.text_field)
                if text:
                    yield text


class SequenceBatcher:
    """Packs contiguous token streams into fixed-length batches."""

    def __init__(
        self,
        text_stream: TextStream,
        tokenizer,
        *,
        seq_len: int,
        micro_batch_size: int,
    ) -> None:
        from collections import deque

        self.stream = text_stream
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.window = self.seq_len + 1
        self.micro_batch = max(1, int(micro_batch_size))
        self._buffer = deque()  # type: ignore[var-annotated]
        self._iter = iter(self.stream)
        log_progress(
            f"SequenceBatcher init -> seq_len={self.seq_len} micro_batch={self.micro_batch}"
        )

    def reset(self) -> None:
        self._buffer.clear()
        self._iter = iter(self.stream)

    def _ensure_tokens(self) -> None:
        while len(self._buffer) < self.window:
            try:
                text = next(self._iter)
            except StopIteration:
                self._iter = iter(self.stream)
                continue
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=True,
            )
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
        input_tensor = torch.tensor(inputs, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return input_tensor, label_tensor


def prepare_tokenizer(name_or_path: str, seq_len: int):
    log_progress(f"Preparing tokenizer '{name_or_path}' (seq_len={seq_len})")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|eos|>"})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|bos|>"})
    tokenizer.model_max_length = seq_len + 1
    tokenizer.truncation_side = "right"
    tokenizer.padding_side = "right"
    log_progress(
        f"Tokenizer ready -> vocab_size={tokenizer.vocab_size} pad={tokenizer.pad_token} eos={tokenizer.eos_token}"
    )
    return tokenizer


def tie_embeddings(model: nn.Module) -> None:
    if hasattr(model, "embed") and hasattr(model, "lm_head"):
        embed = getattr(model, "embed")
        head = getattr(model, "lm_head")
        if isinstance(embed, nn.Embedding) and isinstance(head, nn.Linear):
            head.weight = embed.weight  # type: ignore[assignment]
            log_progress("Tied input embedding weights to LM head.")


def count_model_params(
    base: str,
    vocab_size: int,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_mlp: int,
    dropout: float,
    positional_encoding: str,
    wave_interleave: bool,
    wave_kernel_size: int,
    wave_dilation_growth: int,
) -> int:
    factory = get_base(base)
    cfg = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        dropout=dropout,
        positional_encoding=positional_encoding,
        wave_interleave=wave_interleave,
        wave_kernel_size=wave_kernel_size,
        wave_dilation_growth=wave_dilation_growth,
    )
    log_progress(
        f"Counting params -> base={base} d_model={d_model} n_layers={n_layers} n_heads={n_heads}"
    )
    model = factory(**cfg)
    total = sum(p.numel() for p in model.parameters())
    del model
    return int(total)


def suggest_head_count(d_model: int, max_heads: int) -> int:
    target = max(4, min(max_heads, d_model // 64 or 1))
    while target > 1 and d_model % target != 0:
        target -= 1
    heads = max(1, target)
    log_progress(f"suggest_head_count -> d_model={d_model} heads={heads}")
    return heads


def land_configs(
    targets: List[Tuple[str, int]],
    *,
    vocab_size: int,
    base: str,
    width_choices: Sequence[int],
    layer_min: int,
    layer_max: int,
    layer_step: int,
    dropout: float,
    positional_encoding: str,
    wave_interleave: bool,
    wave_kernel_size: int,
    wave_dilation_growth: int,
    max_heads: int,
    max_error: float,
) -> List[LandingConfig]:
    landings: List[LandingConfig] = []
    cache: Dict[Tuple[int, int, int], int] = {}
    for label, target in targets:
        best: Optional[LandingConfig] = None
        log_progress(f"Landing config for size {label} ({target/1e6:.2f}M params target)")
        for d_model in width_choices:
            n_heads = suggest_head_count(d_model, max_heads)
            if d_model % n_heads != 0:
                continue
            d_mlp = 4 * d_model
            for n_layers in range(layer_min, layer_max + 1, layer_step):
                key = (d_model, n_layers, n_heads)
                params = cache.get(key)
                if params is None:
                    params = count_model_params(
                        base,
                        vocab_size,
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        d_mlp=d_mlp,
                        dropout=dropout,
                        positional_encoding=positional_encoding,
                        wave_interleave=wave_interleave,
                        wave_kernel_size=wave_kernel_size,
                        wave_dilation_growth=wave_dilation_growth,
                    )
                    cache[key] = params
                error = abs(params - target) / target
                candidate = LandingConfig(
                    label=label,
                    target_params=target,
                    landed_params=params,
                    error_pct=error,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_mlp=d_mlp,
                    dropout=dropout,
                    positional_encoding=positional_encoding,
                    wave_interleave=wave_interleave,
                    wave_kernel_size=wave_kernel_size,
                    wave_dilation_growth=wave_dilation_growth,
                )
                if best is None or error < best.error_pct or (
                    math.isclose(error, best.error_pct) and params < best.landed_params
                ):
                    best = candidate
        if best is None:
            raise RuntimeError(f"Unable to land configuration for size {label}.")
        if best.error_pct > max_error:
            print(
                f"[warn] Size {label}: landed error {best.error_pct*100:.2f}% exceeds tolerance "
                f"({max_error*100:.2f}%). Consider adjusting width/layer search.",
                flush=True,
            )
        log_progress(
            f"Landed {label}: params={best.landed_params} "
            f"error={best.error_pct*100:.2f}% d_model={best.d_model} layers={best.n_layers}"
        )
        landings.append(best)
    return landings


def build_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    warmup = max(0, int(warmup_steps))
    log_progress(f"Building scheduler -> total_steps={total_steps} warmup={warmup}")

    def lr_lambda(step: int) -> float:
        s = step + 1
        if warmup > 0 and s <= warmup:
            return float(s) / float(max(1, warmup))
        if total_steps <= warmup:
            return 1.0
        progress = float(s - warmup) / float(max(1, total_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


class StepLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def log(self, payload: Dict[str, object]) -> None:
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class GPUStats:
    def __init__(self) -> None:
        self.enabled = False
        self._handle = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
            self.max_gpu_util = 0.0
            self.max_mem_util = 0.0
        except Exception:
            self.enabled = False
            self._pynvml = None
            self.max_gpu_util = 0.0
            self.max_mem_util = 0.0
        log_progress(f"GPUStats init -> enabled={self.enabled}")

    def sample(self) -> None:
        if not self.enabled or self._handle is None:
            return
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.max_gpu_util = max(self.max_gpu_util, float(util.gpu))
            mem_util = 100.0 * float(mem.used) / float(mem.total)
            self.max_mem_util = max(self.max_mem_util, mem_util)
        except Exception:
            pass

    def close(self) -> None:
        if self.enabled:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def compute_step_quantiles(times_ms: List[float]) -> Tuple[float, float, float]:
    if not times_ms:
        return 0.0, 0.0, 0.0
    p50 = statistics.median(times_ms)
    if len(times_ms) == 1:
        return p50, p50, p50
    p90 = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) >= 10 else max(times_ms)
    p99 = statistics.quantiles(times_ms, n=100)[98] if len(times_ms) >= 100 else max(times_ms)
    return p50, p90, p99


def evaluate_model(
    model: nn.Module,
    batcher: SequenceBatcher,
    *,
    device: torch.device,
    vocab_size: int,
    eval_batches: int,
) -> Optional[Dict[str, float]]:
    if eval_batches <= 0:
        return None
    log_progress(f"Starting eval pass -> batches={eval_batches}")
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            inputs, labels = batcher.next_batch()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += float(loss.item())
            total_tokens += inputs.numel()
    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    log_progress(f"Eval finished -> loss={avg_loss:.4f} ppl={ppl:.2f}")
    return {"loss": avg_loss, "ppl": ppl}


def snapshot_participation_ratio(
    model: nn.Module,
    batcher: SequenceBatcher,
    *,
    device: torch.device,
    samples: int,
) -> Optional[float]:
    if samples <= 0:
        return None
    log_progress(f"Running participation ratio diagnostic -> samples={samples}")
    collected: List[torch.Tensor] = []
    handle = None

    def _hook(_module, _inp, output):
        collected.append(output.detach().float().cpu())
        return output

    if hasattr(model, "ln_f"):
        handle = model.ln_f.register_forward_hook(_hook)  # type: ignore[attr-defined]
    model.eval()
    with torch.no_grad():
        for _ in range(samples):
            inputs, _ = batcher.next_batch()
            inputs = inputs.to(device, non_blocking=True)
            model(inputs)
    if handle is not None:
        handle.remove()
    if not collected:
        return None
    feats = torch.cat([f.mean(dim=1) for f in collected], dim=0)
    if feats.ndim != 2:
        feats = feats.reshape(feats.size(0), -1)
    pr = participation_ratio(feats)
    log_progress(f"Participation ratio computed -> {pr:.4f}")
    return pr


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
        commit = out.decode().strip()
        log_progress(f"Git commit detected -> {commit}")
        return commit
    except Exception:
        log_progress("Git commit unavailable.")
        return None


def train_one_size(
    landing: LandingConfig,
    *,
    args,
    tokenizer,
    train_stream: TextStream,
    val_stream: Optional[TextStream],
    device: torch.device,
    run_dir: Path,
) -> RunResult:
    log_progress(f"train_one_size -> label={landing.label} seed={args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"Output directory: {run_dir}")
    step_log = StepLogger(run_dir / "step_metrics.jsonl")
    loss_curve_path = run_dir / "loss_curve.csv"

    micro_batch = max(1, args.tokens_per_step // (args.seq_len * args.grad_accum))
    tokens_per_step = micro_batch * args.seq_len * args.grad_accum
    if tokens_per_step != args.tokens_per_step:
        print(
            f"[warn] Requested tokens/step={args.tokens_per_step} but "
            f"achievable value is {tokens_per_step} "
            f"(seq_len={args.seq_len}, grad_accum={args.grad_accum}, micro_batch={micro_batch}).",
            flush=True,
        )
    log_progress(
        f"Train setup -> micro_batch={micro_batch} tokens_per_step={tokens_per_step} grad_accum={args.grad_accum}"
    )

    train_batcher = SequenceBatcher(
        train_stream,
        tokenizer,
        seq_len=args.seq_len,
        micro_batch_size=micro_batch,
    )
    eval_batcher = None
    if val_stream is not None:
        eval_batcher = SequenceBatcher(
            val_stream,
            tokenizer,
            seq_len=args.seq_len,
            micro_batch_size=micro_batch,
        )

    factory = get_base(args.base)
    log_progress(f"Instantiating model base='{args.base}' for {landing.label}")
    sine_cfg = SineConfig(
        amp_init=args.sine_amp,
        freq_init=args.sine_freq,
        damp_init=args.sine_damp,
        trainable=not args.freeze_sine,
    )
    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=landing.d_model,
        n_layers=landing.n_layers,
        n_heads=landing.n_heads,
        d_mlp=landing.d_mlp,
        dropout=landing.dropout,
        positional_encoding=landing.positional_encoding,
        mlp_activation=args.mlp_activation,
        sine=sine_cfg,
        wave_interleave=landing.wave_interleave,
        wave_kernel_size=landing.wave_kernel_size,
        wave_dilation_growth=landing.wave_dilation_growth,
    )
    model = factory(**model_kwargs)
    tie_embeddings(model)
    if args.grad_checkpoint and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)  # type: ignore[attr-defined]
        log_progress("Gradient checkpointing enabled on model.")
    model.to(device)
    log_progress(f"Model moved to device {device}.")
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            log_progress("torch.compile succeeded.")
        except Exception as exc:
            print(f"[warn] torch.compile failed ({exc}); continuing without compilation.")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = build_scheduler(optimizer, args.steps, args.warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp_mode == "fp16" and device.type == "cuda"))
    amp_enabled = args.amp_mode in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_mode == "bf16" else torch.float16
    criterion = nn.CrossEntropyLoss()
    log_progress(
        f"Optimization ready -> lr={args.lr} amp={args.amp_mode} grad_clip={args.grad_clip} tf32={args.tf32}"
    )

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    tokens_total = 0
    loss_history: List[Tuple[int, float]] = []
    throughput_history: List[float] = []
    step_times_ms: List[float] = []
    moving_tps = collections.deque(maxlen=20)
    grad_norms: List[float] = []
    nan_flag = False
    oom_flag = False
    best_loss = float("inf")
    start_time = time.perf_counter()
    gpu_stats = GPUStats()

    loss_curve_fh = loss_curve_path.open("w", newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_curve_fh)
    loss_writer.writerow(["step", "loss"])

    try:
        log_progress(f"Starting training loop for {landing.label} with {args.steps} steps.")
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_tokens = 0
            micro_losses = []
            for _ in range(args.grad_accum):
                inputs, labels = train_batcher.next_batch()
                step_tokens += inputs.numel()
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype if amp_enabled else torch.float32,
                    enabled=amp_enabled,
                ):
                    logits = model(inputs)
                    loss = criterion(
                        logits.view(-1, tokenizer.vocab_size),
                        labels.view(-1),
                    )
                    loss = loss / args.grad_accum
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_flag = True
                    raise RuntimeError("NaN detected in loss.")
                micro_losses.append(float(loss.item()))
                if scaler is not None and amp_enabled and args.amp_mode == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            )
            grad_norms.append(grad_norm)
            if scaler is not None and amp_enabled and args.amp_mode == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_time = time.perf_counter() - step_start
            step_ms = step_time * 1000.0
            step_times_ms.append(step_ms)
            tokens_total += step_tokens
            tokens_per_sec = step_tokens / max(step_time, 1e-6)
            moving_tps.append(tokens_per_sec)
            throughput_history.append(tokens_per_sec)
            avg_loss = float(sum(micro_losses) / max(1, len(micro_losses)))
            loss_history.append((step, avg_loss))
            best_loss = min(best_loss, avg_loss)
            loss_writer.writerow([step, avg_loss])
            gpu_stats.sample()
            payload = {
                "step": step,
                "loss": avg_loss,
                "tokens_per_sec": tokens_per_sec,
                "moving_avg_tokens_per_sec": sum(moving_tps) / max(1, len(moving_tps)),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "ms_per_step": step_ms,
                "tokens_this_step": step_tokens,
            }
            if step % args.log_interval == 0 or step == 1:
                print(
                    f"[{landing.label}] step {step}/{args.steps} "
                    f"loss={avg_loss:.4f} "
                    f"tps={tokens_per_sec:,.0f} "
                    f"lr={payload['lr']:.2e}",
                    flush=True,
                )
            step_log.log(payload)
            if args.max_seconds and (time.perf_counter() - start_time) > args.max_seconds:
                print(f"[info] Max wall-clock {args.max_seconds}s reached; stopping early.", flush=True)
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            oom_flag = True
            print("[error] CUDA OOM encountered; aborting run.", flush=True)
            torch.cuda.empty_cache()
            log_progress("Encountered CUDA OOM; emptied cache.")
        else:
            log_progress(f"Runtime error encountered: {exc}")
            raise
    finally:
        step_log.close()
        loss_curve_fh.close()
        gpu_stats.close()
        log_progress("Training loop finished; files closed.")

    steps_completed = len(loss_history)
    wall_clock = time.perf_counter() - start_time
    avg_tokens_sec = sum(throughput_history) / max(1, len(throughput_history))
    p50_ms, p90_ms, p99_ms = compute_step_quantiles(step_times_ms)
    peak_mem_gb = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device)
        peak_mem_gb = peak_mem / (1024 ** 3)
    eval_stats = None
    if eval_batcher is not None and steps_completed > 0:
        log_progress("Running evaluation on validation stream.")
        eval_stats = evaluate_model(
            model,
            eval_batcher,
            device=device,
            vocab_size=tokenizer.vocab_size,
            eval_batches=args.eval_batches,
        )
    pr_value = None
    if args.pr_samples > 0 and val_stream is not None and steps_completed > 0:
        log_progress("Running participation ratio diagnostic.")
        diag_batcher = SequenceBatcher(
            val_stream,
            tokenizer,
            seq_len=args.seq_len,
            micro_batch_size=max(1, min(4, micro_batch)),
        )
        pr_value = snapshot_participation_ratio(
            model,
            diag_batcher,
            device=device,
            samples=args.pr_samples,
        )

    metrics = {
        "label": landing.label,
        "seed": args.seed,
        "target_params": landing.target_params,
        "landed_params": landing.landed_params,
        "param_error_pct": landing.error_pct,
        "config": asdict(landing),
        "training": {
            "steps_requested": args.steps,
            "steps_completed": steps_completed,
            "tokens_per_step": tokens_per_step,
            "tokens_total": tokens_total,
            "tokens_per_sec_avg": avg_tokens_sec,
            "wall_clock_s": wall_clock,
            "grad_accum": args.grad_accum,
            "seq_len": args.seq_len,
            "grad_clip": args.grad_clip,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "amp_mode": args.amp_mode,
            "compile": args.compile,
        },
        "optimizer": {
            "beta1": args.beta1,
            "beta2": args.beta2,
        },
        "loss": {
            "final": loss_history[-1][1] if loss_history else None,
            "best": best_loss if best_loss < float("inf") else None,
        },
        "eval": eval_stats,
        "diagnostics": {
            "participation_ratio": pr_value,
            "grad_norms": grad_norms[-min(len(grad_norms), 16):],
            "step_time_ms": {
                "p50": p50_ms,
                "p90": p90_ms,
                "p99": p99_ms,
            },
            "gpu_util_max": gpu_stats.max_gpu_util,
            "gpu_mem_util_max": gpu_stats.max_mem_util,
        },
        "stability": {
            "nan": nan_flag,
            "oom": oom_flag,
        },
        "environment": {
            "git_commit": get_git_commit(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu",
            "torch_version": torch.__version__,
        },
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_progress(f"Wrote metrics.json for {landing.label} at {metrics_path}")

    final_loss = loss_history[-1][1] if loss_history else float("nan")
    eval_ppl = eval_stats["ppl"] if eval_stats else None
    return RunResult(
        label=landing.label,
        seed=args.seed,
        target_params=landing.target_params,
        landed_params=landing.landed_params,
        steps_completed=steps_completed,
        tokens_per_step=tokens_per_step,
        tokens_total=tokens_total,
        avg_tokens_sec=avg_tokens_sec,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p99_ms=p99_ms,
        peak_mem_gb=peak_mem_gb,
        final_train_loss=final_loss,
        eval_ppl=eval_ppl,
        wall_clock_s=wall_clock,
        metrics_path=metrics_path,
        stability={"nan": nan_flag, "oom": oom_flag},
        throughput_trace=throughput_history,
        loss_trace=loss_history,
        participation_ratio=pr_value,
    )


def aggregate_results(results: List[RunResult]) -> List[Dict[str, object]]:
    log_progress("Aggregating run results.")
    by_label: Dict[str, List[RunResult]] = collections.defaultdict(list)
    for res in results:
        by_label[res.label].append(res)
    rows: List[Dict[str, object]] = []
    for label, runs in sorted(by_label.items(), key=lambda kv: kv[0]):
        tokens_per_step = runs[0].tokens_per_step if runs else 0
        target_params = runs[0].target_params if runs else 0
        landed_params = statistics.mean(r.landed_params for r in runs)
        avg_tokens_sec = statistics.mean(r.avg_tokens_sec for r in runs)
        std_tokens_sec = statistics.pstdev(r.avg_tokens_sec for r in runs) if len(runs) > 1 else 0.0
        final_loss = statistics.mean(r.final_train_loss for r in runs)
        eval_ppl = (
            statistics.mean(r.eval_ppl for r in runs if r.eval_ppl is not None)
            if any(r.eval_ppl is not None for r in runs)
            else None
        )
        wall_clock = statistics.mean(r.wall_clock_s for r in runs)
        peak_mem = statistics.mean(r.peak_mem_gb for r in runs)
        log_progress(
            f"Aggregate {label}: avg_tokens_sec={avg_tokens_sec:.0f} wall_clock={wall_clock:.1f}s seeds={len(runs)}"
        )
        rows.append(
            {
                "size_label": label,
                "target_params": target_params,
                "landed_params": landed_params,
                "tokens_per_step": tokens_per_step,
                "avg_tokens_sec": avg_tokens_sec,
                "std_tokens_sec": std_tokens_sec,
                "p50_ms": statistics.mean(r.p50_ms for r in runs),
                "p90_ms": statistics.mean(r.p90_ms for r in runs),
                "p99_ms": statistics.mean(r.p99_ms for r in runs),
                "peak_mem_gb": peak_mem,
                "final_train_loss": final_loss,
                "eval_ppl": eval_ppl,
                "wall_clock_s": wall_clock,
                "seeds": [r.seed for r in runs],
                "participation_ratio": statistics.mean(
                    [r.participation_ratio for r in runs if r.participation_ratio is not None]
                )
                if any(r.participation_ratio is not None for r in runs)
                else None,
            }
        )
    return rows


def save_summary(rows: List[Dict[str, object]], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    if not rows:
        return
    log_progress(f"Saving summary artifacts to {out_dir}")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot_scaling(rows: List[Dict[str, object]], results: List[RunResult], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[warn] matplotlib not available; skipping plots.", flush=True)
        return
    if not rows:
        return
    log_progress("Generating scaling plots.")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r["target_params"])
    sizes = [r["target_params"] / 1e6 for r in sorted_rows]
    labels = [r["size_label"] for r in sorted_rows]
    throughput = [r["avg_tokens_sec"] for r in sorted_rows]
    wall_clock = [r["wall_clock_s"] for r in sorted_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, throughput, marker="o")
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Tokens / second")
    plt.title("Throughput vs Model Size")
    plt.grid(True, alpha=0.2)
    plt.xticks(sizes, labels, rotation=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "throughput_vs_size.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, wall_clock)
    plt.ylabel("Wall-clock (s) to steps")
    plt.title("Wall-clock vs Model Size")
    plt.tight_layout()
    plt.savefig(plot_dir / "wallclock_vs_size.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    for label in labels:
        run = next((r for r in results if r.label == label), None)
        if not run or not run.loss_trace:
            continue
        xs = [step * run.tokens_per_step / 1e6 for step, _ in run.loss_trace]
        ys = [loss for _, loss in run.loss_trace]
        plt.plot(xs, ys, label=label)
    plt.xlabel("Tokens seen (Millions)")
    plt.ylabel("Train loss (xe)")
    plt.title("Loss vs Tokens")
    plt.grid(True, alpha=0.2)
    if any(run.loss_trace for run in results):
        plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_vs_tokens.png", dpi=200)
    plt.close()

    if any(r["participation_ratio"] for r in rows):
        pr_values = [r["participation_ratio"] or 0.0 for r in sorted_rows]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, pr_values)
        plt.ylabel("Participation Ratio")
        plt.title("Jacobian Participation Ratio (post-train)")
        plt.tight_layout()
        plt.savefig(plot_dir / "participation_ratio.png", dpi=200)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSANN-LM throughput bench harness")
    parser.add_argument("--sizes", type=str, required=True, help="Comma-separated target sizes (e.g., 15M,50M,125M).")
    parser.add_argument("--base", type=str, default="waveresnet", help="Model base (waveresnet or respsann).")
    parser.add_argument("--dataset", type=str, default="wikitext-103", help="Dataset alias or HF repo id.")
    parser.add_argument("--dataset-hub-id", type=str, default=None, help="Explicit HF hub dataset id (overrides alias).")
    parser.add_argument("--dataset-name", type=str, default=None, help="HF dataset config/name if required.")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--dataset-cache", type=str, default=None, help="HF cache directory.")
    parser.add_argument("--hf-token", type=str, default=None, help="HF auth token (if dataset gated).")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mlp-activation", type=str, default="sine")
    parser.add_argument("--sine-amp", type=float, default=1.0)
    parser.add_argument("--sine-freq", type=float, default=1.0)
    parser.add_argument("--sine-damp", type=float, default=0.01)
    parser.add_argument("--freeze-sine", action="store_true")
    parser.add_argument("--amp-mode", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "none"])
    parser.add_argument("--bf16", action="store_true", help="Shortcut for --amp-mode bf16")
    parser.add_argument("--fp16", action="store_true", help="Shortcut for --amp-mode fp16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Enable HF streaming datasets.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset between epochs.")
    parser.add_argument("--shuffle-buffer", type=int, default=8192, help="Shuffle buffer size for streaming datasets.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--multi-seed-sizes", type=str, default="", help="Comma-separated size labels to run multiple seeds for.")
    parser.add_argument("--multi-seed-repeats", type=int, default=1, help="Number of seeds for specified sizes.")
    parser.add_argument("--save-dir", type=str, default="runs/psannlm_bench")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional wall-clock cap per run.")
    parser.add_argument("--width-choices", type=str, default="384,512,640,768,896,1024,1152,1280,1536,1792,2048")
    parser.add_argument("--min-layers", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=48)
    parser.add_argument("--layer-step", type=int, default=2)
    parser.add_argument("--max-heads", type=int, default=32)
    parser.add_argument("--max-param-error", type=float, default=0.03)
    parser.add_argument("--wave-kernel-size", type=int, default=3)
    parser.add_argument("--wave-dilation-growth", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--pr-samples", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--positional-encoding", type=str, default="rope")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_progress(f"Arguments parsed: {args}")
    if args.bf16:
        args.amp_mode = "bf16"
    if args.fp16:
        args.amp_mode = "fp16"
    if args.amp_mode == "none":
        args.amp_mode = "fp32"
    width_choices = [int(w.strip()) for w in args.width_choices.split(",") if w.strip()]
    log_progress(f"Width choices: {width_choices}")
    targets = parse_size_targets(args.sizes)
    dataset_id, dataset_name = resolve_dataset(args.dataset, args.dataset_hub_id, args.dataset_name)
    log_progress(
        f"Preparing to load dataset id={dataset_id} name={dataset_name} split={args.train_split}"
    )
    train_ds = load_dataset(
        dataset_id,
        name=dataset_name,
        split=args.train_split,
        streaming=args.stream,
        use_auth_token=args.hf_token,
        cache_dir=args.dataset_cache,
    )
    try:
        val_ds = load_dataset(
            dataset_id,
            name=dataset_name,
            split=args.eval_split,
            streaming=args.stream,
            use_auth_token=args.hf_token,
            cache_dir=args.dataset_cache,
        )
    except Exception:
        print(f"[warn] Eval split '{args.eval_split}' unavailable; using train split for eval.", flush=True)
        val_ds = None
    text_field = _detect_text_field(train_ds)
    tokenizer = prepare_tokenizer(args.tokenizer, args.seq_len)
    vocab_size = tokenizer.vocab_size
    log_progress(f"Tokenizer vocab size: {vocab_size}")

    train_stream = TextStream(
        train_ds,
        text_field,
        streaming=args.stream,
        shuffle=args.shuffle,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    val_stream = (
        TextStream(
            val_ds,
            text_field,
            streaming=args.stream,
            shuffle=False,
            seed=args.seed + 1,
            shuffle_buffer=args.shuffle_buffer,
        )
        if val_ds is not None
        else None
    )

    seed_everything(args.seed)

    landings = land_configs(
        targets,
        vocab_size=vocab_size,
        base=args.base,
        width_choices=width_choices,
        layer_min=args.min_layers,
        layer_max=args.max_layers,
        layer_step=args.layer_step,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding,
        wave_interleave=True,
        wave_kernel_size=args.wave_kernel_size,
        wave_dilation_growth=args.wave_dilation_growth,
        max_heads=args.max_heads,
        max_error=args.max_param_error,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save_dir) / f"{timestamp}_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params_landing.json"
    params_path.write_text(json.dumps([asdict(l) for l in landings], indent=2), encoding="utf-8")
    log_progress(f"Saved parameter landing to {params_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log_progress(f"Using device: {device}")
    results: List[RunResult] = []
    multi_seed_targets = {label.strip().upper() for label in args.multi_seed_sizes.split(",") if label.strip()}

    for landing in landings:
        seeds = [args.seed]
        if landing.label.upper() in multi_seed_targets and args.multi_seed_repeats > 1:
            seeds = [args.seed + idx for idx in range(args.multi_seed_repeats)]
        for run_seed in seeds:
            print(f"\n[info] === Running size {landing.label} (seed {run_seed}) ===", flush=True)
            log_progress(f"Dispatching run for {landing.label} seed={run_seed}")
            seed_everything(run_seed)
            run_dir = out_dir / f"{landing.label}_seed{run_seed}"
            run_result = train_one_size(
                landing,
                args=argparse.Namespace(**{**vars(args), "seed": run_seed}),
                tokenizer=tokenizer,
                train_stream=train_stream,
                val_stream=val_stream,
                device=device,
                run_dir=run_dir,
            )
            results.append(run_result)

    summary_rows = aggregate_results(results)
    save_summary(summary_rows, out_dir)
    if not args.no_plots:
        plot_scaling(summary_rows, results, out_dir)

    log_progress("Benchmark runs complete. Emitting summary to stdout.")
    print("\n[done] Benchmark complete. Summary rows:")
    for row in summary_rows:
        print(
            f"- {row['size_label']}: params={row['landed_params']/1e6:.1f}M "
            f"tokens/sec={row['avg_tokens_sec']:.0f} "
            f"p50={row['p50_ms']:.1f}ms wall={row['wall_clock_s']:.1f}s "
            f"loss={row['final_train_loss']:.3f} "
            f"eval_ppl={row['eval_ppl'] if row['eval_ppl'] else 'n/a'}",
            flush=True,
        )
    print(f"\nArtifacts stored under: {out_dir}")


if __name__ == "__main__":
    main()
