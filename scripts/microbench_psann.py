#!/usr/bin/env python3
"""Microbench PSANN vs baseline models for throughput and memory.

Example:
  python scripts/microbench_psann.py --device cpu --out reports/benchmarks/microbench_cpu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn

from psann.nn import PSANNNet


@dataclass
class BenchCase:
    name: str
    model: nn.Module
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    tokens_per_step: int


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("dtype must be one of: float32, float16, bfloat16")


def _parse_shape(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", " ").split()
    if len(parts) != 2:
        raise ValueError("shape must be formatted like 8x8")
    return int(parts[0]), int(parts[1])


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _build_dense_mlp(input_dim: int, output_dim: int, layers: int, width: int) -> nn.Module:
    blocks: list[nn.Module] = []
    prev = input_dim
    for _ in range(layers):
        blocks.append(nn.Linear(prev, width))
        blocks.append(nn.ReLU())
        prev = width
    blocks.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*blocks)


class _TransformerBench(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=max(d_model * 4, 64),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.encoder(x)
        return self.head(x)


def _build_cases(args: argparse.Namespace) -> list[BenchCase]:
    cases: list[BenchCase] = []
    batch = int(args.batch_size)
    input_dim = int(args.input_dim)
    output_dim = int(args.output_dim)
    hidden_layers = int(args.hidden_layers)
    hidden_units = int(args.hidden_units)

    psann = PSANNNet(
        input_dim,
        output_dim,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation_type=args.activation,
    )
    cases.append(
        BenchCase(
            name=f"psann_{args.activation}",
            model=psann,
            input_shape=(batch, input_dim),
            target_shape=(batch, output_dim),
            tokens_per_step=batch,
        )
    )

    for baseline in _parse_list(args.baselines):
        if baseline == "dense":
            dense = _build_dense_mlp(input_dim, output_dim, hidden_layers, hidden_units)
            cases.append(
                BenchCase(
                    name="dense_relu",
                    model=dense,
                    input_shape=(batch, input_dim),
                    target_shape=(batch, output_dim),
                    tokens_per_step=batch,
                )
            )
        elif baseline == "transformer":
            seq_len = int(args.seq_len)
            transformer = _TransformerBench(
                input_dim,
                output_dim,
                d_model=hidden_units,
                layers=max(1, hidden_layers),
                heads=int(args.transformer_heads),
                dropout=float(args.transformer_dropout),
            )
            cases.append(
                BenchCase(
                    name="transformer",
                    model=transformer,
                    input_shape=(batch, seq_len, input_dim),
                    target_shape=(batch, seq_len, output_dim),
                    tokens_per_step=batch * seq_len,
                )
            )
        elif baseline:
            raise ValueError(f"Unknown baseline '{baseline}'.")

    return cases


def _bench_case(
    case: BenchCase,
    *,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    warmup_steps: int,
    compile_mode: Optional[str],
    compile_backend: Optional[str],
    tf32: bool,
) -> Dict[str, Any]:
    model = case.model.to(device=device, dtype=dtype)
    model.train()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32

    compiled = False
    compile_time_s = None
    if compile_mode:
        start = time.perf_counter()
        model = torch.compile(model, mode=compile_mode, backend=compile_backend)
        compile_time_s = time.perf_counter() - start
        compiled = True

    x = torch.randn(case.input_shape, device=device, dtype=dtype)
    y = torch.randn(case.target_shape, device=device, dtype=dtype)
    loss_fn = nn.MSELoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup_steps):
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        model.zero_grad(set_to_none=True)

    _sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    _sync(device)
    elapsed = time.perf_counter() - start

    max_alloc = None
    max_reserved = None
    if device.type == "cuda":
        max_alloc = int(torch.cuda.max_memory_allocated())
        max_reserved = int(torch.cuda.max_memory_reserved())

    tokens_per_sec = (case.tokens_per_step * steps) / max(elapsed, 1e-9)
    return {
        "steps": steps,
        "warmup_steps": warmup_steps,
        "elapsed_s": elapsed,
        "tokens_per_step": case.tokens_per_step,
        "tokens_per_s": tokens_per_sec,
        "param_count": _param_count(model),
        "compiled": compiled,
        "compile_time_s": compile_time_s,
        "max_memory_allocated": max_alloc,
        "max_memory_reserved": max_reserved,
        "input_shape": case.input_shape,
        "target_shape": case.target_shape,
    }


def run_bench(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)
    compile_mode = args.compile_mode if args.compile else None
    compile_backend = args.compile_backend if args.compile else None

    cases = _build_cases(args)
    results = {
        case.name: _bench_case(
            case,
            device=device,
            dtype=dtype,
            steps=int(args.steps),
            warmup_steps=int(args.warmup_steps),
            compile_mode=compile_mode,
            compile_backend=compile_backend,
            tf32=bool(args.tf32),
        )
        for case in cases
    }

    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "compile": bool(args.compile),
            "compile_mode": compile_mode,
            "compile_backend": compile_backend,
            "tf32": bool(args.tf32),
            "seed": int(args.seed),
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--input-dim", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=1)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--hidden-units", type=int, default=128)
    p.add_argument("--activation", type=str, default="psann")
    p.add_argument("--baselines", type=str, default="dense")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-dropout", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=3)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--compile-backend", type=str, default=None)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="reports/benchmarks/microbench.json")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    payload = run_bench(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[microbench] Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
