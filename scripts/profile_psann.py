#!/usr/bin/env python3
"""Profile PSANN or baseline models with torch.profiler and optional NVTX markers.

Example:
  python scripts/profile_psann.py --device cpu --model psann --out reports/profiles/psann_cpu
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn

from psann.nn import PSANNNet
from psann.nn_geo_sparse import GeoSparseNet


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


def _build_model(args: argparse.Namespace) -> tuple[nn.Module, tuple[int, ...], tuple[int, ...]]:
    batch = int(args.batch_size)
    input_dim = int(args.input_dim)
    output_dim = int(args.output_dim)
    hidden_layers = int(args.hidden_layers)
    hidden_units = int(args.hidden_units)
    model_type = args.model.lower()

    if model_type == "psann":
        model = PSANNNet(
            input_dim,
            output_dim,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation_type=args.activation,
        )
        return model, (batch, input_dim), (batch, output_dim)

    if model_type == "dense":
        model = _build_dense_mlp(input_dim, output_dim, hidden_layers, hidden_units)
        return model, (batch, input_dim), (batch, output_dim)

    if model_type == "transformer":
        seq_len = int(args.seq_len)
        model = _TransformerBench(
            input_dim,
            output_dim,
            d_model=hidden_units,
            layers=max(1, hidden_layers),
            heads=int(args.transformer_heads),
            dropout=float(args.transformer_dropout),
        )
        return model, (batch, seq_len, input_dim), (batch, seq_len, output_dim)

    if model_type == "geosparse":
        shape = _parse_shape(args.shape)
        model = GeoSparseNet(
            input_dim=shape[0] * shape[1],
            output_dim=output_dim,
            shape=shape,
            depth=hidden_layers,
            k=int(args.k),
            activation_type=args.activation,
        )
        return model, (batch, shape[0] * shape[1]), (batch, output_dim)

    raise ValueError("model must be one of: psann, dense, transformer, geosparse")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="psann")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--input-dim", type=int, default=64)
    p.add_argument("--output-dim", type=int, default=1)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--hidden-units", type=int, default=128)
    p.add_argument("--activation", type=str, default="psann")
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--transformer-heads", type=int, default=4)
    p.add_argument("--transformer-dropout", type=float, default=0.0)
    p.add_argument("--shape", type=str, default="8x8")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--compile-backend", type=str, default=None)
    p.add_argument("--nvtx", action="store_true")
    p.add_argument("--out", type=str, default="reports/profiles/psann_profile")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    model, input_shape, target_shape = _build_model(args)
    model = model.to(device=device, dtype=dtype)
    model.train()

    if args.compile:
        model = torch.compile(model, mode=args.compile_mode, backend=args.compile_backend)

    x = torch.randn(input_shape, device=device, dtype=dtype)
    y = torch.randn(target_shape, device=device, dtype=dtype)
    loss_fn = nn.MSELoss()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
        with_modules=True,
    ) as prof:
        for _ in range(int(args.warmup_steps)):
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            model.zero_grad(set_to_none=True)

        for idx in range(int(args.steps)):
            if args.nvtx and device.type == "cuda":
                torch.cuda.nvtx.range_push(f"step_{idx}")
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            if args.nvtx and device.type == "cuda":
                torch.cuda.nvtx.range_pop()
            prof.step()

    trace_path = out_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=40)
    (out_dir / "summary.txt").write_text(summary)

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "input_shape": input_shape,
        "target_shape": target_shape,
        "compile": bool(args.compile),
        "compile_mode": args.compile_mode if args.compile else None,
        "compile_backend": args.compile_backend if args.compile else None,
        "nvtx": bool(args.nvtx),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"[profile] Wrote {trace_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
