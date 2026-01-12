#!/usr/bin/env python
"""Microbenchmark GeoSparse layers and blocks (forward/backward)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from psann.layers.geo_sparse import GeoSparseLinear, build_geo_connectivity
from psann.nn_geo_sparse import GeoSparseResidualBlock
from psann.utils import choose_device, seed_all


def _parse_shape(text: str) -> Tuple[int, int]:
    for sep in ("x", "X", ","):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    raise ValueError("shape must be formatted as HxW or H,W")


def _set_precision(tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    if tf32:
        torch.set_float32_matmul_precision("high")


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _make_autocast_context(device: torch.device, amp: bool, amp_dtype: torch.dtype):
    if not amp:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)


def _get_env_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32
        if torch.cuda.is_available()
        else None,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
        "matmul_precision": torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else None,
        "torch_compile_available": hasattr(torch, "compile"),
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = torch.cuda.get_device_capability(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info["gpu_mem_total_bytes"] = int(total_mem)
            info["gpu_mem_free_bytes"] = int(free_mem)
        except Exception:
            pass
    return info


def _time_forward(
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    warmup: int,
    amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            with _make_autocast_context(device, amp, amp_dtype):
                _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(steps):
                with _make_autocast_context(device, amp, amp_dtype):
                    _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            return float(start_event.elapsed_time(end_event)) / float(steps)
        start = time.perf_counter()
        for _ in range(steps):
            with _make_autocast_context(device, amp, amp_dtype):
                _ = model(x)
        return (time.perf_counter() - start) * 1000.0 / float(steps)


def _time_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    warmup: int,
    amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    use_scaler = bool(amp and amp_dtype == torch.float16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        with _make_autocast_context(device, amp, amp_dtype):
            out = model(x)
            loss = out.sum()
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with _make_autocast_context(device, amp, amp_dtype):
                out = model(x)
                loss = out.sum()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        end_event.record()
        torch.cuda.synchronize()
        return float(start_event.elapsed_time(end_event)) / float(steps)

    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with _make_autocast_context(device, amp, amp_dtype):
            out = model(x)
            loss = out.sum()
        loss.backward()
        optimizer.step()
    return (time.perf_counter() - start) * 1000.0 / float(steps)


def _bench_model(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    *,
    device: torch.device,
    steps: int,
    warmup: int,
    amp: bool,
    amp_dtype: torch.dtype,
    compile_model: bool,
    compile_backend: Optional[str],
    compile_mode: Optional[str],
) -> Dict[str, Any]:
    model = model.to(device=device)
    if compile_model:
        model = torch.compile(model, backend=compile_backend, mode=compile_mode)
    fwd_ms = _time_forward(
        model,
        x,
        device=device,
        steps=steps,
        warmup=warmup,
        amp=amp,
        amp_dtype=amp_dtype,
    )
    fwd_bwd_ms = _time_forward_backward(
        model,
        x,
        device=device,
        steps=steps,
        warmup=warmup,
        amp=amp,
        amp_dtype=amp_dtype,
    )
    return {"name": name, "forward_ms": fwd_ms, "forward_backward_ms": fwd_bwd_ms}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=str, default="8x8")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--pattern", type=str, default="local", choices=["local", "random", "hash"])
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--wrap-mode", type=str, default="clamp", choices=["clamp", "wrap"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-backend", type=str, default="inductor")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--compute-mode", type=str, default="gather", choices=["gather", "scatter"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(int(args.seed))

    shape = _parse_shape(args.shape)
    features = shape[0] * shape[1]

    device = choose_device(args.device)
    _set_precision(bool(args.tf32))
    dtype = _resolve_dtype(args.dtype)
    amp_dtype = _resolve_dtype(args.amp_dtype)
    if device.type == "cpu" and amp_dtype == torch.float16:
        amp_dtype = torch.bfloat16

    indices = build_geo_connectivity(
        shape,
        k=int(args.k),
        pattern=str(args.pattern),
        radius=int(args.radius),
        wrap_mode=str(args.wrap_mode),
        seed=int(args.seed),
    ).to(device=device)

    layer = GeoSparseLinear(
        features,
        features,
        indices,
        bias=True,
        compute_mode=str(args.compute_mode),
    ).to(dtype=dtype)
    block = GeoSparseResidualBlock(
        features,
        indices,
        activation_type="relu",
        norm="rms",
        drop_path=0.0,
        residual_alpha_init=0.0,
        bias=True,
        compute_mode=str(args.compute_mode),
    ).to(dtype=dtype)

    x = torch.randn(int(args.batch_size), features, device=device, dtype=dtype)

    results = [
        _bench_model(
            "geo_sparse_linear",
            layer,
            x,
            device=device,
            steps=int(args.steps),
            warmup=int(args.warmup),
            amp=bool(args.amp),
            amp_dtype=amp_dtype,
            compile_model=bool(args.compile),
            compile_backend=args.compile_backend,
            compile_mode=args.compile_mode,
        ),
        _bench_model(
            "geo_sparse_block",
            block,
            x,
            device=device,
            steps=int(args.steps),
            warmup=int(args.warmup),
            amp=bool(args.amp),
            amp_dtype=amp_dtype,
            compile_model=bool(args.compile),
            compile_backend=args.compile_backend,
            compile_mode=args.compile_mode,
        ),
    ]

    out_dir = Path(args.out) if args.out else Path("reports") / "geo_sparse_micro" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "shape": list(shape),
        "features": features,
        "k": int(args.k),
        "pattern": str(args.pattern),
        "radius": int(args.radius),
        "wrap_mode": str(args.wrap_mode),
        "batch_size": int(args.batch_size),
        "steps": int(args.steps),
        "warmup": int(args.warmup),
        "device": str(device),
        "dtype": str(args.dtype),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "amp_dtype_effective": str(amp_dtype),
        "tf32": bool(args.tf32),
        "compile": bool(args.compile),
        "compile_backend": str(args.compile_backend),
        "compile_mode": str(args.compile_mode),
        "compute_mode": str(args.compute_mode),
        "seed": int(args.seed),
    }

    payload = {
        "environment": _get_env_info(device),
        "manifest": manifest,
        "results": results,
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary_rows = []
    for entry in results:
        summary_rows.append(
            {
                "model": entry["name"],
                "forward_ms": entry["forward_ms"],
                "forward_backward_ms": entry["forward_backward_ms"],
            }
        )
    (out_dir / "summary.csv").write_text(
        "model,forward_ms,forward_backward_ms\n"
        + "\n".join(
            f"{row['model']},{row['forward_ms']},{row['forward_backward_ms']}"
            for row in summary_rows
        ),
        encoding="utf-8",
    )

    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
