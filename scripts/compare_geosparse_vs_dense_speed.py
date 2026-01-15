#!/usr/bin/env python3
"""Run a small benchmark on GeoSparse vs dense ReLU and report throughput.

This is a convenience wrapper around `scripts/benchmark_geo_sparse_vs_dense.py` that:
- runs one or more seeds
- parses `results.json`
- prints per-model train time and samples/sec
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_list_int(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _is_cuda_device(device: str) -> bool:
    key = str(device).strip().lower()
    return key == "cuda" or key.startswith("cuda:")


def _cuda_preflight(device: str) -> None:
    if not _is_cuda_device(device):
        return
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise SystemExit(f"--device {device} requested but torch import failed: {exc}") from exc

    if torch.cuda.is_available():
        return

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    hint = (
        "CUDA requested but no GPU is visible to PyTorch.\n"
        f"- torch={getattr(torch, '__version__', '?')}\n"
        f"- CUDA_VISIBLE_DEVICES={cuda_visible!r}\n\n"
        "In Docker, this usually means the container was started without GPU access.\n"
        "Check inside the container:\n"
        "  nvidia-smi\n"
        "  python3 -c \"import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())\"\n"
        "Then restart the container with `--gpus all` (or a working NVIDIA runtime).\n"
    )

    try:
        proc = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            hint += f"\n`nvidia-smi -L`:\n{proc.stdout.strip()}\n"
        else:
            hint += f"\n`nvidia-smi -L` failed (rc={proc.returncode}):\n{proc.stderr.strip()}\n"
    except FileNotFoundError:
        hint += "\n`nvidia-smi` not found in PATH.\n"

    raise SystemExit(hint)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=str, default="16x16")
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--task", type=str, default="teacher_relu")
    p.add_argument("--sparse-activation", type=str, default="relu")
    p.add_argument("--activation-config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-size", type=int, default=8192)
    p.add_argument("--test-size", type=int, default=2048)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="bfloat16")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-backend", type=str, default="inductor")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--timing-warmup-steps", type=int, default=5)
    p.add_argument("--timing-epochs", type=int, default=1)
    p.add_argument("--scale-x", action="store_true")
    p.add_argument("--scale-y", action="store_true")
    p.add_argument("--seeds", type=str, default="0,1")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def _run_one(*, args: argparse.Namespace, seed: int, out_dir: Path) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/benchmark_geo_sparse_vs_dense.py",
        "--shape",
        str(args.shape),
        "--depth",
        str(args.depth),
        "--k",
        str(args.k),
        "--sparse-activation",
        str(args.sparse_activation),
        "--task",
        str(args.task),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--dtype",
        str(args.dtype),
        "--amp-dtype",
        str(args.amp_dtype),
        "--compile-backend",
        str(args.compile_backend),
        "--compile-mode",
        str(args.compile_mode),
        "--timing-warmup-steps",
        str(args.timing_warmup_steps),
        "--timing-epochs",
        str(args.timing_epochs),
        "--train-size",
        str(args.train_size),
        "--test-size",
        str(args.test_size),
        "--out",
        str(out_dir),
    ]
    if args.activation_config is not None:
        cmd.extend(["--activation-config", str(args.activation_config)])
    if args.amp:
        cmd.append("--amp")
    if args.tf32:
        cmd.append("--tf32")
    if args.compile:
        cmd.append("--compile")
    if args.scale_x:
        cmd.append("--scale-x")
    if args.scale_y:
        cmd.append("--scale-y")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "src")
    print("[run]", " ".join(cmd))
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Benchmark failed (rc={proc.returncode}). See {out_dir}/stderr.log for details."
        )
    results_path = out_dir / "results.json"
    if not results_path.exists():
        raise RuntimeError(f"Missing results.json in {out_dir}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _model_summary(results: Dict[str, Any], model_name: str) -> Optional[Dict[str, float]]:
    for entry in results.get("models", []):
        if entry.get("name") != model_name:
            continue
        train = entry.get("train", {}) if isinstance(entry.get("train"), dict) else {}
        return {
            "mse_test": float(entry.get("mse_test")),
            "train_time_s": float(train.get("train_time_s")),
            "samples_per_sec": float(train.get("samples_per_sec")),
        }
    return None


def main() -> None:
    args = _parse_args()
    seeds = _parse_list_int(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer seed.")

    _cuda_preflight(str(args.device))

    out_root = (
        Path(args.out)
        if args.out
        else Path("reports")
        / "geo_sparse_speed"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    geo_sps: List[float] = []
    dense_sps: List[float] = []
    geo_time: List[float] = []
    dense_time: List[float] = []

    for seed in seeds:
        run_dir = out_root / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        results = _run_one(args=args, seed=seed, out_dir=run_dir)
        env = results.get("environment", {})
        print(
            f"[env] device={env.get('device')} torch={env.get('torch_version')} cuda={env.get('cuda_available')} gpu={env.get('gpu_name')}"
        )

        geo = _model_summary(results, "geo_sparse")
        dense = _model_summary(results, "dense_resrelu")
        if geo is None or dense is None:
            raise RuntimeError("results.json missing geo_sparse or dense_resrelu entries")

        geo_sps.append(geo["samples_per_sec"])
        dense_sps.append(dense["samples_per_sec"])
        geo_time.append(geo["train_time_s"])
        dense_time.append(dense["train_time_s"])

        ratio = geo["samples_per_sec"] / max(1e-9, dense["samples_per_sec"])
        print(
            f"[seed {seed}] geo_sparse: {geo['samples_per_sec']:.0f} samp/s, {geo['train_time_s']:.2f}s, mse_test={geo['mse_test']:.4g}"
        )
        print(
            f"[seed {seed}] dense_resrelu: {dense['samples_per_sec']:.0f} samp/s, {dense['train_time_s']:.2f}s, mse_test={dense['mse_test']:.4g}"
        )
        print(f"[seed {seed}] speedup (geo/dense): {ratio:.3f}x\n")

    print("[summary]")
    print(f"- runs: {len(seeds)} -> {out_root}")
    print(
        f"- geo_sparse: {statistics.mean(geo_sps):.0f}±{statistics.pstdev(geo_sps):.0f} samp/s, "
        f"{statistics.mean(geo_time):.2f}±{statistics.pstdev(geo_time):.2f}s train_time_s"
    )
    print(
        f"- dense_resrelu: {statistics.mean(dense_sps):.0f}±{statistics.pstdev(dense_sps):.0f} samp/s, "
        f"{statistics.mean(dense_time):.2f}±{statistics.pstdev(dense_time):.2f}s train_time_s"
    )
    print(
        f"- mean speedup (geo/dense): {statistics.mean(g/d for g,d in zip(geo_sps,dense_sps)):.3f}x"
    )


if __name__ == "__main__":
    main()
