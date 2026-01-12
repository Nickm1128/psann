#!/usr/bin/env python
"""Run a sweep of GeoSparse vs dense benchmarks and aggregate results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _slugify(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(",", "_")
        .replace("=", "_")
    )


def _run_command(cmd: List[str], *, dry_run: bool) -> subprocess.CompletedProcess:
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _load_results(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _flatten(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                flat[f"{prefix}{key}_{sub_k}"] = sub_v
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _write_summary(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def _aggregate_by_model(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        metrics = agg.setdefault(model, {})
        counts[model] = counts.get(model, 0) + 1
        for key in ("mse_test", "train_train_time_s", "train_step_time_ms_mean", "train_samples_per_sec"):
            val = row.get(key)
            if isinstance(val, (int, float)):
                metrics[key] = float(metrics.get(key, 0.0) + float(val))
    for model, metrics in agg.items():
        denom = float(max(1, counts.get(model, 1)))
        for key, value in metrics.items():
            metrics[key] = float(value) / denom
    return agg


def _maybe_plot(summary: Dict[str, Dict[str, float]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[warn] matplotlib not available; skipping plots.")
        return

    models = sorted(summary.keys())
    if not models:
        return

    mean_times = [summary[m].get("train_train_time_s", 0.0) for m in models]
    mean_mse = [summary[m].get("mse_test", 0.0) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(models, mean_times)
    axes[0].set_title("Mean Train Time (s)")
    axes[0].set_ylabel("seconds")
    axes[1].bar(models, mean_mse)
    axes[1].set_title("Mean Test MSE")
    axes[1].set_ylabel("mse")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_plot.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shapes", type=str, default="4x4,8x8", help="Comma list of HxW shapes.")
    p.add_argument("--depths", type=str, default="4,8", help="Comma list of depths.")
    p.add_argument("--ks", type=str, default="4,8,16", help="Comma list of k values.")
    p.add_argument("--activations", type=str, default="relu,psann", help="Comma list of activations.")
    p.add_argument("--seeds", type=str, default="0,1", help="Comma list of seeds.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", type=str, default="bfloat16")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-backend", type=str, default="inductor")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--dense-depth", type=int, default=None)
    p.add_argument("--dense-max-width", type=int, default=4096)
    p.add_argument("--match-tolerance", type=float, default=0.01)
    p.add_argument("--train-size", type=int, default=4096)
    p.add_argument("--test-size", type=int, default=1024)
    p.add_argument("--pattern", type=str, default="local")
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--wrap-mode", type=str, default="clamp")
    p.add_argument("--activation-config", type=str, default=None)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shapes = _parse_list(args.shapes)
    depths = [int(v) for v in _parse_list(args.depths)]
    ks = [int(v) for v in _parse_list(args.ks)]
    activations = _parse_list(args.activations)
    seeds = [int(v) for v in _parse_list(args.seeds)]

    out_root = (
        Path(args.out)
        if args.out
        else Path("reports") / "geo_sparse_sweep" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for shape in shapes:
        for depth in depths:
            for k in ks:
                for activation in activations:
                    for seed in seeds:
                        run_id = f"shape={shape}_depth={depth}_k={k}_act={activation}_seed={seed}"
                        slug = _slugify(run_id)
                        run_dir = out_root / slug
                        results_path = run_dir / "results.json"
                        if args.resume and results_path.exists():
                            payload = _load_results(results_path)
                            if payload is not None:
                                rows.extend(_rows_from_results(payload, run_id, run_dir))
                            continue

                        cmd = [
                            sys.executable,
                            "scripts/benchmark_geo_sparse_vs_dense.py",
                            "--shape",
                            shape,
                            "--depth",
                            str(depth),
                            "--k",
                            str(k),
                            "--sparse-activation",
                            activation,
                            "--seed",
                            str(seed),
                            "--device",
                            args.device,
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
                            "--dense-max-width",
                            str(args.dense_max_width),
                            "--match-tolerance",
                            str(args.match_tolerance),
                            "--train-size",
                            str(args.train_size),
                            "--test-size",
                            str(args.test_size),
                            "--pattern",
                            str(args.pattern),
                            "--radius",
                            str(args.radius),
                            "--wrap-mode",
                            str(args.wrap_mode),
                            "--out",
                            str(run_dir),
                        ]
                        if args.dense_depth is not None:
                            cmd.extend(["--dense-depth", str(args.dense_depth)])
                        if args.amp:
                            cmd.append("--amp")
                        if args.tf32:
                            cmd.append("--tf32")
                        if args.compile:
                            cmd.append("--compile")
                        if args.activation_config is not None:
                            cmd.extend(["--activation-config", str(args.activation_config)])

                        result = _run_command(cmd, dry_run=bool(args.dry_run))
                        if result.returncode != 0:
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "status": "error",
                                    "returncode": int(result.returncode),
                                    "stderr": result.stderr.strip(),
                                }
                            )
                            continue
                        payload = _load_results(results_path)
                        if payload is None:
                            rows.append(
                                {
                                    "run_id": run_id,
                                    "status": "error",
                                    "returncode": 0,
                                    "stderr": "missing results.json",
                                }
                            )
                            continue
                        rows.extend(_rows_from_results(payload, run_id, run_dir))

    _write_summary(rows, out_root / "summary.csv")
    summary_by_model = _aggregate_by_model(rows)
    (out_root / "summary_by_model.json").write_text(
        json.dumps(summary_by_model, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.plot:
        _maybe_plot(summary_by_model, out_root)
    print(f"Wrote sweep results to {out_root}")


def _rows_from_results(
    payload: Dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    env = payload.get("environment", {})
    manifest = payload.get("manifest", {})
    for model in payload.get("models", []):
        row: Dict[str, Any] = {
            "run_id": run_id,
            "model": model.get("name"),
            "status": "ok",
            "result_dir": str(run_dir),
        }
        row.update(_flatten("env_", env))
        row.update(_flatten("manifest_", manifest))
        row.update(
            {
                "params_empirical": model.get("params_empirical"),
                "params_trainable": model.get("params_trainable"),
                "params_analytic": model.get("params_analytic"),
                "param_mismatch": model.get("param_mismatch"),
                "param_mismatch_ratio": model.get("param_mismatch_ratio"),
                "mse_test": model.get("mse_test"),
            }
        )
        if isinstance(model.get("train"), dict):
            row.update(_flatten("train_", model["train"]))
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
