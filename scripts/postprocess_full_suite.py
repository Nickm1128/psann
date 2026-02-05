"""Postprocess full-suite outputs into compact tables/plots.

Example:
  python scripts/postprocess_full_suite.py --run reports/full_suite/20260205_194552
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(row: Dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _summarize_light_probes(run_dir: Path, out_dir: Path) -> None:
    summary_path = run_dir / "light_probes" / "summary.csv"
    if not summary_path.exists():
        return
    rows = _read_csv(summary_path)
    by_task: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], {})[row["model"]] = row

    out_rows: List[Dict[str, Any]] = []
    for task in sorted(by_task):
        models = by_task[task]
        if "psann_conv" not in models or "mlp" not in models:
            continue
        psann = models["psann_conv"]
        mlp = models["mlp"]
        psann_rmse = float(psann["test_rmse_mean"])
        mlp_rmse = float(mlp["test_rmse_mean"])
        rmse_delta = mlp_rmse - psann_rmse
        rmse_improve_pct = (rmse_delta / mlp_rmse) * 100.0 if mlp_rmse else 0.0
        out_rows.append(
            {
                "task": task,
                "psann_test_rmse": psann_rmse,
                "mlp_test_rmse": mlp_rmse,
                "rmse_delta_mlp_minus_psann": rmse_delta,
                "rmse_improve_pct_psann_vs_mlp": rmse_improve_pct,
                "psann_test_r2": float(psann["test_r2_mean"]),
                "mlp_test_r2": float(mlp["test_r2_mean"]),
                "psann_epoch_time_s": float(psann["epoch_time_s_mean_mean"]),
                "mlp_epoch_time_s": float(mlp["epoch_time_s_mean_mean"]),
                "n_runs": int(psann.get("n_runs", 0)),
            }
        )

    if out_rows:
        _write_csv(
            out_dir / "light_probes_rmse_r2.csv",
            out_rows,
            list(out_rows[0].keys()),
        )


def _summarize_ablations(run_dir: Path, out_dir: Path) -> None:
    summary_path = run_dir / "ablations" / "seed_summary.csv"
    if not summary_path.exists():
        return
    rows = _read_csv(summary_path)
    classification = {"classification_clusters", "context_rotating_moons"}

    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        dataset = row["dataset"]
        model = row["model"]
        if dataset in classification:
            metric = _as_float(row, "metrics_test_accuracy_mean")
            if metric is None:
                continue
            cur = best.get(dataset)
            if cur is None or metric > cur["metric"]:
                best[dataset] = {
                    "dataset": dataset,
                    "kind": "classification",
                    "model": model,
                    "metric": metric,
                    "test_accuracy_mean": metric,
                    "test_macro_f1_mean": _as_float(row, "metrics_test_macro_f1_mean"),
                }
        else:
            metric = _as_float(row, "metrics_test_mse_mean")
            if metric is None:
                continue
            cur = best.get(dataset)
            if cur is None or metric < cur["metric"]:
                best[dataset] = {
                    "dataset": dataset,
                    "kind": "regression",
                    "model": model,
                    "metric": metric,
                    "test_mse_mean": metric,
                    "test_rmse_mean": _as_float(row, "metrics_test_rmse_mean"),
                    "test_r2_mean": _as_float(row, "metrics_test_r2_mean"),
                }

    if best:
        fieldnames = [
            "dataset",
            "kind",
            "model",
            "metric",
            "test_mse_mean",
            "test_rmse_mean",
            "test_r2_mean",
            "test_accuracy_mean",
            "test_macro_f1_mean",
        ]
        _write_csv(out_dir / "ablations_best_by_dataset.csv", list(best.values()), fieldnames)

    model_reg: Dict[str, List[float]] = {}
    model_cls: Dict[str, List[float]] = {}
    for row in rows:
        model = row["model"]
        dataset = row["dataset"]
        if dataset in classification:
            acc = _as_float(row, "metrics_test_accuracy_mean")
            if acc is None:
                continue
            model_cls.setdefault(model, []).append(acc)
        else:
            mse = _as_float(row, "metrics_test_mse_mean")
            if mse is None:
                continue
            model_reg.setdefault(model, []).append(mse)

    reg_rows = [
        {"model": model, "avg_test_mse": sum(vals) / len(vals), "datasets": len(vals)}
        for model, vals in sorted(model_reg.items(), key=lambda x: sum(x[1]) / len(x[1]))
    ]
    if reg_rows:
        _write_csv(
            out_dir / "ablations_avg_mse_by_model.csv",
            reg_rows,
            ["model", "avg_test_mse", "datasets"],
        )

    cls_rows = [
        {"model": model, "avg_test_accuracy": sum(vals) / len(vals), "datasets": len(vals)}
        for model, vals in sorted(model_cls.items(), key=lambda x: -sum(x[1]) / len(x[1]))
    ]
    if cls_rows:
        _write_csv(
            out_dir / "ablations_avg_accuracy_by_model.csv",
            cls_rows,
            ["model", "avg_test_accuracy", "datasets"],
        )


def _summarize_geo_sparse(run_dir: Path, out_dir: Path) -> None:
    bench = run_dir / "geo_sparse" / "summary.csv"
    if bench.exists():
        out_path = out_dir / "geo_sparse_bench_summary.csv"
        out_path.write_text(bench.read_text(encoding="utf-8"), encoding="utf-8")

    sweep = run_dir / "geo_sparse_sweep" / "summary_by_model.json"
    if sweep.exists():
        payload = json.loads(sweep.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        for model, metrics in payload.items():
            row = {"model": model}
            row.update(metrics)
            rows.append(row)
        if rows:
            fieldnames = sorted(rows[0].keys())
            _write_csv(out_dir / "geo_sparse_sweep_summary.csv", rows, fieldnames)

    micro = run_dir / "geo_sparse_micro" / "summary.csv"
    if micro.exists():
        out_path = out_dir / "geo_sparse_micro_summary.csv"
        out_path.write_text(micro.read_text(encoding="utf-8"), encoding="utf-8")


def _plot_bundle(out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        (out_dir / "plot_note.txt").write_text(f"Plot generation skipped: {exc}\n", encoding="utf-8")
        return

    lp_path = out_dir / "light_probes_rmse_r2.csv"
    if lp_path.exists():
        rows = _read_csv(lp_path)
        tasks = [r["task"] for r in rows]
        psann = [float(r["psann_test_rmse"]) for r in rows]
        mlp = [float(r["mlp_test_rmse"]) for r in rows]
        x = range(len(tasks))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([i - width / 2 for i in x], psann, width, label="psann_conv")
        ax.bar([i + width / 2 for i in x], mlp, width, label="mlp")
        ax.set_xticks(list(x))
        ax.set_xticklabels(tasks, rotation=15)
        ax.set_ylabel("Test RMSE")
        ax.set_title("Light Probes: Test RMSE (lower is better)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "light_probes_test_rmse.png", dpi=160)
        plt.close(fig)

    gs_path = out_dir / "geo_sparse_sweep_summary.csv"
    if gs_path.exists():
        rows = _read_csv(gs_path)
        models = [r["model"] for r in rows]
        mse = [float(r["mse_test"]) for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(models, mse, color=["#4C78A8", "#F58518", "#54A24B"])
        ax.set_ylabel("Mean Test MSE")
        ax.set_title("GeoSparse Sweep: Mean Test MSE")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out_dir / "geo_sparse_sweep_mse.png", dpi=160)
        plt.close(fig)

    ab_path = out_dir / "ablations_avg_mse_by_model.csv"
    if ab_path.exists():
        rows = _read_csv(ab_path)
        models = [r["model"] for r in rows]
        mse = [float(r["avg_test_mse"]) for r in rows]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(models, mse)
        ax.set_yscale("log")
        ax.set_ylabel("Avg Test MSE (log scale)")
        ax.set_title("Ablations: Avg Test MSE by Model (Regression Datasets)")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(out_dir / "ablations_avg_mse_log.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Postprocess full-suite outputs.")
    ap.add_argument("--run", type=str, required=True, help="Path to a full_suite run directory.")
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory (default: <run>/analysis).",
    )
    ap.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    out_dir = Path(args.out) if args.out else run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    _summarize_light_probes(run_dir, out_dir)
    _summarize_ablations(run_dir, out_dir)
    _summarize_geo_sparse(run_dir, out_dir)
    if not args.no_plots:
        _plot_bundle(out_dir)


if __name__ == "__main__":
    main()
