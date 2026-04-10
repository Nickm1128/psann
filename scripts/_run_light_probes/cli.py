# ruff: noqa: F403,F405
from __future__ import annotations

from .env import (
    DATA_ROOT,
    _ensure_torch_dynamo_stub,
    configure_results_root,
    ensure_dependencies,
    get_results_root,
    maybe_extract_datasets_zip,
    pick_device,
)
from .shared import *
from .tasks import run_light_task


def run_all(tasks, seeds, epochs, device_str, pr_snapshots, match_params: bool, scale_y: bool):
    maybe_extract_datasets_zip()
    device = pick_device(device_str)
    results_root = get_results_root()
    print(f"[env] DATA_ROOT={DATA_ROOT}")
    print(f"[env] RESULTS_ROOT={results_root}")
    print(f"[env] device={device}")
    metrics_rows: List[dict] = []
    history_rows: List[dict] = []
    env_info = gather_env_info()
    env_info["selected_device"] = str(device)
    env_info["results_root"] = str(results_root)
    (results_root / "env.json").write_text(
        json.dumps(env_info, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "tasks": list(tasks),
        "seeds": list(seeds),
        "epochs": int(epochs),
        "device": str(device),
        "match_params": bool(match_params),
        "scale_y": bool(scale_y),
        "y_scaler": "standard" if scale_y else None,
        "data_root": str(DATA_ROOT),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (results_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    for task in tasks:
        print(f"[run] task={task}")
        run_light_task(
            task,
            seeds,
            device,
            epochs,
            pr_snapshots,
            match_params,
            scale_y,
            metrics_rows,
            history_rows,
        )
    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        out = results_root / "metrics.csv"
        df.to_csv(out, index=False)
        print(df.head())
        print(f"[done] Wrote metrics to {out}")
        metric_cols = [
            c
            for c in df.columns
            if c.startswith(("train_", "val_", "test_"))
            and any(key in c for key in ("mse", "rmse", "mae", "smape", "r2"))
        ]
        stability_cols = [
            c
            for c in df.columns
            if c
            in (
                "loss_curve_volatility",
                "loss_curve_mean_abs_diff",
                "val_curve_volatility",
                "grad_norm_max",
                "loss_nonfinite_steps",
                "grad_nonfinite_steps",
                "epoch_time_s_mean",
                "step_time_s_mean",
            )
        ]
        if metric_cols or stability_cols:
            agg_cols = metric_cols + stability_cols
            summary = df.groupby(["task", "model"])[agg_cols].agg(["mean", "std"])
            summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
            summary["n_runs"] = df.groupby(["task", "model"]).size()
            summary = summary.reset_index()
            summary_path = results_root / "summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[done] Wrote summary to {summary_path}")
        if history_rows:
            hist_path = results_root / "history.jsonl"
            with hist_path.open("w", encoding="utf-8") as fh:
                for row in history_rows:
                    fh.write(json.dumps(row) + "\n")
            print(f"[done] Wrote history to {hist_path}")
    else:
        print("[warn] No metrics collected")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight PSANN probe benchmarks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["jena", "beijing", "eaf"],
        help="Tasks to run (jena, beijing, eaf).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[7, 8], help="Random seeds to evaluate."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epoch budget for each model.")
    parser.add_argument("--device", default="auto", help="Device preference: auto | cpu | cuda.")
    parser.add_argument(
        "--pr-snapshots", action="store_true", help="Record Jacobian participation ratio snapshots."
    )
    parser.add_argument(
        "--match-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match MLP hidden width to PSANN parameter count.",
    )
    parser.add_argument(
        "--scale-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Standardize target values using train split (metrics stay in original scale).",
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation checks."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for metrics/log outputs (default: <repo>/colab_results_light).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.skip_deps:
        ensure_dependencies()
    _ensure_torch_dynamo_stub()
    configure_results_root(args.results_dir)
    run_all(
        args.tasks,
        args.seeds,
        args.epochs,
        args.device,
        args.pr_snapshots,
        args.match_params,
        args.scale_y,
    )
