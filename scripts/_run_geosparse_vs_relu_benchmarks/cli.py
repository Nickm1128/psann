# ruff: noqa: F403,F405
from __future__ import annotations

from .bench import evaluate_regression, fit_with_timing, warmup_fit
from .data import (
    DATASET_NAMES,
    DEFAULT_GEO_ACTIVATIONS,
    _parse_csv,
    load_dataset,
    seed_all,
    split_and_scale,
)
from .models import (
    build_dense_estimator,
    build_geosparse_estimator,
    match_dense_width_to_geosparse,
)
from .shared import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASET_NAMES))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--target-steps", type=int, default=400)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--scale-y", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--geo-depth", type=int, default=4)
    parser.add_argument("--geo-k", type=int, default=8)
    parser.add_argument(
        "--geo-activations",
        default=DEFAULT_GEO_ACTIVATIONS,
        help=(
            "Comma-separated GeoSparse activation types to benchmark "
            "(e.g. psann,relu_sigmoid_psann)."
        ),
    )
    parser.add_argument(
        "--geo-activation-config",
        default=None,
        help="Optional JSON object forwarded as GeoSparseRegressor activation config.",
    )
    parser.add_argument("--dense-depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--param-tol", type=float, default=0.01)
    parser.add_argument(
        "--amp", action="store_true", help="Enable autocast mixed precision (CUDA only)."
    )
    parser.add_argument(
        "--amp-dtype", default="bfloat16", help="Autocast dtype: bfloat16|float16|float32."
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (CUDA only).")
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--compile-fullgraph", action="store_true")
    parser.add_argument("--compile-dynamic", action="store_true")
    parser.add_argument("--warmup-samples", type=int, default=512)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument(
        "--timing-warmup-epochs",
        type=int,
        default=1,
        help="Exclude the first N epochs from timing (useful to amortize torch.compile).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_list = _parse_csv(args.datasets)
    seeds = [int(s) for s in _parse_csv(args.seeds)]
    geo_activations = _parse_csv(args.geo_activations)
    if not geo_activations:
        raise SystemExit("--geo-activations must include at least one activation type.")

    geo_activation_config: Optional[Dict[str, Any]] = None
    geo_activation_config_json: Optional[str] = None
    if args.geo_activation_config:
        try:
            parsed = json.loads(str(args.geo_activation_config))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid --geo-activation-config JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("--geo-activation-config must decode to a JSON object.")
        geo_activation_config = parsed
        geo_activation_config_json = json.dumps(parsed, sort_keys=True)

    if any(str(a).lower() == "mixed" for a in geo_activations):
        raw = geo_activation_config or {}
        types = raw.get("activation_types", raw.get("types"))
        if not isinstance(types, list) or not types:
            raise SystemExit(
                "--geo-activations includes 'mixed' but --geo-activation-config is missing "
                "'activation_types' (non-empty list)."
            )

    if args.quick:
        seeds = seeds[:1]
        args.target_steps = min(args.target_steps, 150)
        print(f"[bench] QUICK mode: seeds={seeds} target_steps={args.target_steps}")

    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "reports" / f"geosparse_vs_relu_gx10_{ts}"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for dataset_name in dataset_list:
        for seed in seeds:
            seed_all(seed)
            X, y, task = load_dataset(dataset_name, seed=seed)
            split = split_and_scale(
                X,
                y,
                seed=seed,
                test_size=0.2,
                val_size=0.1,
                scale_y=args.scale_y,
            )
            input_dim = split.X_train.shape[1]
            for geo_activation in geo_activations:
                geo_activation = str(geo_activation).strip()
                geo_activation_key = geo_activation.lower().replace("-", "_")
                match = match_dense_width_to_geosparse(
                    input_dim=input_dim,
                    output_dim=1,
                    geo_depth=args.geo_depth,
                    geo_k=args.geo_k,
                    geo_shape=None,
                    geo_activation_type=geo_activation,
                    geo_activation_config_json=geo_activation_config_json,
                    dense_depth=args.dense_depth,
                    tol=args.param_tol,
                )

                def geo_factory(epochs: int) -> GeoSparseRegressor:
                    return build_geosparse_estimator(
                        input_dim=input_dim,
                        shape=None,
                        geo_depth=args.geo_depth,
                        geo_k=args.geo_k,
                        activation_type=geo_activation,
                        activation_config=geo_activation_config,
                        amp=bool(args.amp),
                        amp_dtype=str(args.amp_dtype),
                        compile=bool(args.compile),
                        compile_backend=str(args.compile_backend),
                        compile_mode=str(args.compile_mode),
                        compile_fullgraph=bool(args.compile_fullgraph),
                        compile_dynamic=bool(args.compile_dynamic),
                        device=args.device,
                        seed=seed,
                        epochs=epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                    )

                def dense_factory(epochs: int) -> PSANNRegressor:
                    return build_dense_estimator(
                        dense_depth=match["dense_depth"],
                        dense_width=match["dense_width"],
                        amp=bool(args.amp),
                        amp_dtype=str(args.amp_dtype),
                        compile=bool(args.compile),
                        compile_backend=str(args.compile_backend),
                        compile_mode=str(args.compile_mode),
                        compile_fullgraph=bool(args.compile_fullgraph),
                        compile_dynamic=bool(args.compile_dynamic),
                        device=args.device,
                        seed=seed,
                        epochs=epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                    )

                warmup_fit(
                    geo_factory,
                    split.X_train,
                    split.y_train,
                    split.X_val,
                    split.y_val,
                    warmup_samples=args.warmup_samples,
                    warmup_epochs=args.warmup_epochs,
                )
                warmup_fit(
                    dense_factory,
                    split.X_train,
                    split.y_train,
                    split.X_val,
                    split.y_val,
                    warmup_samples=args.warmup_samples,
                    warmup_epochs=args.warmup_epochs,
                )

                for model_name, factory in (
                    (f"geosparse_{geo_activation_key}", geo_factory),
                    (f"dense_relu_match_{geo_activation_key}", dense_factory),
                ):
                    timing = fit_with_timing(
                        factory,
                        split.X_train,
                        split.y_train,
                        split.X_val,
                        split.y_val,
                        batch_size=args.batch_size,
                        target_steps=args.target_steps,
                        progress_every_steps=args.progress_every,
                        timing_warmup_epochs=(
                            int(args.timing_warmup_epochs) if bool(args.compile) else 0
                        ),
                    )
                    metrics = evaluate_regression(timing["model"], split)
                    is_geo_model = str(model_name).startswith("geosparse_")
                    results.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "seed": seed,
                            "task": task,
                            "geo_activation": geo_activation,
                            "params": (
                                match["target_params"] if is_geo_model else match["dense_params"]
                            ),
                            "geo_depth": int(args.geo_depth),
                            "geo_k": int(args.geo_k),
                            "dense_depth": int(match["dense_depth"]),
                            "dense_width": int(match["dense_width"]),
                            "target_params": int(match["target_params"]),
                            "dense_params": int(match["dense_params"]),
                            "rel_mismatch": match["rel_mismatch"],
                            **metrics,
                            "train_time_s": timing["train_time_s"],
                            "train_time_total_s": timing["train_time_total_s"],
                            "steps_per_sec": timing["steps_per_sec"],
                            "samples_per_sec": timing["samples_per_sec"],
                            "epochs": timing["epochs"],
                            "warmup_epochs": timing["warmup_epochs"],
                            "total_steps": timing["total_steps"],
                            "timed_steps": timing["timed_steps"],
                        }
                    )

    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    summary = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in results:
        grouped[(row["dataset"], row["model"])].append(row)
    for (dataset, model), rows in grouped.items():
        mse = np.array([r["mse"] for r in rows], dtype=np.float64)
        mae = np.array([r["mae"] for r in rows], dtype=np.float64)
        r2 = np.array([r["r2"] for r in rows], dtype=np.float64)
        time_s = np.array([r["train_time_s"] for r in rows], dtype=np.float64)
        time_total_s = np.array([r["train_time_total_s"] for r in rows], dtype=np.float64)
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "mse_mean": float(mse.mean()),
                "mse_std": float(mse.std(ddof=0)),
                "mae_mean": float(mae.mean()),
                "mae_std": float(mae.std(ddof=0)),
                "r2_mean": float(r2.mean()),
                "r2_std": float(r2.std(ddof=0)),
                "train_time_mean": float(time_s.mean()),
                "train_time_std": float(time_s.std(ddof=0)),
                "train_time_total_mean": float(time_total_s.mean()),
                "train_time_total_std": float(time_total_s.std(ddof=0)),
                "seeds": [r["seed"] for r in rows],
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)

    print(f"[bench] Wrote results -> {results_path}")
    print(f"[bench] Wrote summary -> {summary_path}")
    print(f"[bench] Wrote summary CSV -> {csv_path}")
    return 0
