# ruff: noqa: F403,F405
from __future__ import annotations

from .data import DATASETS
from .io import (
    _aggregate_seed_summary,
    _default_output_dir,
    _flatten_result,
    _load_existing_run_ids,
    _load_results,
    _write_summary,
    parse_args,
)
from .models import MODELS
from .runner import _run_single
from .shared import *


def main() -> None:
    args = parse_args()

    if args.list_datasets:
        for key in sorted(DATASETS):
            spec = DATASETS[key]
            print(f"{key} ({spec.task}, {spec.kind})")
        raise SystemExit(0)
    if args.list_models:
        for key in sorted(MODELS):
            print(key)
        raise SystemExit(0)

    datasets = parse_comma_list(args.datasets)
    models = parse_comma_list(args.models)
    seeds = [int(s) for s in parse_comma_list(args.seeds)]
    if not datasets or not models or not seeds:
        raise ValueError("datasets, models, and seeds must be non-empty.")

    out_dir = Path(args.out) if args.out else _default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    existing = _load_existing_run_ids(results_path) if args.resume else set()

    manifest = {
        "datasets": datasets,
        "models": models,
        "seeds": seeds,
        "device": args.device,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "val_fraction": float(args.val_fraction),
        "scale_y": bool(args.scale_y),
        "target_scaler": "standard" if args.scale_y else None,
        "save_models": bool(args.save_models),
        "save_preds": bool(args.save_preds),
        "resume": bool(args.resume),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    env_info = gather_env_info()
    env_info["selected_device"] = str(args.device)
    (out_dir / "env.json").write_text(
        json.dumps(env_info, indent=2, sort_keys=True), encoding="utf-8"
    )

    for dataset_name in datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset_name}'.")
        dataset_spec = DATASETS[dataset_name]
        for seed in seeds:
            dataset = dataset_spec.builder(seed)
            for model_name in models:
                if model_name not in MODELS:
                    raise ValueError(f"Unknown model '{model_name}'.")
                model_spec = MODELS[model_name]
                run_id = f"{dataset.name}:{model_spec.name}:seed{seed}"
                if run_id in existing:
                    continue
                slug = slugify(run_id, colon="__")
                record: Dict[str, Any] = {
                    "run_id": run_id,
                    "dataset": dataset.name,
                    "task": dataset.task,
                    "kind": dataset.kind,
                    "model": model_spec.name,
                    "seed": seed,
                    "device": args.device,
                    "train_size": int(dataset.X_train.shape[0]),
                    "test_size": int(dataset.X_test.shape[0]),
                    "input_shape": list(dataset.X_train.shape[1:]),
                }
                record.update({f"meta_{k}": v for k, v in dataset.meta.items()})
                model_path = out_dir / "models" / f"{slug}.pt" if args.save_models else None
                preds_path = out_dir / "preds" / f"{slug}.npz" if args.save_preds else None
                try:
                    result = _run_single(
                        model_spec,
                        dataset,
                        seed=seed,
                        device=args.device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        val_fraction=args.val_fraction,
                        scale_y=bool(args.scale_y),
                        save_model_path=model_path,
                        save_preds_path=preds_path,
                    )
                    record.update(result)
                    record["status"] = "ok"
                    if model_path is not None:
                        record["model_path"] = str(model_path)
                    if preds_path is not None:
                        record["preds_path"] = str(preds_path)
                except Exception as exc:
                    record["status"] = "error"
                    record["error"] = str(exc)
                    record["traceback"] = traceback.format_exc()

                with results_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")

    results = _load_results(results_path)
    summary_rows = [_flatten_result(entry) for entry in results]
    if summary_rows:
        _write_summary(summary_rows, out_dir / "summary.csv")
        seed_summary_rows = _aggregate_seed_summary(summary_rows)
        if seed_summary_rows:
            _write_summary(seed_summary_rows, out_dir / "seed_summary.csv")
    print(f"Wrote results to {results_path}")
