# ruff: noqa: F403,F405
from __future__ import annotations

from .data import (
    DatasetBundle,
    _build_dataset,
    _fit_standard_scaler,
    _parse_shape,
    _print_header,
    _train_val_split,
)
from .models import (
    DenseResidualNet,
    _dense_residual_params_with_activation,
    _match_dense_residual_width_with_activation,
)
from .shared import *
from .train import (
    _build_loader,
    _evaluate_metrics,
    _flatten_dict,
    _get_env_info,
    _resolve_dtype,
    _set_precision,
    _train_model,
    _write_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=str, default="4x4", help="Layer shape HxW.")
    p.add_argument("--depth", type=int, default=4, help="Sparse depth (blocks).")
    p.add_argument("--k", type=int, default=8, help="Sparse fan-in per output.")
    p.add_argument("--pattern", type=str, default="local", choices=["local", "random", "hash"])
    p.add_argument("--radius", type=int, default=1, help="Neighborhood radius for connectivity.")
    p.add_argument("--wrap-mode", type=str, default="clamp", choices=["clamp", "wrap"])
    p.add_argument(
        "--sparse-activation",
        type=str,
        default="psann",
        help=("GeoSparse activation type (e.g. psann, relu, tanh, mixed, " "relu_sigmoid_psann)."),
    )
    p.add_argument(
        "--activation-config",
        type=str,
        default=None,
        help="JSON string for activation config (optional).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="mixed",
        choices=["sine", "mixed", "teacher_relu", "teacher_tanh"],
        help="Synthetic regression task used to generate training data.",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"]
    )
    p.add_argument("--amp", action="store_true", help="Enable autocast mixed precision.")
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Autocast dtype.",
    )
    p.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on CUDA.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile.")
    p.add_argument("--compile-backend", type=str, default="inductor")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--timing-warmup-steps", type=int, default=2)
    p.add_argument("--timing-epochs", type=int, default=1)
    p.add_argument("--train-size", type=int, default=4096)
    p.add_argument("--test-size", type=int, default=1024)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data held out for validation (0 disables).",
    )
    p.add_argument(
        "--scale-x",
        action="store_true",
        help="Standardize X using train mean/std (fit on train split, applied to train+test).",
    )
    p.add_argument(
        "--scale-y",
        action="store_true",
        help=(
            "Standardize y using train mean/std. During evaluation, predictions are inverse-"
            "transformed back to original scale before computing MSE."
        ),
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--dense-depth",
        type=int,
        default=None,
        help="Dense hidden depth (defaults to sparse depth).",
    )
    p.add_argument("--dense-max-width", type=int, default=4096)
    p.add_argument(
        "--match-tolerance",
        type=float,
        default=0.01,
        help="Max relative parameter mismatch allowed (fraction).",
    )
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    shape = _parse_shape(args.shape)
    features = shape[0] * shape[1]
    if args.dense_depth is None:
        args.dense_depth = int(args.depth)

    out_dir = (
        Path(args.out)
        if args.out
        else Path("reports") / "geo_sparse" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    activation_cfg = json.loads(args.activation_config) if args.activation_config else None
    if str(args.sparse_activation).lower() == "mixed":
        types = (
            None
            if activation_cfg is None
            else activation_cfg.get("activation_types", activation_cfg.get("types"))
        )
        if not isinstance(types, list) or not types:
            raise SystemExit(
                "--sparse-activation mixed requires --activation-config JSON with "
                "'activation_types' (a non-empty list)"
            )

    dataset = _build_dataset(
        task=str(args.task),
        seed=int(args.seed),
        n_train=int(args.train_size),
        n_test=int(args.test_size),
        features=features,
    )
    X_train_raw = dataset.X_train
    y_train_raw = dataset.y_train
    X_test_raw = dataset.X_test
    y_test_raw = dataset.y_test
    X_train_raw, y_train_raw, X_val_raw, y_val_raw = _train_val_split(
        X_train_raw,
        y_train_raw,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    y_train_unscaled = y_train_raw
    y_val_unscaled = y_val_raw
    y_test_unscaled = y_test_raw
    x_scaler = None
    y_scaler = None

    X_train = X_train_raw
    X_val = X_val_raw
    X_test = X_test_raw
    y_train = y_train_raw
    y_val = y_val_raw
    y_test = y_test_raw
    if args.scale_x:
        x_scaler = _fit_standard_scaler(X_train_raw)
        X_train = x_scaler.transform(X_train_raw)
        X_val = x_scaler.transform(X_val_raw)
        X_test = x_scaler.transform(X_test_raw)
    if args.scale_y:
        y_scaler = _fit_standard_scaler(y_train_unscaled)
        y_train = y_scaler.transform(y_train_unscaled)
        y_val = y_scaler.transform(y_val_unscaled)
        y_test = y_scaler.transform(y_test_unscaled)

    dataset = DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            **dataset.meta,
            "val_size": int(X_val.shape[0]),
            "scale_x": bool(args.scale_x),
            "scale_y": bool(args.scale_y),
            "x_scaler": "standard" if args.scale_x else None,
            "y_scaler": "standard" if args.scale_y else None,
            "y_scaler_mean": float(y_scaler.mean.reshape(-1)[0]) if y_scaler is not None else None,
            "y_scaler_std": float(y_scaler.scale.reshape(-1)[0]) if y_scaler is not None else None,
        },
    )
    train_loader = _build_loader(
        dataset.X_train,
        dataset.y_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        raise SystemExit(
            "CUDA device requested but torch.cuda.is_available() is False.\n"
            f"- torch={torch.__version__}\n"
            f"- CUDA_VISIBLE_DEVICES={cuda_visible!r}\n\n"
            "If running in Docker, start the container with GPU access (e.g. `--gpus all`)."
        )
    _print_header(args, out_dir, device)
    _set_precision(args.tf32)
    dtype = _resolve_dtype(args.dtype)
    amp_dtype = _resolve_dtype(args.amp_dtype)
    if device.type == "cpu" and amp_dtype == torch.float16:
        amp_dtype = torch.bfloat16

    sparse_model = GeoSparseNet(
        input_dim=features,
        output_dim=1,
        shape=shape,
        depth=int(args.depth),
        k=int(args.k),
        pattern=str(args.pattern),
        radius=int(args.radius),
        wrap_mode=str(args.wrap_mode),
        activation_type=str(args.sparse_activation),
        activation_config=activation_cfg,
        seed=int(args.seed),
    ).to(dtype=dtype)
    sparse_params_emp = count_params(sparse_model)
    sparse_params_analytic = geo_sparse_net_params(
        shape=shape,
        depth=int(args.depth),
        k=int(args.k),
        output_dim=1,
        bias=True,
    )

    sparse_params_trainable = count_params(sparse_model, trainable_only=True)

    dense_resrelu_width, dense_resrelu_mismatch = _match_dense_residual_width_with_activation(
        target_params=int(sparse_params_emp),
        input_dim=features,
        output_dim=1,
        depth=int(args.dense_depth),
        activation_type="relu",
        norm="rms",
        max_width=int(args.dense_max_width),
    )
    dense_resrelu_model = DenseResidualNet(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_resrelu_width),
        depth=int(args.dense_depth),
        activation_type="relu",
        norm="rms",
        bias=True,
    ).to(dtype=dtype)
    dense_resrelu_params_emp = count_params(dense_resrelu_model)
    dense_resrelu_params_trainable = count_params(dense_resrelu_model, trainable_only=True)
    dense_resrelu_params_analytic = _dense_residual_params_with_activation(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_resrelu_width),
        depth=int(args.dense_depth),
        activation_type="relu",
        norm="rms",
        bias=True,
    )
    dense_resrelu_mismatch_ratio = (
        abs(dense_resrelu_params_emp - sparse_params_emp) / float(sparse_params_emp)
        if sparse_params_emp > 0
        else None
    )
    if dense_resrelu_mismatch_ratio is not None and dense_resrelu_mismatch_ratio > float(
        args.match_tolerance
    ):
        print(
            "Warning: dense_resrelu parameter mismatch exceeds tolerance "
            f"({dense_resrelu_mismatch_ratio:.4f} > {float(args.match_tolerance):.4f})."
        )

    dense_respsann_width, dense_respsann_mismatch = _match_dense_residual_width_with_activation(
        target_params=int(sparse_params_emp),
        input_dim=features,
        output_dim=1,
        depth=int(args.dense_depth),
        activation_type="psann",
        norm="rms",
        max_width=int(args.dense_max_width),
    )
    dense_respsann_model = DenseResidualNet(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_respsann_width),
        depth=int(args.dense_depth),
        activation_type="psann",
        activation_config=activation_cfg,
        norm="rms",
        bias=True,
    ).to(dtype=dtype)
    dense_respsann_params_emp = count_params(dense_respsann_model)
    dense_respsann_params_trainable = count_params(dense_respsann_model, trainable_only=True)
    dense_respsann_params_analytic = _dense_residual_params_with_activation(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_respsann_width),
        depth=int(args.dense_depth),
        activation_type="psann",
        norm="rms",
        bias=True,
    )
    dense_respsann_mismatch_ratio = (
        abs(dense_respsann_params_emp - sparse_params_emp) / float(sparse_params_emp)
        if sparse_params_emp > 0
        else None
    )
    if dense_respsann_mismatch_ratio is not None and dense_respsann_mismatch_ratio > float(
        args.match_tolerance
    ):
        print(
            "Warning: dense_respsann parameter mismatch exceeds tolerance "
            f"({dense_respsann_mismatch_ratio:.4f} > {float(args.match_tolerance):.4f})."
        )

    sparse_train = _train_model(
        sparse_model,
        train_loader=train_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        amp=bool(args.amp),
        amp_dtype=amp_dtype,
        compile_model=bool(args.compile),
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        timing_warmup_steps=int(args.timing_warmup_steps),
        timing_epochs=int(args.timing_epochs),
        curve_every=1,
        curve_test=(dataset.X_test, dataset.y_test),
        curve_test_unscaled=y_test_unscaled,
        curve_test_y_scaler=y_scaler,
    )
    dense_resrelu_train = _train_model(
        dense_resrelu_model,
        train_loader=train_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        amp=bool(args.amp),
        amp_dtype=amp_dtype,
        compile_model=bool(args.compile),
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        timing_warmup_steps=int(args.timing_warmup_steps),
        timing_epochs=int(args.timing_epochs),
        curve_every=1,
        curve_test=(dataset.X_test, dataset.y_test),
        curve_test_unscaled=y_test_unscaled,
        curve_test_y_scaler=y_scaler,
    )
    dense_respsann_train = _train_model(
        dense_respsann_model,
        train_loader=train_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        amp=bool(args.amp),
        amp_dtype=amp_dtype,
        compile_model=bool(args.compile),
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        timing_warmup_steps=int(args.timing_warmup_steps),
        timing_epochs=int(args.timing_epochs),
        curve_every=1,
        curve_test=(dataset.X_test, dataset.y_test),
        curve_test_unscaled=y_test_unscaled,
        curve_test_y_scaler=y_scaler,
    )

    sparse_curve = sparse_train.pop("curve", [])
    dense_resrelu_curve = dense_resrelu_train.pop("curve", [])
    dense_respsann_curve = dense_respsann_train.pop("curve", [])

    sparse_metrics_train = _evaluate_metrics(
        sparse_model,
        X=X_train,
        y=y_train,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_train_unscaled,
    )
    sparse_metrics_val = (
        _evaluate_metrics(
            sparse_model,
            X=X_val,
            y=y_val,
            device=device,
            y_scaler=y_scaler,
            y_unscaled=y_val_unscaled,
        )
        if X_val.size
        else None
    )
    sparse_metrics_test = _evaluate_metrics(
        sparse_model,
        X=X_test,
        y=y_test,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_test_unscaled,
    )

    dense_resrelu_metrics_train = _evaluate_metrics(
        dense_resrelu_model,
        X=X_train,
        y=y_train,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_train_unscaled,
    )
    dense_resrelu_metrics_val = (
        _evaluate_metrics(
            dense_resrelu_model,
            X=X_val,
            y=y_val,
            device=device,
            y_scaler=y_scaler,
            y_unscaled=y_val_unscaled,
        )
        if X_val.size
        else None
    )
    dense_resrelu_metrics_test = _evaluate_metrics(
        dense_resrelu_model,
        X=X_test,
        y=y_test,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_test_unscaled,
    )

    dense_respsann_metrics_train = _evaluate_metrics(
        dense_respsann_model,
        X=X_train,
        y=y_train,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_train_unscaled,
    )
    dense_respsann_metrics_val = (
        _evaluate_metrics(
            dense_respsann_model,
            X=X_val,
            y=y_val,
            device=device,
            y_scaler=y_scaler,
            y_unscaled=y_val_unscaled,
        )
        if X_val.size
        else None
    )
    dense_respsann_metrics_test = _evaluate_metrics(
        dense_respsann_model,
        X=X_test,
        y=y_test,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_test_unscaled,
    )

    env = _get_env_info(device)

    manifest = {
        "shape": list(shape),
        "features": features,
        "depth_sparse": int(args.depth),
        "k": int(args.k),
        "dense_depth": int(args.dense_depth),
        "dense_resrelu_width": int(dense_resrelu_width),
        "dense_respsann_width": int(dense_respsann_width),
        "pattern": str(args.pattern),
        "radius": int(args.radius),
        "wrap_mode": str(args.wrap_mode),
        "sparse_activation": str(args.sparse_activation),
        "activation_config": activation_cfg,
        "task": str(args.task),
        "dataset_meta": dataset.meta,
        "train_size": int(args.train_size),
        "val_fraction": float(args.val_fraction),
        "val_size": int(X_val.shape[0]),
        "test_size": int(args.test_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "amp_dtype_effective": str(amp_dtype),
        "tf32": bool(args.tf32),
        "compile": bool(args.compile),
        "compile_backend": str(args.compile_backend),
        "compile_mode": str(args.compile_mode),
        "timing_warmup_steps": int(args.timing_warmup_steps),
        "timing_epochs": int(args.timing_epochs),
        "match_tolerance": float(args.match_tolerance),
    }

    results = {
        "environment": env,
        "manifest": manifest,
        "models": [
            {
                "name": "geo_sparse",
                "params_empirical": sparse_params_emp,
                "params_trainable": sparse_params_trainable,
                "params_analytic": sparse_params_analytic,
                "train": sparse_train,
                "curve": sparse_curve,
                "metrics_train": sparse_metrics_train,
                "metrics_val": sparse_metrics_val,
                "metrics_test": sparse_metrics_test,
                "mse_train": sparse_metrics_train.get("mse"),
                "mse_val": sparse_metrics_val.get("mse") if sparse_metrics_val else None,
                "mse_test": sparse_metrics_test.get("mse"),
            },
            {
                "name": "dense_resrelu",
                "params_empirical": dense_resrelu_params_emp,
                "params_trainable": dense_resrelu_params_trainable,
                "params_analytic": dense_resrelu_params_analytic,
                "param_mismatch": int(abs(dense_resrelu_params_emp - sparse_params_emp)),
                "param_mismatch_ratio": dense_resrelu_mismatch_ratio,
                "train": dense_resrelu_train,
                "curve": dense_resrelu_curve,
                "metrics_train": dense_resrelu_metrics_train,
                "metrics_val": dense_resrelu_metrics_val,
                "metrics_test": dense_resrelu_metrics_test,
                "mse_train": dense_resrelu_metrics_train.get("mse"),
                "mse_val": (
                    dense_resrelu_metrics_val.get("mse") if dense_resrelu_metrics_val else None
                ),
                "mse_test": dense_resrelu_metrics_test.get("mse"),
            },
            {
                "name": "dense_respsann",
                "params_empirical": dense_respsann_params_emp,
                "params_trainable": dense_respsann_params_trainable,
                "params_analytic": dense_respsann_params_analytic,
                "param_mismatch": int(abs(dense_respsann_params_emp - sparse_params_emp)),
                "param_mismatch_ratio": dense_respsann_mismatch_ratio,
                "train": dense_respsann_train,
                "curve": dense_respsann_curve,
                "metrics_train": dense_respsann_metrics_train,
                "metrics_val": dense_respsann_metrics_val,
                "metrics_test": dense_respsann_metrics_test,
                "mse_train": dense_respsann_metrics_train.get("mse"),
                "mse_val": (
                    dense_respsann_metrics_val.get("mse") if dense_respsann_metrics_val else None
                ),
                "mse_test": dense_respsann_metrics_test.get("mse"),
            },
        ],
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary_rows = []
    for entry in results["models"]:
        row = {
            "model": entry["name"],
            "params_empirical": entry["params_empirical"],
            "params_analytic": entry["params_analytic"],
            "mse_train": entry.get("mse_train"),
            "mse_val": entry.get("mse_val"),
            "mse_test": entry["mse_test"],
        }
        if "param_mismatch" in entry:
            row["param_mismatch"] = entry["param_mismatch"]
        if "param_mismatch_ratio" in entry:
            row["param_mismatch_ratio"] = entry["param_mismatch_ratio"]
        if isinstance(entry.get("metrics_train"), dict):
            row.update(_flatten_dict("train_", entry["metrics_train"]))
        if isinstance(entry.get("metrics_val"), dict):
            row.update(_flatten_dict("val_", entry["metrics_val"]))
        if isinstance(entry.get("metrics_test"), dict):
            row.update(_flatten_dict("test_", entry["metrics_test"]))
        row.update(_flatten_dict("", entry["train"]))
        summary_rows.append(row)
    _write_summary(summary_rows, out_dir / "summary.csv")

    print(f"Wrote results to {out_dir}")
