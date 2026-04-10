# ruff: noqa: F403,F405
from __future__ import annotations

from .data import load_beijing_light, load_eaf_temp_lite, load_jena_light
from .env import get_results_root, seed_all
from .metrics import (
    _apply_y_scaler,
    _fit_y_scaler,
    _history_stats,
    _match_mlp_hidden,
    _prefix_metrics,
    evaluate_regressor,
    jacobian_pr,
)
from .models import MLPRegressor, PSANNConvSpine
from .shared import *
from .train import TrainSpec, train_regressor


def run_light_task(
    task: str,
    seeds: List[int],
    device: torch.device,
    epochs: int,
    pr_snapshots: bool,
    match_params: bool,
    scale_y: bool,
    metrics_rows: List[dict],
    history_rows: List[dict],
) -> None:
    if task == "jena":
        train_X, train_y, val_X, val_y, test_X, test_y = load_jena_light(72, 36, 120)
        in_ch, horizon = train_X.shape[-1], train_y.shape[-1]
        y_scaler = None
        train_y_unscaled = train_y
        val_y_unscaled = val_y
        test_y_unscaled = test_y
        if scale_y:
            y_scaler = _fit_y_scaler(train_y)
            train_y = _apply_y_scaler(train_y, y_scaler)
            val_y = _apply_y_scaler(val_y, y_scaler)
            test_y = _apply_y_scaler(test_y, y_scaler)
        specs = [
            (
                "psann_conv",
                TrainSpec("psann_conv", hidden=48, depth=2, kernel_size=5, epochs=epochs),
            ),
            ("mlp", TrainSpec("mlp", hidden=64, depth=2, epochs=epochs)),
        ]
        target_params = None
        mlp_mismatch = None
        if match_params:
            psann_tmp = PSANNConvSpine(
                in_ch, specs[0][1].hidden, specs[0][1].depth, specs[0][1].kernel_size, horizon
            )
            target_params = count_params(psann_tmp)
            mlp_hidden, mlp_mismatch = _match_mlp_hidden(
                target_params=target_params,
                input_dim=train_X.shape[1] * in_ch,
                output_dim=horizon,
                depth=specs[1][1].depth,
            )
            specs[1][1].hidden = mlp_hidden
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                model = (
                    PSANNConvSpine(in_ch, spec.hidden, spec.depth, spec.kernel_size, horizon)
                    if spec.model == "psann_conv"
                    else MLPRegressor(train_X.shape[1] * in_ch, spec.hidden, spec.depth, horizon)
                )
                model, info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                info = dict(info)
                history = info.pop("history", [])
                train_metrics = evaluate_regressor(
                    model, train_X, train_y, device, y_scaler, train_y_unscaled
                )
                val_metrics = evaluate_regressor(
                    model, val_X, val_y, device, y_scaler, val_y_unscaled
                )
                test_metrics = evaluate_regressor(
                    model, test_X, test_y, device, y_scaler, test_y_unscaled
                )
                params_total = count_params(model)
                params_trainable = count_params(model, trainable_only=True)
                history_rows.extend(
                    {
                        "task": "jena_light",
                        "model": name,
                        "seed": seed,
                        **entry,
                    }
                    for entry in history
                )
                history_stats = _history_stats(history)
                metrics_rows.append(
                    {
                        "task": "jena_light",
                        "model": name,
                        "seed": seed,
                        "params_total": params_total,
                        "params_trainable": params_trainable,
                        "param_target": target_params,
                        "param_mismatch": mlp_mismatch if name == "mlp" else 0,
                        "param_mismatch_ratio": (
                            float(mlp_mismatch) / float(target_params)
                            if (name == "mlp" and target_params)
                            else 0.0
                        ),
                        "train_size": int(train_X.shape[0]),
                        "val_size": int(val_X.shape[0]),
                        "test_size": int(test_X.shape[0]),
                        "input_shape": list(train_X.shape[1:]),
                        "scale_y": bool(scale_y),
                        "y_scaler": "standard" if y_scaler is not None else None,
                        "y_scaler_mean": (
                            float(y_scaler.mean_.reshape(-1)[0]) if y_scaler is not None else None
                        ),
                        "y_scaler_std": (
                            float(y_scaler.scale_.reshape(-1)[0]) if y_scaler is not None else None
                        ),
                        **info,
                        **_prefix_metrics("train", train_metrics),
                        **_prefix_metrics("val", val_metrics),
                        **_prefix_metrics("test", test_metrics),
                        **history_stats,
                    }
                )
                if pr_snapshots and name == "psann_conv":
                    idx = np.random.choice(
                        test_X.shape[0], size=min(32, test_X.shape[0]), replace=False
                    )
                    top_sv, sum_sv, pr = jacobian_pr(model, test_X[idx], device)
                    pr_df = pd.DataFrame(
                        [
                            {
                                "task": "jena_light",
                                "model": name,
                                "seed": seed,
                                "phase": "end",
                                "top_sv": top_sv,
                                "sum_sv": sum_sv,
                                "pr": pr,
                            }
                        ]
                    )
                    pr_out = get_results_root() / "jacobian_pr.csv"
                    mode = "a" if pr_out.exists() else "w"
                    pr_df.to_csv(pr_out, index=False, mode=mode, header=(mode == "w"))
    elif task == "beijing":
        train_X, train_y, val_X, val_y, test_X, test_y = load_beijing_light("Guanyuan", 24, 6, 120)
        in_ch, horizon = train_X.shape[-1], train_y.shape[-1]
        y_scaler = None
        train_y_unscaled = train_y
        val_y_unscaled = val_y
        test_y_unscaled = test_y
        if scale_y:
            y_scaler = _fit_y_scaler(train_y)
            train_y = _apply_y_scaler(train_y, y_scaler)
            val_y = _apply_y_scaler(val_y, y_scaler)
            test_y = _apply_y_scaler(test_y, y_scaler)
        specs = [
            (
                "psann_conv",
                TrainSpec("psann_conv", hidden=64, depth=2, kernel_size=5, epochs=epochs),
            ),
            ("mlp", TrainSpec("mlp", hidden=96, depth=2, epochs=epochs)),
        ]
        target_params = None
        mlp_mismatch = None
        if match_params:
            psann_tmp = PSANNConvSpine(
                in_ch, specs[0][1].hidden, specs[0][1].depth, specs[0][1].kernel_size, horizon
            )
            target_params = count_params(psann_tmp)
            mlp_hidden, mlp_mismatch = _match_mlp_hidden(
                target_params=target_params,
                input_dim=train_X.shape[1] * in_ch,
                output_dim=horizon,
                depth=specs[1][1].depth,
            )
            specs[1][1].hidden = mlp_hidden
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                model = (
                    PSANNConvSpine(in_ch, spec.hidden, spec.depth, spec.kernel_size, horizon)
                    if spec.model == "psann_conv"
                    else MLPRegressor(train_X.shape[1] * in_ch, spec.hidden, spec.depth, horizon)
                )
                model, info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                info = dict(info)
                history = info.pop("history", [])
                train_metrics = evaluate_regressor(model, train_X, train_y, device)
                val_metrics = evaluate_regressor(model, val_X, val_y, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                params_total = count_params(model)
                params_trainable = count_params(model, trainable_only=True)
                history_rows.extend(
                    {
                        "task": "beijing_light",
                        "model": name,
                        "seed": seed,
                        **entry,
                    }
                    for entry in history
                )
                history_stats = _history_stats(history)
                metrics_rows.append(
                    {
                        "task": "beijing_light",
                        "model": name,
                        "seed": seed,
                        "params_total": params_total,
                        "params_trainable": params_trainable,
                        "param_target": target_params,
                        "param_mismatch": mlp_mismatch if name == "mlp" else 0,
                        "param_mismatch_ratio": (
                            float(mlp_mismatch) / float(target_params)
                            if (name == "mlp" and target_params)
                            else 0.0
                        ),
                        "train_size": int(train_X.shape[0]),
                        "val_size": int(val_X.shape[0]),
                        "test_size": int(test_X.shape[0]),
                        "input_shape": list(train_X.shape[1:]),
                        "scale_y": bool(scale_y),
                        "y_scaler": "standard" if y_scaler is not None else None,
                        "y_scaler_mean": (
                            float(y_scaler.mean_.reshape(-1)[0]) if y_scaler is not None else None
                        ),
                        "y_scaler_std": (
                            float(y_scaler.scale_.reshape(-1)[0]) if y_scaler is not None else None
                        ),
                        **info,
                        **_prefix_metrics("train", train_metrics),
                        **_prefix_metrics("val", val_metrics),
                        **_prefix_metrics("test", test_metrics),
                        **history_stats,
                    }
                )
    elif task == "eaf":
        train_X, train_y, val_X, val_y, test_X, test_y = load_eaf_temp_lite(16, 1, 5, 120)
        in_ch, horizon = train_X.shape[-1], train_y.shape[-1]
        y_scaler = None
        train_y_unscaled = train_y
        val_y_unscaled = val_y
        test_y_unscaled = test_y
        if scale_y:
            y_scaler = _fit_y_scaler(train_y)
            train_y = _apply_y_scaler(train_y, y_scaler)
            val_y = _apply_y_scaler(val_y, y_scaler)
            test_y = _apply_y_scaler(test_y, y_scaler)
        specs = [
            (
                "psann_conv",
                TrainSpec("psann_conv", hidden=32, depth=2, kernel_size=3, epochs=max(epochs, 8)),
            ),
            ("mlp", TrainSpec("mlp", hidden=48, depth=2, epochs=max(epochs, 8))),
        ]
        target_params = None
        mlp_mismatch = None
        if match_params:
            psann_tmp = PSANNConvSpine(
                in_ch, specs[0][1].hidden, specs[0][1].depth, specs[0][1].kernel_size, horizon
            )
            target_params = count_params(psann_tmp)
            mlp_hidden, mlp_mismatch = _match_mlp_hidden(
                target_params=target_params,
                input_dim=train_X.shape[1] * in_ch,
                output_dim=horizon,
                depth=specs[1][1].depth,
            )
            specs[1][1].hidden = mlp_hidden
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                model = (
                    PSANNConvSpine(in_ch, spec.hidden, spec.depth, spec.kernel_size, horizon)
                    if spec.model == "psann_conv"
                    else MLPRegressor(train_X.shape[1] * in_ch, spec.hidden, spec.depth, horizon)
                )
                model, info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                info = dict(info)
                history = info.pop("history", [])
                train_metrics = evaluate_regressor(model, train_X, train_y, device)
                val_metrics = evaluate_regressor(model, val_X, val_y, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                params_total = count_params(model)
                params_trainable = count_params(model, trainable_only=True)
                history_rows.extend(
                    {
                        "task": "eaf_temp_lite",
                        "model": name,
                        "seed": seed,
                        **entry,
                    }
                    for entry in history
                )
                history_stats = _history_stats(history)
                metrics_rows.append(
                    {
                        "task": "eaf_temp_lite",
                        "model": name,
                        "seed": seed,
                        "params_total": params_total,
                        "params_trainable": params_trainable,
                        "param_target": target_params,
                        "param_mismatch": mlp_mismatch if name == "mlp" else 0,
                        "param_mismatch_ratio": (
                            float(mlp_mismatch) / float(target_params)
                            if (name == "mlp" and target_params)
                            else 0.0
                        ),
                        "train_size": int(train_X.shape[0]),
                        "val_size": int(val_X.shape[0]),
                        "test_size": int(test_X.shape[0]),
                        "input_shape": list(train_X.shape[1:]),
                        "scale_y": bool(scale_y),
                        "y_scaler": "standard" if y_scaler is not None else None,
                        "y_scaler_mean": (
                            float(y_scaler.mean_.reshape(-1)[0]) if y_scaler is not None else None
                        ),
                        "y_scaler_std": (
                            float(y_scaler.scale_.reshape(-1)[0]) if y_scaler is not None else None
                        ),
                        **info,
                        **_prefix_metrics("train", train_metrics),
                        **_prefix_metrics("val", val_metrics),
                        **_prefix_metrics("test", test_metrics),
                        **history_stats,
                    }
                )
    else:
        raise ValueError(f"Unknown task: {task}")
