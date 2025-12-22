#!/usr/bin/env python
"""Run small ablations for PSANN regressors across diverse datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from psann import ResPSANNRegressor, SGRPSANNRegressor, WaveResNetRegressor
from psann.utils import make_regime_switch_ts, seed_all


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass
class DatasetBundle:
    name: str
    task: str
    kind: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]
    y_train_labels: Optional[np.ndarray] = None
    y_test_labels: Optional[np.ndarray] = None


@dataclass
class DatasetSpec:
    name: str
    task: str
    kind: str
    builder: Callable[[int], DatasetBundle]


def _make_tabular_sine(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_train, n_test, features = 1024, 256, 8
    X = rng.uniform(-2.0, 2.0, size=(n_train + n_test, features)).astype(np.float32)
    noise = 0.05 * rng.standard_normal(size=(n_train + n_test,))
    y = (
        np.sin(3.0 * X[:, 0])
        + 0.5 * np.cos(2.0 * X[:, 1])
        + 0.2 * X[:, 2] * X[:, 3]
        - 0.1 * (X[:, 4] ** 2)
        + noise
    )
    y = y.astype(np.float32).reshape(-1, 1)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return DatasetBundle(
        name="tabular_sine",
        task="regression",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={"features": features, "train_size": n_train, "test_size": n_test},
    )


def _make_tabular_shifted(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_train, n_test, features = 1024, 256, 8
    X_train = rng.normal(loc=0.0, scale=1.0, size=(n_train, features)).astype(np.float32)
    X_test = rng.normal(loc=0.6, scale=1.3, size=(n_test, features)).astype(np.float32)

    def _targets(X: np.ndarray) -> np.ndarray:
        noise = 0.2 * rng.standard_t(df=3, size=(X.shape[0],))
        heavy = rng.random(X.shape[0]) < 0.03
        noise = noise + heavy * rng.normal(scale=2.0, size=(X.shape[0],))
        base = np.where(
            X[:, 0] > 0,
            1.5 * X[:, 0] - 0.5 * X[:, 1],
            -1.2 * X[:, 0] + 0.3 * X[:, 2],
        )
        return (base + noise).astype(np.float32).reshape(-1, 1)

    y_train = _targets(X_train)
    y_test = _targets(X_test)
    return DatasetBundle(
        name="tabular_shifted",
        task="regression",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
            "train_shift": 0.0,
            "test_shift": 0.6,
        },
    )


def _make_classification_clusters(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    n_train, n_test = 900, 300
    n_classes = 3
    total = n_train + n_test
    per_class = total // n_classes
    extras = total % n_classes
    centers = np.array([[0.0, 0.0], [2.0, 2.0], [-2.0, 2.0]], dtype=np.float32)

    features = []
    labels = []
    for idx in range(n_classes):
        count = per_class + (1 if idx < extras else 0)
        base = rng.normal(scale=0.5, size=(count, 2)).astype(np.float32) + centers[idx]
        extra1 = np.sin(base[:, :1]) + 0.1 * rng.standard_normal(size=(count, 1))
        extra2 = np.cos(base[:, 1:2]) + 0.1 * rng.standard_normal(size=(count, 1))
        feats = np.concatenate([base, extra1, extra2], axis=1)
        features.append(feats)
        labels.append(np.full((count,), idx, dtype=np.int64))

    X = np.concatenate(features, axis=0).astype(np.float32)
    y_labels = np.concatenate(labels, axis=0)
    order = rng.permutation(X.shape[0])
    X = X[order]
    y_labels = y_labels[order]
    y_onehot = np.eye(n_classes, dtype=np.float32)[y_labels]

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_onehot[:n_train], y_onehot[n_train:]
    y_train_labels = y_labels[:n_train]
    y_test_labels = y_labels[n_train:]
    return DatasetBundle(
        name="classification_clusters",
        task="classification",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "features": X.shape[1],
            "train_size": n_train,
            "test_size": n_test,
            "classes": n_classes,
        },
        y_train_labels=y_train_labels,
        y_test_labels=y_test_labels,
    )


def _make_ts_periodic(seed: int) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total_steps = 1600
    window = 32
    t = np.arange(total_steps, dtype=np.float32)
    daily = np.sin(2.0 * math.pi * t / 24.0)
    weekly = np.sin(2.0 * math.pi * t / (24.0 * 7.0))
    mid = np.cos(2.0 * math.pi * t / 12.0)
    noise = 0.05 * rng.standard_normal(size=total_steps)
    series = (daily + 0.6 * weekly + 0.3 * mid + noise).astype(np.float32)
    feats = np.stack([daily, np.cos(2.0 * math.pi * t / 24.0), weekly], axis=1).astype(
        np.float32
    )

    X, y = _windowed_series(feats, series, window=window)
    return _split_sequence_dataset(
        name="ts_periodic",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window},
    )


def _make_ts_regime(seed: int) -> DatasetBundle:
    series, contexts = make_regime_switch_ts(1400, regimes=3, seed=seed)
    series_np = series.numpy().astype(np.float32)
    contexts_np = contexts.numpy().astype(np.float32)
    feats = np.concatenate([series_np[:, None], contexts_np], axis=1).astype(np.float32)
    window = 32
    X, y = _windowed_series(feats, series_np, window=window)
    return _split_sequence_dataset(
        name="ts_regime_switch",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window, "regimes": 3},
    )


def _windowed_series(
    feats: np.ndarray,
    target: np.ndarray,
    *,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if feats.ndim != 2:
        raise ValueError("feats must be 2D (T, F)")
    if target.ndim != 1:
        raise ValueError("target must be 1D (T,)")
    if feats.shape[0] != target.shape[0]:
        raise ValueError("feats and target must share the time dimension")
    if window <= 0 or window >= target.shape[0]:
        raise ValueError("window must be positive and shorter than series length")

    windows = []
    targets = []
    for idx in range(window, target.shape[0]):
        windows.append(feats[idx - window : idx])
        targets.append(target[idx])
    X = np.stack(windows, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32).reshape(-1, 1)
    return X, y


def _split_sequence_dataset(
    *,
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    kind: str,
    meta: Dict[str, Any],
    train_ratio: float = 0.8,
) -> DatasetBundle:
    split = int(X.shape[0] * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    meta = dict(meta)
    meta.update({"train_size": X_train.shape[0], "test_size": X_test.shape[0], "window": window})
    return DatasetBundle(
        name=name,
        task="regression",
        kind=kind,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta=meta,
    )


DATASETS: Dict[str, DatasetSpec] = {
    "tabular_sine": DatasetSpec(
        name="tabular_sine", task="regression", kind="tabular", builder=_make_tabular_sine
    ),
    "tabular_shifted": DatasetSpec(
        name="tabular_shifted",
        task="regression",
        kind="tabular",
        builder=_make_tabular_shifted,
    ),
    "classification_clusters": DatasetSpec(
        name="classification_clusters",
        task="classification",
        kind="tabular",
        builder=_make_classification_clusters,
    ),
    "ts_periodic": DatasetSpec(
        name="ts_periodic", task="regression", kind="sequence", builder=_make_ts_periodic
    ),
    "ts_regime_switch": DatasetSpec(
        name="ts_regime_switch",
        task="regression",
        kind="sequence",
        builder=_make_ts_regime,
    ),
}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    name: str
    estimator: Callable[..., Any]
    params: Dict[str, Any]


MODELS: Dict[str, ModelSpec] = {
    # ResPSANN ablations
    "res_base": ModelSpec(
        name="res_base",
        estimator=ResPSANNRegressor,
        params={"hidden_layers": 4, "hidden_units": 64, "norm": "rms", "drop_path_max": 0.0},
    ),
    "res_drop_path": ModelSpec(
        name="res_drop_path",
        estimator=ResPSANNRegressor,
        params={"hidden_layers": 4, "hidden_units": 64, "norm": "rms", "drop_path_max": 0.1},
    ),
    "res_no_norm": ModelSpec(
        name="res_no_norm",
        estimator=ResPSANNRegressor,
        params={"hidden_layers": 4, "hidden_units": 64, "norm": "none", "drop_path_max": 0.0},
    ),
    # WaveResNet ablations
    "wrn_base": ModelSpec(
        name="wrn_base",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_film": True,
            "use_phase_shift": True,
        },
    ),
    "wrn_no_phase": ModelSpec(
        name="wrn_no_phase",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_film": True,
            "use_phase_shift": False,
        },
    ),
    "wrn_no_film": ModelSpec(
        name="wrn_no_film",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_film": False,
            "use_phase_shift": True,
        },
    ),
    "wrn_spec_gate_rfft": ModelSpec(
        name="wrn_spec_gate_rfft",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_spectral_gate": True,
            "k_fft": 64,
            "gate_type": "rfft",
            "gate_groups": "depthwise",
            "gate_strength": 1.0,
        },
    ),
    "wrn_spec_gate_feats": ModelSpec(
        name="wrn_spec_gate_feats",
        estimator=WaveResNetRegressor,
        params={
            "hidden_layers": 6,
            "hidden_units": 64,
            "norm": "rms",
            "use_spectral_gate": True,
            "k_fft": 64,
            "gate_type": "fourier_features",
            "gate_groups": "depthwise",
            "gate_strength": 1.0,
        },
    ),
    # SGR-PSANN ablations
    "sgr_base": ModelSpec(
        name="sgr_base",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "k_fft": 64,
            "gate_type": "rfft",
            "use_spectral_gate": True,
            "phase_trainable": True,
        },
    ),
    "sgr_no_gate": ModelSpec(
        name="sgr_no_gate",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "use_spectral_gate": False,
            "phase_trainable": True,
        },
    ),
    "sgr_fourier_feats": ModelSpec(
        name="sgr_fourier_feats",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "k_fft": 64,
            "gate_type": "fourier_features",
            "use_spectral_gate": True,
            "phase_trainable": True,
        },
    ),
    "sgr_no_phase": ModelSpec(
        name="sgr_no_phase",
        estimator=SGRPSANNRegressor,
        params={
            "hidden_layers": 3,
            "hidden_units": 64,
            "k_fft": 64,
            "gate_type": "rfft",
            "use_spectral_gate": True,
            "phase_trainable": False,
        },
    ),
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _coerce_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - (u / v if v != 0 else np.nan))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    rmse = math.sqrt(float(np.mean((y_pred - y_true) ** 2)))
    span = float(y_true.max() - y_true.min())
    if span == 0:
        return float("nan")
    return float(rmse / span)


def _mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    numerator = float(np.mean(np.abs(y_pred - y_true)))
    diffs = np.abs(np.diff(y_true, axis=0))
    denom = float(np.mean(diffs))
    if denom == 0:
        return float("nan")
    return float(numerator / denom)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = math.sqrt(mse)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": _r2_score(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "nrmse": _nrmse(y_true, y_pred),
        "mase": _mase(y_true, y_pred),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp + fp == 0 or tp + fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _classification_metrics(
    y_true_labels: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    num_classes: int,
) -> Dict[str, float]:
    y_pred_scores = _coerce_2d(y_pred_scores)
    y_pred = y_pred_scores.argmax(axis=1)
    acc = float(np.mean(y_pred == y_true_labels))
    return {
        "accuracy": acc,
        "macro_f1": _macro_f1(y_true_labels, y_pred, num_classes=num_classes),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _count_params(estimator: Any) -> int:
    model = getattr(estimator, "model_", None)
    if model is None:
        return 0
    return int(sum(int(p.numel()) for p in model.parameters()))


def _run_single(
    model: ModelSpec,
    dataset: DatasetBundle,
    *,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    save_model_path: Optional[Path] = None,
    save_preds_path: Optional[Path] = None,
) -> Dict[str, Any]:
    seed_all(seed)
    params = dict(model.params)
    params.update(
        {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "device": device,
            "random_state": int(seed),
        }
    )
    estimator = model.estimator(**params)
    start = time.perf_counter()
    estimator.fit(dataset.X_train, dataset.y_train, verbose=0)
    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        estimator.save(str(save_model_path))
    elapsed = time.perf_counter() - start

    pred_train = estimator.predict(dataset.X_train)
    pred_test = estimator.predict(dataset.X_test)
    result: Dict[str, Any] = {
        "train_time_s": float(elapsed),
        "n_params": _count_params(estimator),
    }
    if dataset.task == "classification":
        if dataset.y_train_labels is None or dataset.y_test_labels is None:
            raise ValueError("classification dataset missing label arrays")
        num_classes = int(dataset.meta.get("classes", pred_train.shape[-1]))
        result["metrics_train"] = _classification_metrics(
            dataset.y_train_labels, pred_train, num_classes=num_classes
        )
        result["metrics_test"] = _classification_metrics(
            dataset.y_test_labels, pred_test, num_classes=num_classes
        )
    else:
        result["metrics_train"] = _regression_metrics(dataset.y_train, pred_train)
        result["metrics_test"] = _regression_metrics(dataset.y_test, pred_test)
    if save_preds_path is not None:
        save_preds_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y_train": dataset.y_train,
            "y_test": dataset.y_test,
            "pred_train": pred_train,
            "pred_test": pred_test,
        }
        if dataset.y_train_labels is not None:
            payload["y_train_labels"] = dataset.y_train_labels
        if dataset.y_test_labels is not None:
            payload["y_test_labels"] = dataset.y_test_labels
        np.savez_compressed(save_preds_path, **payload)
    return result


def _parse_list(value: str) -> List[str]:
    return [entry.strip() for entry in value.split(",") if entry.strip()]


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("reports") / "ablations" / f"{stamp}_regressor_ablations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="tabular_sine,tabular_shifted,classification_clusters,ts_periodic,ts_regime_switch",
        help="Comma-separated dataset names to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="res_base,res_drop_path,res_no_norm,wrn_base,wrn_no_phase,wrn_no_film,wrn_spec_gate_rfft,wrn_spec_gate_feats,sgr_base,sgr_no_gate,sgr_fourier_feats,sgr_no_phase",
        help="Comma-separated model keys to run.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1",
        help="Comma-separated seeds for dataset/model runs.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|cuda|auto.")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs per run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--out", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Persist fitted estimators under outputs/models/.",
    )
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Persist predictions under outputs/preds/.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs already present in results.jsonl.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model keys and exit.",
    )
    return parser.parse_args()


def _load_existing_run_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    run_ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = payload.get("run_id")
        if isinstance(run_id, str):
            run_ids.add(run_id)
    return run_ids


def _write_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _flatten_result(entry: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in entry.items():
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                flat[f"{key}_{sub_k}"] = sub_v
        else:
            flat[key] = value
    return flat


def _slugify(value: str) -> str:
    return (
        value.replace(":", "__")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def _load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


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

    datasets = _parse_list(args.datasets)
    models = _parse_list(args.models)
    seeds = [int(s) for s in _parse_list(args.seeds)]
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
                slug = _slugify(run_id)
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
                model_path = (
                    out_dir / "models" / f"{slug}.pt" if args.save_models else None
                )
                preds_path = (
                    out_dir / "preds" / f"{slug}.npz" if args.save_preds else None
                )
                try:
                    result = _run_single(
                        model_spec,
                        dataset,
                        seed=seed,
                        device=args.device,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
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
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
