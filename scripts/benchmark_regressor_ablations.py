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
from psann.params import count_params as psann_count_params
from psann.utils import make_context_rotating_moons, make_drift_series, make_regime_switch_ts, make_shock_series, seed_all
from gpu_env_report import gather_env_info

try:
    from scripts._cli_utils import parse_comma_list, slugify
except ImportError:  # pragma: no cover - supports `python scripts/foo.py`
    from _cli_utils import parse_comma_list, slugify  # type: ignore


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
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    y_val_labels: Optional[np.ndarray] = None
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


def _make_ts_drift(seed: int) -> DatasetBundle:
    X_raw, y_raw = make_drift_series(1600, drift=0.001, frequency=0.02, noise=0.02, seed=seed)
    series = torch.cat([X_raw.squeeze(-1), y_raw[-1].squeeze(-1).reshape(1)], dim=0)
    series_np = series.numpy().astype(np.float32)
    feats = series_np[:-1].reshape(-1, 1)
    targets = series_np[1:]
    window = 32
    X, y = _windowed_series(feats, targets, window=window)
    return _split_sequence_dataset(
        name="ts_drift",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window, "drift": 0.001},
    )


def _make_ts_shock(seed: int) -> DatasetBundle:
    X_raw, y_raw = make_shock_series(
        1600, shock_prob=0.05, shock_scale=2.0, noise=0.05, mean_revert=0.85, seed=seed
    )
    series = torch.cat([X_raw.squeeze(-1), y_raw[-1].squeeze(-1).reshape(1)], dim=0)
    series_np = series.numpy().astype(np.float32)
    feats = series_np[:-1].reshape(-1, 1)
    targets = series_np[1:]
    window = 32
    X, y = _windowed_series(feats, targets, window=window)
    return _split_sequence_dataset(
        name="ts_shock",
        X=X,
        y=y,
        window=window,
        kind="sequence",
        meta={"features": feats.shape[1], "window": window, "shock_prob": 0.05},
    )


def _make_context_rotating_moons(seed: int) -> DatasetBundle:
    feats, labels, contexts = make_context_rotating_moons(1200, noise=0.05, seed=seed)
    X = torch.cat([feats, contexts], dim=1).numpy().astype(np.float32)
    y_labels = labels.numpy().astype(np.int64)
    y_onehot = np.eye(2, dtype=np.float32)[y_labels]
    order = np.random.default_rng(seed).permutation(X.shape[0])
    X = X[order]
    y_onehot = y_onehot[order]
    y_labels = y_labels[order]
    n_train = 900
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y_onehot[:n_train], y_onehot[n_train:]
    y_train_labels = y_labels[:n_train]
    y_test_labels = y_labels[n_train:]
    return DatasetBundle(
        name="context_rotating_moons",
        task="classification",
        kind="tabular",
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "features": X.shape[1],
            "train_size": n_train,
            "test_size": X_test.shape[0],
            "classes": 2,
            "context_dim": 1,
        },
        y_train_labels=y_train_labels,
        y_test_labels=y_test_labels,
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
    "ts_drift": DatasetSpec(
        name="ts_drift", task="regression", kind="sequence", builder=_make_ts_drift
    ),
    "ts_shock": DatasetSpec(
        name="ts_shock", task="regression", kind="sequence", builder=_make_ts_shock
    ),
    "context_rotating_moons": DatasetSpec(
        name="context_rotating_moons",
        task="classification",
        kind="tabular",
        builder=_make_context_rotating_moons,
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
    "res_relu_sigmoid_psann": ModelSpec(
        name="res_relu_sigmoid_psann",
        estimator=ResPSANNRegressor,
        params={
            "hidden_layers": 4,
            "hidden_units": 64,
            "norm": "rms",
            "drop_path_max": 0.0,
            "activation_type": "relu_sigmoid_psann",
            "activation": {"slope_init": 1.0, "clip_max": 1.0},
        },
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


def _split_train_val(
    dataset: DatasetBundle,
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    if dataset.X_val is not None and dataset.y_val is not None:
        return (
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            dataset.y_train_labels,
            dataset.y_val_labels,
        )

    X = dataset.X_train
    y = dataset.y_train
    n = X.shape[0]
    if n < 2 or val_fraction <= 0.0:
        return X, y, X[:0], y[:0], dataset.y_train_labels, None

    rng = np.random.default_rng(seed)
    if dataset.task == "classification" and dataset.y_train_labels is not None:
        labels = dataset.y_train_labels
        train_idx = []
        val_idx = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * val_fraction)))
            val_idx.extend(cls_idx[:n_val])
            train_idx.extend(cls_idx[n_val:])
        train_idx = np.array(train_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)
    else:
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * val_fraction)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    y_train_labels = dataset.y_train_labels[train_idx] if dataset.y_train_labels is not None else None
    y_val_labels = dataset.y_train_labels[val_idx] if dataset.y_train_labels is not None else None
    return X_train, y_train, X_val, y_val, y_train_labels, y_val_labels


def _history_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {}
    train_losses = [h.get("train_loss") for h in history if h.get("train_loss") is not None]
    val_losses = [h.get("val_loss") for h in history if h.get("val_loss") is not None]
    epoch_times = [h.get("epoch_time_s") for h in history if h.get("epoch_time_s") is not None]
    step_times = [h.get("step_time_s_mean") for h in history if h.get("step_time_s_mean") is not None]
    grad_norms = [h.get("grad_norm_max") for h in history if h.get("grad_norm_max") is not None]
    loss_nonfinite = sum(int(h.get("loss_nonfinite_steps") or 0) for h in history)
    grad_nonfinite = sum(int(h.get("grad_nonfinite_steps") or 0) for h in history)
    if train_losses:
        mean = float(sum(train_losses) / len(train_losses))
        var = float(sum((val - mean) ** 2 for val in train_losses) / len(train_losses))
        loss_vol = math.sqrt(var)
        mad = (
            float(sum(abs(train_losses[i] - train_losses[i - 1]) for i in range(1, len(train_losses))))
            / float(max(1, len(train_losses) - 1))
        )
    else:
        loss_vol = None
        mad = None
    if val_losses:
        mean_val = float(sum(val_losses) / len(val_losses))
        var_val = float(sum((val - mean_val) ** 2 for val in val_losses) / len(val_losses))
        val_vol = math.sqrt(var_val)
    else:
        val_vol = None
    return {
        "history_length": int(len(history)),
        "loss_curve_volatility": loss_vol,
        "loss_curve_mean_abs_diff": mad,
        "val_curve_volatility": val_vol,
        "epoch_time_s_mean": float(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        "step_time_s_mean": float(sum(step_times) / len(step_times)) if step_times else None,
        "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
        "loss_nonfinite_steps": int(loss_nonfinite),
        "grad_nonfinite_steps": int(grad_nonfinite),
        "train_time_s_total": history[-1].get("train_time_s_total"),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _count_params(estimator: Any) -> int:
    model = getattr(estimator, "model_", None)
    if model is None:
        return 0
    return int(psann_count_params(model))


def _run_single(
    model: ModelSpec,
    dataset: DatasetBundle,
    *,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    val_fraction: float,
    scale_y: bool,
    save_model_path: Optional[Path] = None,
    save_preds_path: Optional[Path] = None,
) -> Dict[str, Any]:
    seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    params = dict(model.params)
    params.update(
        {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "device": device,
            "random_state": int(seed),
            "target_scaler": "standard" if scale_y else None,
        }
    )
    estimator = model.estimator(**params)
    start = time.perf_counter()
    X_train, y_train, X_val, y_val, y_train_labels, y_val_labels = _split_train_val(
        dataset, val_fraction=float(val_fraction), seed=int(seed)
    )
    validation_data = (X_val, y_val) if X_val.size else None
    estimator.fit(X_train, y_train, verbose=0, validation_data=validation_data)
    if save_model_path is not None:
        save_model_path.parent.mkdir(parents=True, exist_ok=True)
        estimator.save(str(save_model_path))
    elapsed = time.perf_counter() - start

    pred_train = estimator.predict(X_train)
    pred_val = estimator.predict(X_val) if X_val.size else None
    pred_test = estimator.predict(dataset.X_test)
    result: Dict[str, Any] = {
        "train_time_s": float(elapsed),
        "n_params": _count_params(estimator),
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(dataset.X_test.shape[0]),
        "scale_y": bool(scale_y),
    }
    history = getattr(estimator, "history_", []) or []
    result.update(_history_stats(history))
    if dataset.task == "classification":
        if y_train_labels is None or dataset.y_test_labels is None:
            raise ValueError("classification dataset missing label arrays")
        num_classes = int(dataset.meta.get("classes", pred_train.shape[-1]))
        metrics_train = _classification_metrics(
            y_train_labels, pred_train, num_classes=num_classes
        )
        metrics_train.update(_regression_metrics(y_train, pred_train))
        result["metrics_train"] = metrics_train
        if pred_val is not None and y_val_labels is not None:
            metrics_val = _classification_metrics(
                y_val_labels, pred_val, num_classes=num_classes
            )
            metrics_val.update(_regression_metrics(y_val, pred_val))
            result["metrics_val"] = metrics_val
        metrics_test = _classification_metrics(
            dataset.y_test_labels, pred_test, num_classes=num_classes
        )
        metrics_test.update(_regression_metrics(dataset.y_test, pred_test))
        result["metrics_test"] = metrics_test
    else:
        result["metrics_train"] = _regression_metrics(y_train, pred_train)
        if pred_val is not None and y_val.size:
            result["metrics_val"] = _regression_metrics(y_val, pred_val)
        result["metrics_test"] = _regression_metrics(dataset.y_test, pred_test)
    if save_preds_path is not None:
        save_preds_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y_train": y_train,
            "y_val": y_val,
            "y_test": dataset.y_test,
            "pred_train": pred_train,
            "pred_val": pred_val,
            "pred_test": pred_test,
        }
        if y_train_labels is not None:
            payload["y_train_labels"] = y_train_labels
        if y_val_labels is not None:
            payload["y_val_labels"] = y_val_labels
        if dataset.y_test_labels is not None:
            payload["y_test_labels"] = dataset.y_test_labels
        np.savez_compressed(save_preds_path, **payload)
    return result


def _default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("reports") / "ablations" / f"{stamp}_regressor_ablations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        default="tabular_sine,tabular_shifted,classification_clusters,context_rotating_moons,ts_periodic,ts_regime_switch,ts_drift,ts_shock",
        help="Comma-separated dataset names to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="res_base,res_relu_sigmoid_psann,res_drop_path,res_no_norm,wrn_base,wrn_no_phase,wrn_no_film,wrn_spec_gate_rfft,wrn_spec_gate_feats,sgr_base,sgr_no_gate,sgr_fourier_feats,sgr_no_phase",
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
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of training data held out for validation (0 disables).",
    )
    parser.add_argument(
        "--scale-y",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Standardize target values using train split (metrics stay in original scale).",
    )
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


def _aggregate_seed_summary(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        dataset = str(row.get("dataset"))
        model = str(row.get("model"))
        groups.setdefault((dataset, model), []).append(row)

    def _mean_std(values: List[float]) -> Tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        mean = float(sum(values) / len(values))
        var = float(sum((v - mean) ** 2 for v in values) / len(values))
        return mean, math.sqrt(var)

    allowed_prefixes = (
        "metrics_",
        "train_time_s",
        "train_time_s_total",
        "n_params",
        "loss_curve_volatility",
        "loss_curve_mean_abs_diff",
        "val_curve_volatility",
        "grad_norm_max",
        "epoch_time_s_mean",
        "step_time_s_mean",
    )
    summary_rows: List[Dict[str, Any]] = []
    for (dataset, model), entries in sorted(groups.items()):
        row: Dict[str, Any] = {"dataset": dataset, "model": model, "n_runs": len(entries)}
        keys = set()
        for entry in entries:
            keys.update(entry.keys())
        for key in sorted(keys):
            if key in {"dataset", "model", "seed", "status"}:
                continue
            if not key.startswith(allowed_prefixes):
                continue
            vals = []
            for entry in entries:
                val = entry.get(key)
                if isinstance(val, (int, float)):
                    vals.append(float(val))
            if not vals:
                continue
            mean, std = _mean_std(vals)
            row[f"{key}_mean"] = mean
            row[f"{key}_std"] = std
        summary_rows.append(row)
    return summary_rows


def _flatten_result(entry: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in entry.items():
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                flat[f"{key}_{sub_k}"] = sub_v
        else:
            flat[key] = value
    return flat


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


if __name__ == "__main__":
    main()
