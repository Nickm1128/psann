#!/usr/bin/env python3
"""Run GeoSparse activation variants vs dense ReLU regression benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.datasets import (
    fetch_california_housing,
    load_diabetes,
    load_linnerud,
    make_friedman1,
    make_regression,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from psann import GeoSparseRegressor, PSANNRegressor, count_params  # noqa: E402
from psann.nn import PSANNNet  # noqa: E402
from psann.nn_geo_sparse import GeoSparseNet  # noqa: E402


DATASET_NAMES = [
    "syn_sparse_linear",
    "syn_friedman1",
    "syn_piecewise_sine",
    "real_california_housing",
    "real_diabetes",
    "real_linnerud",
]
DEFAULT_GEO_ACTIVATIONS = "psann,relu_sigmoid_psann"


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: Optional[StandardScaler]


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def resolve_geo_shape(input_dim: int, shape: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    if shape is not None:
        return int(shape[0]), int(shape[1])
    return 1, int(input_dim)


def _maybe_cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _compute_epochs(n_train: int, batch_size: int, target_steps: int) -> Tuple[int, int]:
    steps_per_epoch = int(math.ceil(n_train / batch_size))
    epochs = int(math.ceil(target_steps / steps_per_epoch))
    return max(1, epochs), steps_per_epoch


def _subsample(X: np.ndarray, y: np.ndarray, n_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_samples is None or len(X) <= n_samples:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=int(n_samples), replace=False)
    return X[idx], y[idx]


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    test_size: float,
    val_size: float,
    scale_y: bool,
) -> DatasetSplit:
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=test_size + val_size, random_state=seed
    )
    val_frac = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1.0 - val_frac, random_state=seed
    )

    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train).astype(np.float32, copy=False)
    X_val_s = x_scaler.transform(X_val).astype(np.float32, copy=False)
    X_test_s = x_scaler.transform(X_test).astype(np.float32, copy=False)

    y_scaler = None
    if scale_y:
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    y_train = y_train.astype(np.float32, copy=False)
    y_val = y_val.astype(np.float32, copy=False)
    y_test = y_test.astype(np.float32, copy=False)

    return DatasetSplit(
        X_train=X_train_s,
        y_train=y_train,
        X_val=X_val_s,
        y_val=y_val,
        X_test=X_test_s,
        y_test=y_test,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def load_dataset(name: str, *, seed: int) -> Tuple[np.ndarray, np.ndarray, str]:
    if name == "syn_sparse_linear":
        X, y = make_regression(
            n_samples=5000,
            n_features=200,
            n_informative=10,
            noise=5.0,
            bias=10.0,
            random_state=seed,
        )
    elif name == "syn_friedman1":
        X, y = make_friedman1(
            n_samples=5000,
            n_features=20,
            noise=0.5,
            random_state=seed,
        )
    elif name == "syn_piecewise_sine":
        rng = np.random.default_rng(seed)
        n_samples = 5000
        n_features = 10
        X = rng.normal(size=(n_samples, n_features))
        t = X[:, 0]
        y = np.where(
            t < 0,
            0.5 * np.sin(3.0 * t) + 0.2 * t,
            1.0 * np.sin(6.0 * t + 0.5) - 0.1 * t + 0.5,
        )
        y = y + 0.1 * rng.normal(size=n_samples)
    elif name == "real_california_housing":
        data = fetch_california_housing()
        X, y = data.data, data.target
    elif name == "real_diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
    elif name == "real_linnerud":
        data = load_linnerud()
        X = data.data
        y = data.target[:, 0]
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y, "regression"


@lru_cache(maxsize=None)
def count_geosparse_params(
    input_dim: int,
    output_dim: int,
    *,
    depth: int,
    k: int,
    shape: Optional[Tuple[int, int]],
    activation_type: str,
    activation_config_json: Optional[str] = None,
) -> int:
    activation_config = (
        json.loads(str(activation_config_json)) if activation_config_json is not None else None
    )
    model = GeoSparseNet(
        int(input_dim),
        int(output_dim),
        shape=resolve_geo_shape(input_dim, shape),
        depth=int(depth),
        k=int(k),
        activation_type=activation_type,
        activation_config=activation_config,
        norm="rms",
        drop_path_max=0.0,
        residual_alpha_init=0.0,
        bias=True,
        compute_mode="gather",
        seed=1337,
    )
    return count_params(model, trainable_only=True)


@lru_cache(maxsize=None)
def count_dense_params(
    input_dim: int,
    output_dim: int,
    *,
    depth: int,
    hidden_units: int,
    activation_type: str,
) -> int:
    model = PSANNNet(
        int(input_dim),
        int(output_dim),
        hidden_layers=int(depth),
        hidden_units=int(hidden_units),
        hidden_width=int(hidden_units),
        activation_type=activation_type,
    )
    return count_params(model, trainable_only=True)


def _best_dense_width_for_target(
    *,
    input_dim: int,
    output_dim: int,
    dense_depth: int,
    target_params: int,
    max_width: int = 16384,
) -> Dict[str, int]:
    if max_width < 1:
        raise ValueError("max_width must be >= 1")

    def params_for(width: int) -> int:
        return count_dense_params(
            input_dim,
            output_dim,
            depth=dense_depth,
            hidden_units=int(width),
            activation_type="relu",
        )

    lo = 1
    hi = 1
    hi_params = params_for(hi)
    while hi_params < target_params and hi < max_width:
        hi = min(max_width, hi * 2)
        hi_params = params_for(hi)

    if hi_params < target_params:
        # Can't reach target within width budget; return best-at-max_width.
        mismatch = abs(hi_params - target_params)
        return {"dense_width": hi, "dense_params": hi_params, "mismatch": mismatch}

    # Binary search for smallest width whose params >= target.
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        mid_params = params_for(mid)
        if mid_params < target_params:
            lo = mid + 1
        else:
            hi = mid

    candidates = [lo]
    if lo > 1:
        candidates.append(lo - 1)
    # Guard against tiny non-monotonicity / initialization quirks.
    if lo + 1 <= max_width:
        candidates.append(lo + 1)

    best_width = None
    best_params = None
    best_mismatch = None
    for w in sorted(set(candidates)):
        p = params_for(w)
        mismatch = abs(p - target_params)
        if best_mismatch is None or mismatch < best_mismatch:
            best_width = w
            best_params = p
            best_mismatch = mismatch
    return {
        "dense_width": int(best_width),
        "dense_params": int(best_params),
        "mismatch": int(best_mismatch),
    }


def match_dense_width_to_geosparse(
    *,
    input_dim: int,
    output_dim: int,
    geo_depth: int,
    geo_k: int,
    geo_shape: Optional[Tuple[int, int]],
    geo_activation_type: str,
    geo_activation_config_json: Optional[str],
    dense_depth: int,
    tol: float,
) -> Dict[str, Any]:
    target = count_geosparse_params(
        input_dim,
        output_dim,
        depth=geo_depth,
        k=geo_k,
        shape=geo_shape,
        activation_type=geo_activation_type,
        activation_config_json=geo_activation_config_json,
    )
    preferred = _best_dense_width_for_target(
        input_dim=input_dim,
        output_dim=output_dim,
        dense_depth=dense_depth,
        target_params=target,
    )
    rel_mismatch = float(preferred["mismatch"]) / max(1, target)
    if rel_mismatch <= tol:
        return {
            "target_params": int(target),
            "dense_depth": int(dense_depth),
            "dense_width": int(preferred["dense_width"]),
            "dense_params": int(preferred["dense_params"]),
            "rel_mismatch": rel_mismatch,
        }

    # Fallback: for small targets, parameter granularity can make exact matching
    # impossible at higher depths. Try shallower dense models to honor tol.
    best = {
        "target_params": int(target),
        "dense_depth": int(dense_depth),
        "dense_width": int(preferred["dense_width"]),
        "dense_params": int(preferred["dense_params"]),
        "rel_mismatch": rel_mismatch,
        "_mismatch_abs": int(preferred["mismatch"]),
    }
    for depth in range(1, max(1, int(dense_depth))):
        cand = _best_dense_width_for_target(
            input_dim=input_dim,
            output_dim=output_dim,
            dense_depth=depth,
            target_params=target,
        )
        cand_rel = float(cand["mismatch"]) / max(1, target)
        if cand_rel < best["rel_mismatch"]:
            best = {
                "target_params": int(target),
                "dense_depth": int(depth),
                "dense_width": int(cand["dense_width"]),
                "dense_params": int(cand["dense_params"]),
                "rel_mismatch": cand_rel,
                "_mismatch_abs": int(cand["mismatch"]),
            }
        if cand_rel <= tol:
            best = {
                "target_params": int(target),
                "dense_depth": int(depth),
                "dense_width": int(cand["dense_width"]),
                "dense_params": int(cand["dense_params"]),
                "rel_mismatch": cand_rel,
                "_mismatch_abs": int(cand["mismatch"]),
            }
            break

    if best["rel_mismatch"] > tol:
        raise RuntimeError(
            f"Could not match dense params within tol={tol:.3f}: "
            f"target={target} best_dense={best['dense_params']} "
            f"(depth={best['dense_depth']} width={best['dense_width']}) "
            f"rel_mismatch={best['rel_mismatch']:.3f}"
        )
    if best["dense_depth"] != dense_depth:
        print(
            f"[match] dense_depth fallback {dense_depth} -> {best['dense_depth']} "
            f"to satisfy tol={tol:.3f} (abs_mismatch={best['_mismatch_abs']})"
        )
    best.pop("_mismatch_abs", None)
    return best


def build_geosparse_estimator(
    *,
    input_dim: int,
    shape: Optional[Tuple[int, int]],
    geo_depth: int,
    geo_k: int,
    activation_type: str,
    activation_config: Optional[Dict[str, Any]],
    amp: bool,
    amp_dtype: str,
    compile: bool,
    compile_backend: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> GeoSparseRegressor:
    return GeoSparseRegressor(
        hidden_layers=geo_depth,
        activation_type=activation_type,
        activation=activation_config,
        shape=resolve_geo_shape(input_dim, shape),
        k=geo_k,
        compute_mode="gather",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer="adam",
        weight_decay=0.0,
        amp=amp,
        amp_dtype=amp_dtype,
        compile=compile,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        device=device,
        random_state=seed,
    )


def build_dense_estimator(
    *,
    dense_depth: int,
    dense_width: int,
    amp: bool,
    amp_dtype: str,
    compile: bool,
    compile_backend: str,
    compile_mode: str,
    compile_fullgraph: bool,
    compile_dynamic: bool,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> PSANNRegressor:
    return PSANNRegressor(
        hidden_layers=dense_depth,
        hidden_units=dense_width,
        activation_type="relu",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer="adam",
        weight_decay=0.0,
        amp=amp,
        amp_dtype=amp_dtype,
        compile=compile,
        compile_backend=compile_backend,
        compile_mode=compile_mode,
        compile_fullgraph=compile_fullgraph,
        compile_dynamic=compile_dynamic,
        device=device,
        random_state=seed,
    )


def _attach_progress(model: Any, steps_per_epoch: int, progress_every_steps: int) -> None:
    next_report = {"step": progress_every_steps}

    def _epoch_callback(self, epoch: int, train_loss: float, val_loss: Optional[float], *_: Any) -> None:
        steps_done = (epoch + 1) * steps_per_epoch
        if steps_done >= next_report["step"]:
            msg = f"[step ~{steps_done}] train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            print(msg, flush=True)
            next_report["step"] += progress_every_steps

    model.epoch_callback = _epoch_callback.__get__(model, model.__class__)


def warmup_fit(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    warmup_samples: int,
    warmup_epochs: int,
) -> None:
    if not torch.cuda.is_available():
        return
    X_w, y_w = _subsample(X_train, y_train, warmup_samples, seed=123)
    model = model_factory(epochs=warmup_epochs)
    _maybe_cuda_sync()
    model.fit(X_w, y_w, validation_data=(X_val, y_val), verbose=0)
    _maybe_cuda_sync()


def fit_with_timing(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    target_steps: int,
    progress_every_steps: int,
    timing_warmup_epochs: int,
) -> Dict[str, Any]:
    epochs, steps_per_epoch = _compute_epochs(len(X_train), batch_size, target_steps)
    model = model_factory(epochs=epochs)

    warmup_epochs = int(max(0, timing_warmup_epochs))
    warmup_epochs = int(min(warmup_epochs, max(0, epochs - 1)))

    timing: Dict[str, Optional[float]] = {"start": None, "end": None}
    next_report = {"step": progress_every_steps}

    def _epoch_callback(self, epoch: int, train_loss: float, val_loss: Optional[float], *_: Any) -> None:
        steps_done = (epoch + 1) * steps_per_epoch
        if steps_done >= next_report["step"]:
            msg = f"[step ~{steps_done}] train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            print(msg, flush=True)
            next_report["step"] += progress_every_steps

        if warmup_epochs > 0 and timing["start"] is None and epoch + 1 == warmup_epochs:
            _maybe_cuda_sync()
            timing["start"] = time.perf_counter()
        if epoch + 1 == epochs:
            _maybe_cuda_sync()
            timing["end"] = time.perf_counter()

    model.epoch_callback = _epoch_callback.__get__(model, model.__class__)
    _maybe_cuda_sync()
    start_total = time.perf_counter()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    _maybe_cuda_sync()
    end_total = time.perf_counter()
    elapsed_total = end_total - start_total

    elapsed = elapsed_total
    if timing["start"] is not None and timing["end"] is not None:
        elapsed = max(0.0, float(timing["end"] - timing["start"]))
    total_steps = steps_per_epoch * epochs
    timed_epochs = epochs - warmup_epochs if elapsed != elapsed_total else epochs
    samples_seen = len(X_train) * timed_epochs
    timed_steps = steps_per_epoch * timed_epochs
    return {
        "model": model,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "timed_steps": timed_steps,
        "train_time_total_s": elapsed_total,
        "train_time_s": elapsed,
        "steps_per_sec": timed_steps / max(elapsed, 1e-9),
        "samples_per_sec": samples_seen / max(elapsed, 1e-9),
    }


def _unscale_y(y: np.ndarray, y_scaler: Optional[StandardScaler]) -> np.ndarray:
    if y_scaler is None:
        return y
    return y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()


def evaluate_regression(model: Any, split: DatasetSplit) -> Dict[str, float]:
    y_true = _unscale_y(split.y_test, split.y_scaler)
    y_pred = np.asarray(model.predict(split.X_test), dtype=np.float32)
    y_pred = _unscale_y(y_pred, split.y_scaler)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


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
    parser.add_argument("--amp", action="store_true", help="Enable autocast mixed precision (CUDA only).")
    parser.add_argument("--amp-dtype", default="bfloat16", help="Autocast dtype: bfloat16|float16|float32.")
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
                            "params": match["target_params"] if is_geo_model else match["dense_params"],
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


if __name__ == "__main__":
    raise SystemExit(main())
