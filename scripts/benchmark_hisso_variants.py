#!/usr/bin/env python
"""Benchmark dense vs. convolutional HISSO estimators across CPU/GPU devices.

The harness fits the residual dense and convolutional PSANN regressors using the
primary-only HISSO trainer, records wall-clock timing, and captures reward
trends for each configuration. Results are emitted as JSON so they can be
archived directly or post-processed later.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from psann import ResConvPSANNRegressor, ResPSANNRegressor


# ---------------------------------------------------------------------------
# Synthetic datasets
# ---------------------------------------------------------------------------

PRIMARY_DIM = 3


def _make_dense_series(
    seed: int,
    *,
    steps: int = 768,
    features: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32)

    base = rng.normal(scale=0.35, size=(steps, features)).astype(np.float32)
    freqs = np.linspace(1.0, 3.0, features, dtype=np.float32)
    oscillations = 0.25 * np.sin(np.outer(t, freqs))
    base += oscillations.astype(np.float32)

    y1 = np.sin(t) + 0.05 * rng.normal(size=steps).astype(np.float32)
    y2 = np.cos(0.5 * t) + 0.05 * rng.normal(size=steps).astype(np.float32)
    y3 = 0.4 * base[:, 0] - 0.2 * base[:, 1] + 0.1 * rng.normal(size=steps).astype(np.float32)
    targets = np.stack([y1, y2, y3], axis=1).astype(np.float32)
    return base, targets


def _make_conv_series(
    seed: int,
    *,
    steps: int = 512,
    channels: int = 2,
    height: int = 8,
    width: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, steps, dtype=np.float32)

    base = rng.normal(scale=0.3, size=(steps, channels, height, width)).astype(np.float32)
    t_broadcast = t.reshape(-1, 1, 1, 1)
    time_term = 0.2 * np.sin(t_broadcast)
    freq = np.arange(1, channels + 1, dtype=np.float32).reshape(1, -1, 1, 1)
    channel_term = 0.15 * np.cos(t_broadcast * freq)
    base += time_term
    base += channel_term.astype(np.float32)

    flat = base.reshape(steps, channels, -1)
    y1 = flat.mean(axis=(1, 2))
    y2 = flat[:, 0].mean(axis=1)
    y3 = flat[:, 1].std(axis=1)
    targets = np.stack([y1, y2, y3], axis=1).astype(np.float32)
    return base, targets


# ---------------------------------------------------------------------------
# Dataset specifications
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    description: str
    variants: Tuple[str, ...]
    build_fn: Callable[[str, int], Tuple[np.ndarray, np.ndarray]]
    metadata: Dict[str, Any]

    def supports(self, variant: str) -> bool:
        return variant in self.variants


def _load_portfolio_prices(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2), dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two feature columns in {path}")
    return data


def _portfolio_returns(prices: np.ndarray) -> np.ndarray:
    log_prices = np.log(np.clip(prices, a_min=1e-6, a_max=None))
    returns = np.vstack(
        [
            np.zeros((1, prices.shape[1]), dtype=np.float32),
            np.diff(log_prices, axis=0),
        ]
    ).astype(np.float32)
    return returns


def _portfolio_dense_series(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    returns = _portfolio_returns(prices)
    return returns.astype(np.float32), returns.astype(np.float32)


def _portfolio_conv_series(
    prices: np.ndarray,
    *,
    window: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    if window <= 0 or window % 4 != 0:
        raise ValueError("window must be a positive multiple of 4 for conv reshaping.")

    returns = _portfolio_returns(prices)
    if returns.shape[0] <= window:
        raise ValueError("Not enough timesteps for convolutional portfolio dataset.")

    segments: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    for idx in range(window, returns.shape[0]):
        window_slice = returns[idx - window : idx]
        segment = window_slice.T.reshape(returns.shape[1], 4, window // 4)
        segments.append(segment.astype(np.float32))
        targets.append(returns[idx].astype(np.float32))

    X = np.stack(segments, axis=0)
    y = np.stack(targets, axis=0)
    return X, y


def resolve_dataset(name: str, dataset_path: Optional[str]) -> DatasetSpec:
    name_lower = name.lower()
    if name_lower == "synthetic":
        def build(variant: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
            if variant == "dense":
                return _make_dense_series(seed)
            if variant == "conv":
                return _make_conv_series(seed)
            raise ValueError(f"Synthetic dataset does not support variant '{variant}'.")

        return DatasetSpec(
            name="synthetic",
            description="Synthetic sine-based series with analytic targets.",
            variants=("dense", "conv"),
            build_fn=build,
            metadata={"primary_dim": 3},
        )

    if name_lower == "portfolio":
        csv_path = Path(dataset_path) if dataset_path else Path("benchmarks") / "hisso_portfolio_prices.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Portfolio dataset not found at {csv_path}")
        prices = _load_portfolio_prices(csv_path)

        def build(variant: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
            _ = seed  # seed ignored for deterministic dataset
            if variant == "dense":
                return _portfolio_dense_series(prices)
            if variant == "conv":
                return _portfolio_conv_series(prices)
            raise ValueError(f"Portfolio dataset does not support variant '{variant}'.")

        return DatasetSpec(
            name="portfolio",
            description="Historical AAPL Open/Close prices converted to log returns.",
            variants=("dense", "conv"),
            build_fn=build,
            metadata={"source": str(csv_path)},
        )

    raise ValueError(f"Unknown dataset '{name}'. Available options: synthetic, portfolio.")


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _resolve_devices(raw: str | None) -> List[str]:
    if not raw or raw.lower() == "auto":
        return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    items = [entry.strip().lower() for entry in raw.split(",") if entry.strip()]
    resolved: List[str] = []
    for entry in items:
        if entry == "auto":
            resolved.extend(["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
            continue
        if entry == "cuda" and not torch.cuda.is_available():
            continue
        resolved.append(entry)
    return list(dict.fromkeys(resolved)) or ["cpu"]


def _build_estimator(variant: str, device: str, *, epochs: int, seed: int):
    common = dict(
        hidden_layers=2,
        hidden_units=64,
        epochs=epochs,
        batch_size=64,
        lr=1e-3,
        device=device,
        random_state=seed,
        patience=10,
        early_stopping=False,
    )
    if variant == "dense":
        return ResPSANNRegressor(**common)
    if variant == "conv":
        conv_params = dict(common)
        conv_params["hidden_units"] = 32
        return ResConvPSANNRegressor(conv_channels=32, **conv_params)
    raise ValueError(f"Unsupported variant '{variant}'. Expected 'dense' or 'conv'.")


def _nan_series(values: Iterable[float]) -> List[float]:
    result = []
    for value in values:
        if value is None:
            result.append(float("nan"))
        else:
            result.append(float(value))
    return result


def _reward_fn(primary: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
    penalties = primary.pow(2).mean(dim=-1)
    return -penalties


def _summarise_runs(runs: List[Dict[str, List[float] | float | int]]) -> Dict[str, object]:
    if not runs:
        return {
            "runs": 0,
            "mean_wall_time_s": math.nan,
            "std_wall_time_s": math.nan,
            "mean_profile_time_s": math.nan,
            "std_profile_time_s": math.nan,
            "reward_trend_mean": [],
            "reward_trend_std": [],
            "episodes_per_epoch_mean": [],
            "episodes_per_epoch_std": [],
            "final_reward_mean": math.nan,
            "final_reward_std": math.nan,
        }

    wall_times = np.array([run["wall_time_s"] for run in runs], dtype=np.float64)
    profile_times = np.array([run["profile_time_s"] for run in runs], dtype=np.float64)
    final_rewards = np.array([run["final_reward"] for run in runs], dtype=np.float64)
    reward_matrix = np.array([run["reward_trend"] for run in runs], dtype=np.float64)
    episodes_matrix = np.array([run["episodes_per_epoch"] for run in runs], dtype=np.float64)

    with np.errstate(invalid="ignore"):
        reward_mean = np.nanmean(reward_matrix, axis=0).tolist()
        reward_std = np.nanstd(reward_matrix, axis=0).tolist()
        final_reward_mean = float(np.nanmean(final_rewards)) if final_rewards.size else math.nan
        final_reward_std = float(np.nanstd(final_rewards)) if final_rewards.size > 1 else 0.0

    summary = {
        "runs": len(runs),
        "mean_wall_time_s": float(wall_times.mean()) if wall_times.size else math.nan,
        "std_wall_time_s": float(wall_times.std(ddof=0)) if wall_times.size > 1 else 0.0,
        "mean_profile_time_s": float(profile_times.mean()) if profile_times.size else math.nan,
        "std_profile_time_s": float(profile_times.std(ddof=0)) if profile_times.size > 1 else 0.0,
        "reward_trend_mean": reward_mean,
        "reward_trend_std": reward_std,
        "episodes_per_epoch_mean": episodes_matrix.mean(axis=0).tolist() if episodes_matrix.size else [],
        "episodes_per_epoch_std": episodes_matrix.std(axis=0, ddof=0).tolist() if len(runs) > 1 else [0.0] * episodes_matrix.shape[1],
        "final_reward_mean": final_reward_mean,
        "final_reward_std": final_reward_std,
    }
    return summary


def _benchmark_variant(
    variant: str,
    device: str,
    *,
    epochs: int,
    repeats: int,
    window: int,
    transition_penalty: float,
    warmstart_epochs: int,
    base_seed: int,
    dataset: DatasetSpec,
) -> Dict[str, object]:
    run_stats: List[Dict[str, object]] = []
    allow_full_window = True
    episode_length_seen: Optional[int] = None
    feature_shape: Optional[Tuple[int, ...]] = None

    for repeat in range(repeats):
        seed = base_seed + repeat
        X, y = dataset.build_fn(variant, seed)
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Dataset builder must return numpy arrays for inputs and targets.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Dataset inputs and targets must share the same first dimension.")
        X = X.astype(np.float32, copy=False)
        y = y.astype(np.float32, copy=False)

        estimator = _build_estimator(variant, device, epochs=epochs, seed=seed)

        primary_dim = int(y.shape[1]) if y.ndim > 1 else 1
        batch_size = min(128, X.shape[0]) if X.shape[0] > 0 else 1
        hisso_supervised = {
            "y": y,
            "epochs": max(1, warmstart_epochs),
            "batch_size": max(1, batch_size),
        }

        start = time.perf_counter()
        estimator.fit(
            X,
            y,
            hisso=True,
            hisso_window=window,
            hisso_reward_fn=_reward_fn,
            hisso_primary_transform="tanh",
            hisso_transition_penalty=transition_penalty,
            hisso_supervised=hisso_supervised,
            verbose=0,
        )
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        trainer = getattr(estimator, "_hisso_trainer_", None)
        if trainer is None:
            raise RuntimeError("HISSO trainer was not attached after fit().")

        cfg = getattr(estimator, "_hisso_cfg_", None)
        if cfg is None:
            raise RuntimeError("HISSO configuration missing after fit().")

        reward_trend = _nan_series(entry.get("reward") for entry in trainer.history)
        episodes = [int(entry.get("episodes", 0) or 0) for entry in trainer.history]

        allow_full_window = allow_full_window and getattr(cfg, "episode_length", window) == window
        episode_length_seen = getattr(cfg, "episode_length", window)
        feature_shape = tuple(int(dim) for dim in X.shape[1:])

        run_stats.append(
            {
                "wall_time_s": float(elapsed),
                "profile_time_s": float(trainer.profile.get("total_time_s", float("nan"))),
                "reward_trend": reward_trend,
                "episodes_per_epoch": episodes,
                "final_reward": float(reward_trend[-1]) if reward_trend else math.nan,
                "series_length": int(X.shape[0]),
                "episode_length": int(getattr(cfg, "episode_length", window)),
                "primary_dim": primary_dim,
                "feature_shape": feature_shape,
            }
        )

    summary = _summarise_runs(run_stats)
    summary.update(
        {
            "allow_full_window": bool(allow_full_window),
            "episode_length": int(episode_length_seen) if episode_length_seen is not None else int(window),
            "series_length": int(run_stats[0]["series_length"]) if run_stats else math.nan,
            "primary_dim": int(run_stats[0]["primary_dim"]) if run_stats else math.nan,
            "feature_shape": list(feature_shape) if feature_shape is not None else None,
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6, help="Number of HISSO epochs to train.")
    parser.add_argument("--window", type=int, default=64, help="Episode length/window used by HISSO.")
    parser.add_argument(
        "--transition-penalty",
        type=float,
        default=0.02,
        help="Transition penalty applied during HISSO optimisation.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of independent runs per variant/device (averaged in the output).",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Comma-separated list of devices to benchmark (cpu,cuda). Use 'auto' for availability-based defaults.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; increments by one for each repeat.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset to benchmark (synthetic | portfolio).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Optional CSV path for custom portfolio data.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="dense,conv",
        help="Comma-separated subset of variants to benchmark (e.g., 'dense' or 'dense,conv').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional file path to write the JSON payload (stdout if omitted).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = _resolve_devices(args.devices)
    variants = [entry.strip().lower() for entry in args.variants.split(",") if entry.strip()]
    if not variants:
        raise ValueError("No variants specified; provide at least one via --variants.")

    dataset = resolve_dataset(args.dataset, args.dataset_path)

    results = []
    for device in devices:
        for variant in variants:
            if not dataset.supports(variant):
                print(f"Skipping variant '{variant}' for dataset '{dataset.name}' (unsupported).")
                continue
            summary = _benchmark_variant(
                variant,
                device,
                epochs=max(1, int(args.epochs)),
                repeats=max(1, int(args.repeats)),
                window=max(1, int(args.window)),
                transition_penalty=float(args.transition_penalty),
                warmstart_epochs=max(1, int(args.epochs // 2)),
                base_seed=int(args.seed),
                dataset=dataset,
            )
            results.append(
                {
                    "variant": variant,
                    "device": device,
                    "dataset": dataset.name,
                    **summary,
                }
            )

    payload = {
        "metadata": {
            "epochs": int(args.epochs),
            "window": int(args.window),
            "transition_penalty": float(args.transition_penalty),
            "repeats": int(args.repeats),
            "devices": devices,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "variants": variants,
            "dataset": dataset.name,
        },
        "dataset": {
            "name": dataset.name,
            "description": dataset.description,
            **dataset.metadata,
        },
        "results": results,
    }
    output = args.output
    text = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved benchmark payload to {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
