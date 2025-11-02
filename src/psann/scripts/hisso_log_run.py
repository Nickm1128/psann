"""
Unified HISSO logging CLI for remote or batch executions.

This module implements ``python -m psann.scripts.hisso_log_run`` which
loads a configuration file, runs a HISSO training session, and emits
structured metrics/logs ready to sync back into the repository.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import functools
import importlib
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from psann.metrics import portfolio_metrics
from psann.utils import seed_all

EVENT_LOGGER_NAME = "psann.hisso_log"


@dataclass
class DatasetArtifacts:
    X_train: np.ndarray
    y_train: Optional[np.ndarray]
    context_train: Optional[np.ndarray]
    X_val: Optional[np.ndarray]
    y_val: Optional[np.ndarray]
    context_val: Optional[np.ndarray]
    X_test: Optional[np.ndarray]
    y_test: Optional[np.ndarray]
    prices_train: Optional[np.ndarray]
    prices_val: Optional[np.ndarray]
    prices_test: Optional[np.ndarray]
    extras: Dict[str, Any]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to JSON/YAML configuration file.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directory for logs and checkpoints. Recommended: 'runs/hisso/' on local shells; "
            "'/content/hisso_logs/' on Colab/Runpod."
        ),
    )
    parser.add_argument("--run-name", default=None, help="Optional run identifier for output naming.")
    parser.add_argument("--device", default=None, help="Torch device override (e.g., cpu, cuda:0).")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Retain additional checkpoints beyond the best estimator snapshot.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages to stdout in addition to the event log.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "PyYAML is required to load YAML configs; install with `pip install pyyaml`."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        raise ValueError(f"Configuration at {path} is empty.")
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping; received {type(data).__name__}.")
    return data


def _resolve_target(target: str) -> Any:
    if not isinstance(target, str) or "." not in target:
        raise ValueError(f"Target '{target}' must be a dotted import path.")
    module, attr = target.rsplit(".", 1)
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - defensive
        raise ImportError(f"Unable to import module '{module}' from target '{target}'.") from exc
    if not hasattr(mod, attr):
        raise AttributeError(f"Module '{module}' does not define attribute '{attr}'.")
    return getattr(mod, attr)


def _as_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        return tensor.numpy().astype(np.float32, copy=False)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"Unable to coerce value of type {type(value).__name__} to ndarray.") from exc


def _coerce_dataset(result: Any) -> Dict[str, Any]:
    if isinstance(result, Mapping):
        out: Dict[str, Any] = {}
        for key, value in result.items():
            if isinstance(value, (np.ndarray, list, tuple)) or torch.is_tensor(value):
                out[key] = _as_numpy(value)
            else:
                out[key] = value
        return out
    if isinstance(result, (list, tuple)):
        if len(result) == 2:
            return {"X_train": _as_numpy(result[0]), "y_train": _as_numpy(result[1])}
        if len(result) == 3:
            return {
                "X_train": _as_numpy(result[0]),
                "y_train": _as_numpy(result[1]),
                "X_val": _as_numpy(result[2]),
            }
    raise TypeError(
        "Dataset loader must return a mapping or a tuple (X, y[, X_val]]); "
        f"received {type(result).__name__}."
    )


def _load_dataset(config: Mapping[str, Any], base_dir: Path) -> DatasetArtifacts:
    if not config:
        raise ValueError("Configuration must include a 'data' section.")
    if "npz" in config:
        npz_path = Path(config["npz"])
        if not npz_path.is_absolute():
            npz_path = base_dir / npz_path
        with np.load(npz_path, allow_pickle=False) as data:
            dataset = {key: _as_numpy(data[key]) for key in data.files}
    elif "loader" in config:
        loader = _resolve_target(str(config["loader"]))
        kwargs = dict(config.get("kwargs", {}))
        dataset = _coerce_dataset(loader(**kwargs))
    else:
        raise ValueError("Dataset configuration requires either 'npz' or 'loader'.")

    extras: Dict[str, Any] = {}
    keys = dataset.keys()
    get = dataset.get

    def _extract(name: str) -> Optional[np.ndarray]:
        value = get(name)
        return _as_numpy(value) if value is not None else None

    X_train = _extract("X_train")
    if X_train is None:
        raise ValueError("Dataset must provide 'X_train'.")

    y_train = _extract("y_train")
    context_train = _extract("context_train")

    X_val = _extract("X_val")
    y_val = _extract("y_val")
    context_val = _extract("context_val")

    X_test = _extract("X_test")
    y_test = _extract("y_test")

    prices_train = _extract("prices_train")
    prices_val = _extract("prices_val")
    prices_test = _extract("prices_test")

    for key in keys:
        if key not in {
            "X_train",
            "y_train",
            "context_train",
            "X_val",
            "y_val",
            "context_val",
            "X_test",
            "y_test",
            "prices_train",
            "prices_val",
            "prices_test",
        }:
            extras[key] = dataset[key]

    return DatasetArtifacts(
        X_train=X_train,
        y_train=y_train,
        context_train=context_train,
        X_val=X_val,
        y_val=y_val,
        context_val=context_val,
        X_test=X_test,
        y_test=y_test,
        prices_train=prices_train,
        prices_val=prices_val,
        prices_test=prices_test,
        extras=extras,
    )


def _prepare_run_directory(output_dir: Path, run_name: Optional[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = output_dir / run_name
    else:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logging(run_dir: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console_level = logging.INFO if verbose else logging.WARNING
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    events_path = run_dir / "events.csv"
    file_handler = logging.FileHandler(events_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s,%(levelname)s,%(message)s"))
    logger.addHandler(file_handler)

    if events_path.stat().st_size == 0:
        events_path.write_text("timestamp,level,message\n", encoding="utf-8")

    return logger


def _maybe_partial(callable_target: Optional[str], kwargs: Mapping[str, Any]) -> Optional[Callable]:
    if callable_target is None:
        return None
    fn = _resolve_target(callable_target)
    if kwargs:
        return functools.partial(fn, **kwargs)
    return fn


def _build_supervised_payload(
    supervised_cfg: Any,
    dataset: DatasetArtifacts,
) -> Optional[Mapping[str, Any]]:
    if supervised_cfg in (None, False):
        return None
    y_source = dataset.y_train
    if supervised_cfg is True:
        if y_source is None:
            raise ValueError("hisso.supervised=True requires 'y_train' in the dataset.")
        return {"y": y_source}
    if isinstance(supervised_cfg, Mapping):
        cfg = dict(supervised_cfg)
        y_key = cfg.pop("y_key", "y_train")
        if y_key == "y_train":
            y_value = dataset.y_train
        else:
            value = dataset.extras.get(y_key)
            y_value = _as_numpy(value) if value is not None else None
        if y_value is None:
            raise ValueError(f"hisso.supervised y_key='{y_key}' not found in dataset.")
        cfg["y"] = y_value
        return cfg
    raise TypeError("hisso.supervised must be a bool or mapping.")


def _yaml_dump(data: Any) -> str:
    def _scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            if value == "" or any(ch in value for ch in ":#{}[],&*?|\n\t"):
                return json.dumps(value)
            return value
        return json.dumps(value)

    def _dump(value: Any, indent: int) -> list[str]:
        pad = " " * indent
        if isinstance(value, Mapping):
            lines: list[str] = []
            for key in sorted(value.keys()):
                v = value[key]
                if isinstance(v, (Mapping, list, tuple)):
                    lines.append(f"{pad}{key}:")
                    lines.extend(_dump(v, indent + 2))
                else:
                    lines.append(f"{pad}{key}: {_scalar(v)}")
            if not lines:
                lines.append(f"{pad}{{}}")
            return lines
        if isinstance(value, (list, tuple)):
            lines = []
            if not value:
                lines.append(f"{pad}[]")
            for item in value:
                if isinstance(item, (Mapping, list, tuple)):
                    lines.append(f"{pad}-")
                    lines.extend(_dump(item, indent + 2))
                else:
                    lines.append(f"{pad}- {_scalar(item)}")
            return lines
        return [f"{pad}{_scalar(value)}"]

    return "\n".join(_dump(data, 0)) + "\n"


def _shape_or_none(array: Optional[np.ndarray]) -> Optional[list[int]]:
    return list(array.shape) if isinstance(array, np.ndarray) else None


def _compute_mse(y_true: Optional[np.ndarray], y_pred: Optional[np.ndarray]) -> Optional[float]:
    if y_true is None or y_pred is None:
        return None
    if y_true.shape != y_pred.shape:
        try:
            y_pred = y_pred.reshape(y_true.shape)
        except ValueError:
            return None
    diff = y_pred.astype(np.float32) - y_true.astype(np.float32)
    return float(np.mean(diff ** 2))


def toy_hisso_dataset(
    *,
    steps: int = 192,
    features: int = 4,
    seed: int = 0,
    train_fraction: float = 0.6,
    val_fraction: float = 0.2,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, steps, dtype=np.float32)
    harmonics = np.stack([np.sin((idx + 1) * t) for idx in range(features)], axis=1)
    noise = 0.05 * rng.standard_normal(size=harmonics.shape).astype(np.float32)
    X = harmonics + noise
    weights = rng.uniform(-0.5, 0.5, size=(features, 3)).astype(np.float32)
    y = X @ weights
    primary_dim = y.shape[1]

    n_train = max(8, int(steps * train_fraction))
    n_val = max(4, int(steps * val_fraction))
    n_test = steps - n_train - n_val
    if n_test <= 0:
        n_test = max(4, steps - n_train - n_val)
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n_train + n_val + n_test)

    def _prices(split: np.ndarray) -> np.ndarray:
        # Ensure prices align with the HISSO primary dimension by deriving a positive trajectory.
        cumulative = np.cumsum(np.abs(split) + 1.0, axis=0)
        return cumulative[:, :primary_dim].astype(np.float32)

    return {
        "X_train": X[train_slice],
        "y_train": y[train_slice],
        "X_val": X[val_slice] if n_val > 0 else None,
        "y_val": y[val_slice] if n_val > 0 else None,
        "X_test": X[test_slice] if n_test > 0 else None,
        "y_test": y[test_slice] if n_test > 0 else None,
        "prices_train": _prices(y[train_slice]),
        "prices_val": _prices(y[val_slice]) if n_val > 0 else None,
        "prices_test": _prices(y[test_slice]) if n_test > 0 else None,
    }


def _log_dataset_shapes(logger: logging.Logger, dataset: DatasetArtifacts) -> None:
    logger.info(
        "dataset.shapes train=%s val=%s test=%s",
        _shape_or_none(dataset.X_train),
        _shape_or_none(dataset.X_val),
        _shape_or_none(dataset.X_test),
    )
    if dataset.y_train is not None:
        logger.debug("dataset.targets train=%s", _shape_or_none(dataset.y_train))
    if dataset.context_train is not None:
        logger.debug("dataset.context train=%s", _shape_or_none(dataset.context_train))
    if dataset.prices_train is not None:
        logger.debug("dataset.prices train=%s", _shape_or_none(dataset.prices_train))


def _extract_history_metrics(history: list[dict]) -> dict:
    rewards = [
        float(entry["reward"])
        for entry in history
        if entry is not None and isinstance(entry, Mapping) and entry.get("reward") is not None
    ]
    best_epoch = None
    if rewards:
        best_idx = int(np.argmax(rewards))
        best_epoch = int(history[best_idx].get("epoch", best_idx + 1))
    reward_mean = float(np.mean(rewards)) if rewards else None
    reward_std = float(np.std(rewards)) if rewards else None
    return {
        "best_epoch": best_epoch,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }


def _episodes_from_history(history: list[dict]) -> int:
    total = 0
    for entry in history:
        if isinstance(entry, Mapping):
            total += int(entry.get("episodes", 0))
    return total


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    run_dir = _prepare_run_directory(Path(args.output_dir).resolve(), args.run_name)
    logger = _setup_logging(run_dir, args.verbose)

    metrics_path = run_dir / "metrics.json"
    resolved_config_path = run_dir / "config_resolved.yaml"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    resolved: Dict[str, Any] = {
        "seed": int(args.seed),
        "config_path": str(config_path),
        "output_dir": str(run_dir),
        "timestamp": _dt.datetime.now().isoformat(),
    }

    metrics: Dict[str, Any] = {
        "status": "running",
        "start_time": _dt.datetime.now().isoformat(),
    }

    try:
        dataset = _load_dataset(config.get("data", {}), config_path.parent)
        _log_dataset_shapes(logger, dataset)

        estimator_cfg = config.get("estimator", {})
        target = estimator_cfg.get("target", "psann.PSANNRegressor")
        EstimatorCls = _resolve_target(str(target))
        estimator_params = dict(estimator_cfg.get("params", {}))
        if args.device is not None:
            estimator_params["device"] = args.device
        device_override = estimator_params.get("device")
        resolved["estimator"] = {
            "target": str(target),
            "params": estimator_params,
        }

        seed_all(args.seed)

        estimator = EstimatorCls(**estimator_params)

        hisso_cfg = config.get("hisso", {})
        hisso_enabled = bool(hisso_cfg.get("enabled", True))
        mixed_precision = bool(hisso_cfg.get("mixed_precision", False))
        amp_dtype_name = hisso_cfg.get("amp_dtype", "float16")
        amp_dtype = getattr(torch, amp_dtype_name, torch.float16)

        reward_spec = hisso_cfg.get("reward")
        reward_fn = None
        if isinstance(reward_spec, Mapping):
            reward_fn = _maybe_partial(
                reward_spec.get("target"),
                reward_spec.get("kwargs", {}),
            )
        elif isinstance(reward_spec, str):
            reward_fn = _resolve_target(reward_spec)

        context_spec = hisso_cfg.get("context_extractor")
        context_extractor = None
        if isinstance(context_spec, Mapping):
            context_extractor = _maybe_partial(
                context_spec.get("target"),
                context_spec.get("kwargs", {}),
            )
        elif isinstance(context_spec, str):
            context_extractor = _resolve_target(context_spec)

        supervised_payload = _build_supervised_payload(hisso_cfg.get("supervised"), dataset)

        fit_kwargs: Dict[str, Any] = {
            "context": dataset.context_train,
            "validation_data": None,
            "verbose": int(config.get("training", {}).get("verbose", 0)),
            "noisy": hisso_cfg.get("input_noise"),
            "hisso": hisso_enabled,
        }

        if dataset.X_val is not None and dataset.y_val is not None:
            if dataset.context_val is not None:
                fit_kwargs["validation_data"] = (dataset.X_val, dataset.y_val, dataset.context_val)
            else:
                fit_kwargs["validation_data"] = (dataset.X_val, dataset.y_val)

        if hisso_enabled:
            fit_kwargs.update(
                {
                    "hisso_window": hisso_cfg.get("window"),
                    "hisso_reward_fn": reward_fn,
                    "hisso_context_extractor": context_extractor,
                    "hisso_primary_transform": hisso_cfg.get("primary_transform"),
                    "hisso_transition_penalty": hisso_cfg.get("transition_penalty"),
                    "hisso_trans_cost": hisso_cfg.get("trans_cost"),
                    "hisso_supervised": supervised_payload,
                    "lr_max": hisso_cfg.get("lr_max"),
                    "lr_min": hisso_cfg.get("lr_min"),
                }
            )
        else:
            fit_kwargs.update({"hisso": False})

        if hisso_enabled and mixed_precision and device_override:
            dev = torch.device(device_override)
            if dev.type == "cuda" and torch.cuda.is_available():
                setattr(estimator, "_hisso_use_amp", True)
                setattr(estimator, "_hisso_amp_dtype", amp_dtype)
                logger.info(
                    "mixed_precision.enabled device=%s amp_dtype=%s",
                    dev,
                    amp_dtype_name,
                )
            else:
                logger.warning(
                    "mixed_precision requested but CUDA device not available (device=%s).",
                    dev,
                )

        stateful = bool(getattr(estimator, "stateful", False))
        state_reset = getattr(estimator, "state_reset", "batch")
        warmstart_shuffle = None
        if supervised_payload is not None:
            warmstart_shuffle = not (stateful and state_reset in ("epoch", "none"))
        logger.info(
            "trainer.state stateful=%s state_reset=%s warmstart_shuffle=%s",
            stateful,
            state_reset,
            warmstart_shuffle,
        )

        X_train = dataset.X_train
        y_train = dataset.y_train
        if not hisso_enabled and y_train is None:
            raise ValueError("Supervised training requires 'y_train' targets.")

        start = time.perf_counter()
        estimator.fit(X_train, y_train, **fit_kwargs)
        duration = time.perf_counter() - start

        trainer = getattr(estimator, "_hisso_trainer_", None)
        history = getattr(estimator, "history_", []) or []
        if trainer is not None and history:
            for entry in history:
                if not isinstance(entry, Mapping):
                    continue
                logger.debug(
                    "epoch=%s reward=%s episodes=%s",
                    entry.get("epoch"),
                    entry.get("reward"),
                    entry.get("episodes"),
                )

        preds_train = estimator.predict(X_train)
        preds_val = estimator.predict(dataset.X_val) if dataset.X_val is not None else None
        preds_test = estimator.predict(dataset.X_test) if dataset.X_test is not None else None

        train_loss = _compute_mse(y_train, preds_train)
        val_loss = _compute_mse(dataset.y_val, preds_val)
        test_loss = _compute_mse(dataset.y_test, preds_test)

        metrics.update(
            {
                "status": "success",
                "duration_seconds": duration,
                "device": str(getattr(estimator, "_device", lambda: torch.device("cpu"))()),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            }
        )

        if trainer is not None:
            trainer_profile = dict(getattr(trainer, "profile", {}))
            history_metrics = _extract_history_metrics(history)
            episodes = trainer_profile.get("episodes_sampled")
            if episodes is None:
                episodes = _episodes_from_history(history)
            profile_time = float(trainer_profile.get("total_time_s", duration))
            throughput = float(episodes) / max(profile_time, 1e-6) if episodes else None
            metrics.update(
                {
                    "hisso": {
                        "best_epoch": history_metrics["best_epoch"],
                        "reward_mean": history_metrics["reward_mean"],
                        "reward_std": history_metrics["reward_std"],
                        "episodes": episodes,
                        "transition_penalty": getattr(trainer.cfg, "transition_penalty", None),
                        "throughput_eps_per_sec": throughput,
                        "profile": trainer_profile,
                    }
                }
            )
        else:
            metrics["hisso"] = None

        evaluation_cfg = config.get("evaluation", {})
        if evaluation_cfg and preds_test is not None and dataset.prices_test is not None:
            prices_key = evaluation_cfg.get("portfolio_prices_key", "prices_test")
            trans_cost = float(evaluation_cfg.get("trans_cost", 0.0))
            prices = dataset.prices_test
            if prices_key == "prices_val" and dataset.prices_val is not None:
                prices = dataset.prices_val
                preds_eval = preds_val
            else:
                preds_eval = preds_test
            metrics["portfolio_metrics"] = portfolio_metrics(preds_eval, prices, trans_cost=trans_cost)

        metrics["history_length"] = len(history)
        metrics["timestamp"] = _dt.datetime.now().isoformat()

        best_path = checkpoints_dir / "best.pt"
        estimator.save(str(best_path))
        if args.keep_checkpoints:
            latest_path = checkpoints_dir / "latest.pt"
            estimator.save(str(latest_path))

        resolved.update(
            {
                "hisso": {
                    "enabled": hisso_enabled,
                    "window": hisso_cfg.get("window"),
                    "primary_transform": hisso_cfg.get("primary_transform"),
                    "transition_penalty": hisso_cfg.get("transition_penalty"),
                    "trans_cost": hisso_cfg.get("trans_cost"),
                    "episodes_per_batch": getattr(
                        getattr(trainer, "cfg", None), "episodes_per_batch", None
                    )
                    if trainer is not None
                    else None,
                    "mixed_precision": mixed_precision,
                    "amp_dtype": amp_dtype_name if mixed_precision else None,
                },
                "data": {
                    "train_shape": _shape_or_none(dataset.X_train),
                    "val_shape": _shape_or_none(dataset.X_val),
                    "test_shape": _shape_or_none(dataset.X_test),
                },
                "checkpoints": {
                    "best": str(best_path),
                },
            }
        )

        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        resolved_config_path.write_text(_yaml_dump(resolved), encoding="utf-8")
        logger.info("run.completed duration=%.3fs", duration)
        return 0
    except Exception as exc:  # pragma: no cover - exercised via tests expecting success
        metrics["status"] = "error"
        metrics["error"] = "".join(traceback.format_exception(exc))
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        logger.error("run.failed %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
