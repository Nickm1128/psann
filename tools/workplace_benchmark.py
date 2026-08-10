#!/usr/bin/env python
"""Measure the Phase 7 CPU workplace path and compare an optional reference baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import torch

import psann


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def _measure(samples: int, repeats: int) -> tuple[dict[str, float], bool]:
    rng = np.random.default_rng(701)
    inputs = rng.normal(size=(samples, 8)).astype(np.float32)
    targets = (inputs[:, 0] - 0.5 * inputs[:, 1]).astype(np.float32)
    spec = psann.ModelSpec(
        input_schema=psann.DataSchema(input_shape=(8,)),
        activation="relu",
        parameters={"hidden_layers": 2, "hidden_units": 32, "random_state": 701},
    )
    model = psann.create_model(spec)
    tracemalloc.start()
    train_started = time.perf_counter()
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=2,
            batch_size=64,
            deterministic=True,
            device="cpu",
        ),
    )
    train_seconds = time.perf_counter() - train_started
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    with tempfile.TemporaryDirectory(prefix="psann-performance-") as directory:
        artifact = run.export(Path(directory) / "benchmark.psann")
        load_times: list[float] = []
        runtime = None
        for _ in range(repeats):
            started = time.perf_counter()
            runtime = psann.load_runtime(
                artifact,
                config=psann.InferenceConfig(batch_size=64, device="cpu"),
            )
            load_times.append((time.perf_counter() - started) * 1000.0)
        assert runtime is not None
        inference_times: list[float] = []
        result = None
        for _ in range(repeats):
            started = time.perf_counter()
            result = runtime.predict(inputs)
            inference_times.append((time.perf_counter() - started) * 1000.0)
        assert result is not None
        expected = model.predict(inputs)
        correctness = bool(np.allclose(result.values, expected, rtol=1e-5, atol=1e-6))

        explanation_ms: float | None = None
        if importlib.util.find_spec("shap") is not None:
            started = time.perf_counter()
            explained = runtime.explain(
                inputs[:4],
                background=inputs[4:12],
                config=psann.ExplainerConfig(
                    max_evaluations=32,
                    max_explanation_samples=4,
                    seed=701,
                ),
            )
            explanation_ms = (time.perf_counter() - started) * 1000.0
            correctness = correctness and explained.metadata["additivity_error"] < 1e-4

    metrics = {
        "training_samples_per_second": (samples * 2) / max(train_seconds, 1e-12),
        "inference_p50_ms": statistics.median(inference_times),
        "inference_p95_ms": _percentile(inference_times, 0.95),
        "peak_python_memory_bytes": float(peak_memory),
        "artifact_load_p50_ms": statistics.median(load_times),
    }
    if explanation_ms is not None:
        metrics["explanation_ms"] = explanation_ms
    return metrics, correctness


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "psann": psann.__version__,
        "device": "cpu",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--fail-performance", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples < 16 or args.repeats < 1:
        raise SystemExit("--samples must be >= 16 and --repeats must be >= 1.")
    metrics, correctness = _measure(args.samples, args.repeats)
    payload: dict[str, Any] = {
        "schema_version": "1",
        "environment": _environment(),
        "metrics": metrics,
        "correctness_passed": correctness,
        "blocking_status": "ok" if correctness else "failed",
        "performance_status": "not_compared",
    }
    if args.baseline is not None:
        baseline = psann.PerformanceBaseline.load(args.baseline)
        report = psann.compare_performance(
            baseline,
            metrics,
            correctness_passed=correctness,
        )
        payload["comparison"] = report.to_dict()
        payload["performance_status"] = report.performance_status
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not correctness:
        return 1
    if args.fail_performance and payload["performance_status"] == "warning":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
