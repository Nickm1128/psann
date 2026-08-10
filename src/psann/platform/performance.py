"""Portable performance observations and non-blocking regression comparisons."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

MetricDirection = Literal["higher", "lower"]

DEFAULT_DIRECTIONS: dict[str, MetricDirection] = {
    "training_samples_per_second": "higher",
    "inference_p50_ms": "lower",
    "inference_p95_ms": "lower",
    "peak_python_memory_bytes": "lower",
    "artifact_load_p50_ms": "lower",
    "explanation_ms": "lower",
}


@dataclass(frozen=True)
class PerformanceBaseline:
    """Versioned reference metrics with per-metric relative alert tolerances."""

    name: str
    metrics: Mapping[str, float]
    relative_tolerances: Mapping[str, float] = field(default_factory=dict)
    environment: Mapping[str, Any] = field(default_factory=dict)
    correctness_required: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Performance baseline name cannot be empty.")
        for name, value in self.metrics.items():
            if name not in DEFAULT_DIRECTIONS:
                raise ValueError(f"Unknown performance metric {name!r}.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Performance metric {name!r} must be finite and >= 0.")
        for name, value in self.relative_tolerances.items():
            if name not in self.metrics:
                raise ValueError(f"Tolerance provided for absent metric {name!r}.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"Tolerance for {name!r} must be finite and >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "name": self.name,
            "metrics": {name: float(value) for name, value in self.metrics.items()},
            "relative_tolerances": {
                name: float(value) for name, value in self.relative_tolerances.items()
            },
            "environment": dict(self.environment),
            "correctness_required": self.correctness_required,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerformanceBaseline":
        if str(value.get("schema_version", "1")) != "1":
            raise ValueError("Unsupported performance baseline schema version.")
        return cls(
            name=str(value["name"]),
            metrics={str(name): float(metric) for name, metric in value["metrics"].items()},
            relative_tolerances={
                str(name): float(tolerance)
                for name, tolerance in value.get("relative_tolerances", {}).items()
            },
            environment=dict(value.get("environment", {})),
            correctness_required=bool(value.get("correctness_required", True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PerformanceBaseline":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Performance baseline must contain a JSON object.")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class PerformanceRegression:
    """One alert-level metric regression."""

    metric: str
    baseline: float
    observed: float
    tolerance: float
    direction: MetricDirection

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "baseline": self.baseline,
            "observed": self.observed,
            "tolerance": self.tolerance,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class PerformanceReport:
    """Correctness result plus noisy, non-blocking performance alerts."""

    baseline: str
    correctness_passed: bool
    observed: Mapping[str, float]
    regressions: tuple[PerformanceRegression, ...]
    missing_metrics: tuple[str, ...] = ()

    @property
    def performance_status(self) -> str:
        return "warning" if self.regressions or self.missing_metrics else "ok"

    @property
    def blocking_status(self) -> str:
        return "ok" if self.correctness_passed else "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "baseline": self.baseline,
            "correctness_passed": self.correctness_passed,
            "blocking_status": self.blocking_status,
            "performance_status": self.performance_status,
            "observed": {name: float(value) for name, value in self.observed.items()},
            "regressions": [item.to_dict() for item in self.regressions],
            "missing_metrics": list(self.missing_metrics),
        }


def compare_performance(
    baseline: PerformanceBaseline,
    observed: Mapping[str, float],
    *,
    correctness_passed: bool,
    default_relative_tolerance: float = 0.25,
) -> PerformanceReport:
    """Compare metrics while keeping correctness as the only default blocker."""

    if default_relative_tolerance < 0 or not math.isfinite(default_relative_tolerance):
        raise ValueError("default_relative_tolerance must be finite and >= 0.")
    regressions: list[PerformanceRegression] = []
    missing: list[str] = []
    reviewed = {str(name): float(value) for name, value in observed.items()}
    for name, reference in baseline.metrics.items():
        if name not in reviewed:
            missing.append(name)
            continue
        value = reviewed[name]
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Observed performance metric {name!r} must be finite and >= 0.")
        tolerance = float(baseline.relative_tolerances.get(name, default_relative_tolerance))
        direction = DEFAULT_DIRECTIONS[name]
        threshold = (
            float(reference) * (1.0 - tolerance)
            if direction == "higher"
            else float(reference) * (1.0 + tolerance)
        )
        regressed = value < threshold if direction == "higher" else value > threshold
        if regressed:
            regressions.append(
                PerformanceRegression(
                    metric=name,
                    baseline=float(reference),
                    observed=value,
                    tolerance=tolerance,
                    direction=direction,
                )
            )
    return PerformanceReport(
        baseline=baseline.name,
        correctness_passed=bool(correctness_passed),
        observed=reviewed,
        regressions=tuple(regressions),
        missing_metrics=tuple(missing),
    )


__all__ = [
    "DEFAULT_DIRECTIONS",
    "PerformanceBaseline",
    "PerformanceRegression",
    "PerformanceReport",
    "compare_performance",
]
