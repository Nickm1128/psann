"""Serializable contracts and background policy for optional explainability."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

ExplanationAlgorithm = Literal["auto", "permutation", "partition", "gradient", "deep"]
MaskerKind = Literal["auto", "independent", "partition", "domain"]
GroupStrategy = Literal["auto", "feature", "time_step", "channel", "spatial_region"]
ExplanationOutputKind = Literal["auto", "prediction", "probability", "logit"]
FallbackPolicy = Literal["model_agnostic", "error"]


def _finite_json(value: Any, *, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must contain only finite JSON-serializable values; "
            f"received {value!r}."
        ) from exc


class ExplainabilityError(RuntimeError):
    """Base error for PSANN explanation creation and execution."""


class ExplainabilityUnavailableError(ImportError, ExplainabilityError):
    """Raised when the optional SHAP dependency is not installed."""


class BackgroundPolicyError(ExplainabilityError):
    """Raised when no approved explicit explanation background is supplied."""


class ExplanationCapabilityError(ExplainabilityError):
    """Raised when an explicitly requested explainer is not certified."""


@dataclass(frozen=True)
class ExplainerConfig:
    """Serializable, bounded explanation policy."""

    algorithm: ExplanationAlgorithm = "auto"
    masker: MaskerKind = "auto"
    group_strategy: GroupStrategy = "auto"
    output_kind: ExplanationOutputKind = "auto"
    output: int | str | None = None
    batch_size: int = 128
    max_evaluations: int = 2048
    max_explanation_samples: int = 32
    max_background_samples: int = 100
    background_size: int = 50
    gradient_samples: int = 200
    local_smoothing: float = 0.0
    seed: int = 0
    fallback: FallbackPolicy = "model_agnostic"
    layer: str | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in {"auto", "permutation", "partition", "gradient", "deep"}:
            raise ValueError(f"Unsupported explanation algorithm {self.algorithm!r}.")
        if self.masker not in {"auto", "independent", "partition", "domain"}:
            raise ValueError(f"Unsupported masker {self.masker!r}.")
        if self.group_strategy not in {
            "auto",
            "feature",
            "time_step",
            "channel",
            "spatial_region",
        }:
            raise ValueError(f"Unsupported feature-group strategy {self.group_strategy!r}.")
        if self.output_kind not in {"auto", "prediction", "probability", "logit"}:
            raise ValueError(f"Unsupported explanation output kind {self.output_kind!r}.")
        if self.fallback not in {"model_agnostic", "error"}:
            raise ValueError("fallback must be 'model_agnostic' or 'error'.")
        if self.output is not None and (
            isinstance(self.output, bool) or not isinstance(self.output, (int, str))
        ):
            raise TypeError("output must be a non-negative integer, non-empty string, or None.")
        for name in (
            "batch_size",
            "max_evaluations",
            "max_explanation_samples",
            "max_background_samples",
            "background_size",
            "gradient_samples",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1.")
        if self.background_size > self.max_background_samples:
            raise ValueError("background_size cannot exceed max_background_samples.")
        if not math.isfinite(float(self.local_smoothing)) or self.local_smoothing < 0:
            raise ValueError("local_smoothing must be finite and >= 0.")
        if isinstance(self.output, int) and self.output < 0:
            raise ValueError("output index must be >= 0.")
        if isinstance(self.output, str) and not self.output.strip():
            raise ValueError("output name cannot be empty.")
        if self.layer is not None and not self.layer.strip():
            raise ValueError("layer cannot be empty.")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "algorithm": self.algorithm,
            "masker": self.masker,
            "group_strategy": self.group_strategy,
            "output_kind": self.output_kind,
            "output": self.output,
            "batch_size": self.batch_size,
            "max_evaluations": self.max_evaluations,
            "max_explanation_samples": self.max_explanation_samples,
            "max_background_samples": self.max_background_samples,
            "background_size": self.background_size,
            "gradient_samples": self.gradient_samples,
            "local_smoothing": self.local_smoothing,
            "seed": self.seed,
            "fallback": self.fallback,
            "layer": self.layer,
        }
        _finite_json(result, field_name="explainer")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExplainerConfig":
        return cls(**dict(value))


@dataclass(frozen=True)
class BackgroundSummary:
    """Explicit, optionally persistence-approved background sample."""

    values: np.ndarray
    input_shape: tuple[int, ...]
    feature_names: tuple[str, ...] = ()
    data_format: str = "channels_first"
    approved_for_persistence: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.values, dtype=np.float32)
        if array.ndim != len(self.input_shape) + 1:
            raise ValueError(
                "BackgroundSummary values must contain a batch dimension followed by "
                f"input_shape={self.input_shape!r}."
            )
        if tuple(array.shape[1:]) != self.input_shape or int(array.shape[0]) < 1:
            raise ValueError("BackgroundSummary values do not match the declared input shape.")
        if not np.isfinite(array).all():
            raise ValueError("BackgroundSummary values must be finite.")
        _finite_json(dict(self.metadata), field_name="background.metadata")
        object.__setattr__(self, "values", array)

    def to_dict(self, *, include_values: bool = False) -> dict[str, Any]:
        result = {
            "input_shape": list(self.input_shape),
            "feature_names": list(self.feature_names),
            "data_format": self.data_format,
            "approved_for_persistence": self.approved_for_persistence,
            "metadata": dict(self.metadata),
        }
        if include_values:
            result["values"] = self.values.tolist()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackgroundSummary":
        if "values" not in value:
            raise BackgroundPolicyError("Persisted background summary does not contain values.")
        return cls(
            values=np.asarray(value["values"], dtype=np.float32),
            input_shape=tuple(int(item) for item in value["input_shape"]),
            feature_names=tuple(str(item) for item in value.get("feature_names", ())),
            data_format=str(value.get("data_format", "channels_first")),
            approved_for_persistence=bool(value.get("approved_for_persistence", False)),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class FeatureGroup:
    """One declared coalition of flattened raw-input positions."""

    name: str
    indices: tuple[int, ...]
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "indices": list(self.indices), "strategy": self.strategy}


@dataclass(frozen=True)
class ExplanationResult:
    """A standard ``shap.Explanation`` plus PSANN lifecycle metadata."""

    explanation: Any
    feature_groups: tuple[FeatureGroup, ...]
    task: str
    output_names: tuple[str, ...]
    artifact_version: str | None
    model_id: str | None
    run_id: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def values(self) -> Any:
        return self.explanation.values

    @property
    def base_values(self) -> Any:
        return self.explanation.base_values


@dataclass(frozen=True)
class ExplanationDriftSummary:
    """Aggregate attribution drift without retaining explained raw inputs."""

    feature_names: tuple[str, ...]
    reference_importance: tuple[float, ...]
    current_importance: tuple[float, ...]
    absolute_shift: tuple[float, ...]
    mean_absolute_shift: float
    maximum_absolute_shift: float
    cosine_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "reference_importance": list(self.reference_importance),
            "current_importance": list(self.current_importance),
            "absolute_shift": list(self.absolute_shift),
            "mean_absolute_shift": self.mean_absolute_shift,
            "maximum_absolute_shift": self.maximum_absolute_shift,
            "cosine_similarity": self.cosine_similarity,
        }


def summarize_background(
    reference_data: Any,
    *,
    input_shape: Sequence[int] | None = None,
    feature_names: Sequence[str] = (),
    data_format: str = "channels_first",
    max_samples: int = 50,
    seed: int = 0,
    approved_for_persistence: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> BackgroundSummary:
    """Deterministically sample an explicitly supplied reference dataset."""

    if max_samples < 1:
        raise ValueError("max_samples must be >= 1.")
    values = np.asarray(reference_data, dtype=np.float32)
    if values.ndim < 2 or int(values.shape[0]) < 1:
        raise BackgroundPolicyError("Reference data must contain at least one batched sample.")
    if not np.isfinite(values).all():
        raise BackgroundPolicyError("Reference data must contain only finite numeric values.")
    if int(values.shape[0]) > max_samples:
        generator = np.random.default_rng(seed)
        selected = np.sort(generator.choice(values.shape[0], size=max_samples, replace=False))
        values = values[selected]
    shape = tuple(int(item) for item in (input_shape or values.shape[1:]))
    return BackgroundSummary(
        values=values,
        input_shape=shape,
        feature_names=tuple(str(item) for item in feature_names),
        data_format=data_format,
        approved_for_persistence=approved_for_persistence,
        metadata=dict(metadata or {}),
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_explainer_config(
    config: ExplainerConfig,
    path: str | os.PathLike[str],
    *,
    background_summary: BackgroundSummary | None = None,
    include_background: bool = False,
) -> Path:
    """Persist explainer policy separately from a model artifact."""

    if include_background:
        if background_summary is None:
            raise BackgroundPolicyError(
                "include_background=True requires an explicit BackgroundSummary."
            )
        if not background_summary.approved_for_persistence:
            raise BackgroundPolicyError(
                "Background summary persistence requires approved_for_persistence=True."
            )
    payload: dict[str, Any] = {
        "format": "psann.explainer",
        "version": "1.0",
        "config": config.to_dict(),
    }
    if include_background and background_summary is not None:
        payload["background_summary"] = background_summary.to_dict(include_values=True)
    destination = Path(path)
    _atomic_json(destination, payload)
    return destination


def load_explainer_config(
    path: str | os.PathLike[str],
) -> tuple[ExplainerConfig, BackgroundSummary | None]:
    """Load a separately persisted explainer policy and optional approved summary."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("format") != "psann.explainer":
        raise ExplainabilityError("Explainer configuration has an invalid format marker.")
    if payload.get("version") != "1.0":
        raise ExplainabilityError(
            f"Unsupported explainer configuration version {payload.get('version')!r}."
        )
    config = ExplainerConfig.from_dict(payload.get("config", {}))
    value = payload.get("background_summary")
    summary = BackgroundSummary.from_dict(value) if isinstance(value, Mapping) else None
    return config, summary


__all__ = [
    "BackgroundPolicyError",
    "BackgroundSummary",
    "ExplainabilityError",
    "ExplainabilityUnavailableError",
    "ExplainerConfig",
    "ExplanationCapabilityError",
    "ExplanationDriftSummary",
    "ExplanationResult",
    "FeatureGroup",
    "GroupStrategy",
    "_atomic_json",
    "load_explainer_config",
    "save_explainer_config",
    "summarize_background",
]
