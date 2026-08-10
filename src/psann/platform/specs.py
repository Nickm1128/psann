"""Serializable configuration objects for the workplace lifecycle API."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, Mapping, TypeVar

from .contracts import TaskKind

FeaturePolicy = Literal["strict", "reorder", "positional"]
DataFormat = Literal["channels_first", "channels_last"]
DeviceTransferPolicy = Literal["per_batch", "full_batch"]
ClassificationOutput = Literal["probability", "label"]

_SpecT = TypeVar("_SpecT")


def _json_safe(value: Any, *, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{field_name} must contain only finite JSON-serializable values; "
            f"received {value!r}."
        ) from exc


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class TaskSpec:
    """Task-owned output, label, and threshold policy."""

    kind: TaskKind = "regression"
    class_names: tuple[Any, ...] = ()
    threshold: float | tuple[float, ...] = 0.5
    positive_label: Any | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"regression", "binary", "multiclass", "multilabel"}:
            raise ValueError(
                "task.kind must be regression, binary, multiclass, or multilabel; "
                f"received {self.kind!r}."
            )
        _json_safe(list(self.class_names), field_name="task.class_names")
        _json_safe(self.positive_label, field_name="task.positive_label")
        values = self.threshold if isinstance(self.threshold, tuple) else (self.threshold,)
        if not values:
            raise ValueError("task.threshold cannot be empty.")
        for value in values:
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(
                    "task.threshold values must be finite probabilities in [0, 1]; "
                    f"received {value!r}."
                )
        expected = {
            "regression": None,
            "binary": 2,
            "multiclass": None,
            "multilabel": None,
        }[self.kind]
        if expected is not None and self.class_names and len(self.class_names) != expected:
            raise ValueError(
                f"task.class_names for {self.kind} must contain {expected} values; "
                f"received {len(self.class_names)}."
            )
        if self.kind == "multilabel" and isinstance(self.threshold, tuple):
            if self.class_names and len(self.threshold) != len(self.class_names):
                raise ValueError("A multilabel threshold tuple must match task.class_names length.")
        if self.kind != "multilabel" and isinstance(self.threshold, tuple):
            raise ValueError("Only multilabel tasks accept one threshold per output.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "class_names": list(self.class_names),
            "threshold": (
                list(self.threshold) if isinstance(self.threshold, tuple) else self.threshold
            ),
            "positive_label": self.positive_label,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        threshold = value.get("threshold", 0.5)
        if isinstance(threshold, list):
            threshold = tuple(float(item) for item in threshold)
        return cls(
            kind=str(value.get("kind", "regression")),  # type: ignore[arg-type]
            class_names=tuple(value.get("class_names", ())),
            threshold=threshold,
            positive_label=value.get("positive_label"),
        )


@dataclass(frozen=True)
class DataSchema:
    """Input, output, dtype, and preprocessing contract."""

    feature_names: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    input_shape: tuple[int, ...] = ()
    data_format: DataFormat = "channels_first"
    dtype: str = "float32"
    feature_policy: FeaturePolicy = "strict"
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    target_scaling: Mapping[str, Any] = field(default_factory=dict)
    categorical_encoder: str | None = None
    missing_value_imputer: str | None = None

    def __post_init__(self) -> None:
        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError("data_schema.data_format must be channels_first or channels_last.")
        if self.feature_policy not in {"strict", "reorder", "positional"}:
            raise ValueError("data_schema.feature_policy must be strict, reorder, or positional.")
        if any(int(dim) <= 0 for dim in self.input_shape):
            raise ValueError("data_schema.input_shape must contain only positive dimensions.")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("data_schema.feature_names cannot contain duplicates.")
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError("data_schema.output_names cannot contain duplicates.")
        if not self.dtype:
            raise ValueError("data_schema.dtype cannot be empty.")
        _json_safe(_mapping(self.preprocessing), field_name="data_schema.preprocessing")
        _json_safe(_mapping(self.target_scaling), field_name="data_schema.target_scaling")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "output_names": list(self.output_names),
            "input_shape": list(self.input_shape),
            "data_format": self.data_format,
            "dtype": self.dtype,
            "feature_policy": self.feature_policy,
            "preprocessing": _mapping(self.preprocessing),
            "target_scaling": _mapping(self.target_scaling),
            "categorical_encoder": self.categorical_encoder,
            "missing_value_imputer": self.missing_value_imputer,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataSchema":
        return cls(
            feature_names=tuple(str(item) for item in value.get("feature_names", ())),
            output_names=tuple(str(item) for item in value.get("output_names", ())),
            input_shape=tuple(int(item) for item in value.get("input_shape", ())),
            data_format=str(value.get("data_format", "channels_first")),  # type: ignore[arg-type]
            dtype=str(value.get("dtype", "float32")),
            feature_policy=str(value.get("feature_policy", "strict")),  # type: ignore[arg-type]
            preprocessing=_mapping(value.get("preprocessing")),  # type: ignore[arg-type]
            target_scaling=_mapping(value.get("target_scaling")),  # type: ignore[arg-type]
            categorical_encoder=(
                str(value["categorical_encoder"])
                if value.get("categorical_encoder") is not None
                else None
            ),
            missing_value_imputer=(
                str(value["missing_value_imputer"])
                if value.get("missing_value_imputer") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ModelSpec:
    """Serializable model selection and architecture configuration."""

    task: TaskSpec = field(default_factory=TaskSpec)
    backbone: str = "psann_mlp"
    input_schema: DataSchema = field(default_factory=DataSchema)
    activation: str = "psann"
    normalization: str = "none"
    dropout: float = 0.0
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backbone.strip():
            raise ValueError("model.backbone cannot be empty.")
        if not self.activation.strip():
            raise ValueError("model.activation cannot be empty.")
        if not self.normalization.strip():
            raise ValueError("model.normalization cannot be empty.")
        if not math.isfinite(float(self.dropout)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("model.dropout must be finite and in [0, 1).")
        _json_safe(_mapping(self.parameters), field_name="model.parameters")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "backbone": self.backbone,
            "input_schema": self.input_schema.to_dict(),
            "activation": self.activation,
            "normalization": self.normalization,
            "dropout": float(self.dropout),
            "parameters": _mapping(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelSpec":
        task_value = value.get("task", {})
        task = (
            TaskSpec(kind=task_value)  # type: ignore[arg-type]
            if isinstance(task_value, str)
            else TaskSpec.from_dict(task_value)  # type: ignore[arg-type]
        )
        schema_value = value.get("input_schema", {})
        schema = (
            schema_value
            if isinstance(schema_value, DataSchema)
            else DataSchema.from_dict(schema_value)  # type: ignore[arg-type]
        )
        return cls(
            task=task,
            backbone=str(value.get("backbone", "psann_mlp")),
            input_schema=schema,
            activation=str(value.get("activation", "psann")),
            normalization=str(value.get("normalization", "none")),
            dropout=float(value.get("dropout", 0.0)),
            parameters=_mapping(value.get("parameters")),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TrainingConfig:
    """Serializable optimizer-driven training configuration."""

    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    scheduler: str = "none"
    scheduler_params: Mapping[str, Any] = field(default_factory=dict)
    loss: str | None = None
    loss_params: Mapping[str, Any] = field(default_factory=dict)
    loss_reduction: str = "mean"
    metrics: tuple[str, ...] = ()
    early_stopping: bool = False
    patience: int = 20
    device: str = "auto"
    deterministic: bool = False
    amp: bool = False
    amp_dtype: str = "bfloat16"
    compile: bool = False
    nonfinite_policy: str = "error"
    fallback_policy: str = "warn"
    callback_error_policy: str = "raise"
    resume_from: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0
    checkpoint_keep: int = 3

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("training.epochs must be >= 1.")
        if self.batch_size < 1:
            raise ValueError("training.batch_size must be >= 1.")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("training.learning_rate must be finite and > 0.")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("training.weight_decay must be finite and >= 0.")
        if self.patience < 1:
            raise ValueError("training.patience must be >= 1.")
        if self.checkpoint_every < 0 or self.checkpoint_keep < 1:
            raise ValueError(
                "training.checkpoint_every must be >= 0 and checkpoint_keep must be >= 1."
            )
        from .accelerators import canonical_dtype

        amp_dtype = canonical_dtype(self.amp_dtype)
        if self.amp and amp_dtype == "float32":
            raise ValueError("training.amp_dtype must be float16 or bfloat16 when amp=True.")
        if self.amp and self.compile:
            raise ValueError(
                "training.amp and training.compile cannot both be enabled; "
                "that combined path is not certified."
            )
        object.__setattr__(self, "amp_dtype", amp_dtype)
        for name, value in (
            ("optimizer", self.optimizer),
            ("scheduler", self.scheduler),
            ("loss_reduction", self.loss_reduction),
            ("nonfinite_policy", self.nonfinite_policy),
            ("fallback_policy", self.fallback_policy),
            ("callback_error_policy", self.callback_error_policy),
        ):
            if not value.strip():
                raise ValueError(f"training.{name} cannot be empty.")
        _json_safe(_mapping(self.scheduler_params), field_name="training.scheduler_params")
        _json_safe(_mapping(self.loss_params), field_name="training.loss_params")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "weight_decay": self.weight_decay,
            "scheduler": self.scheduler,
            "scheduler_params": _mapping(self.scheduler_params),
            "loss": self.loss,
            "loss_params": _mapping(self.loss_params),
            "loss_reduction": self.loss_reduction,
            "metrics": list(self.metrics),
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "device": self.device,
            "deterministic": self.deterministic,
            "amp": self.amp,
            "amp_dtype": self.amp_dtype,
            "compile": self.compile,
            "nonfinite_policy": self.nonfinite_policy,
            "fallback_policy": self.fallback_policy,
            "callback_error_policy": self.callback_error_policy,
            "resume_from": self.resume_from,
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_every": self.checkpoint_every,
            "checkpoint_keep": self.checkpoint_keep,
        }
        _json_safe(result, field_name="training")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        fields = dict(value)
        fields["metrics"] = tuple(str(item) for item in value.get("metrics", ()))
        return cls(**fields)


@dataclass(frozen=True)
class InferenceConfig:
    """Serializable inference preferences consumed by the deployment layer."""

    batch_size: int = 1024
    device: str = "auto"
    dtype: str = "float32"
    feature_policy: FeaturePolicy = "strict"
    return_logits: bool = False
    classification_output: ClassificationOutput = "probability"
    top_k: int | None = None
    device_transfer: DeviceTransferPolicy = "per_batch"
    fallback_policy: str = "error"

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, Integral):
            raise TypeError("inference.batch_size must be a positive integer.")
        if int(self.batch_size) < 1:
            raise ValueError("inference.batch_size must be >= 1.")
        object.__setattr__(self, "batch_size", int(self.batch_size))
        if self.feature_policy not in {"strict", "reorder", "positional"}:
            raise ValueError("inference.feature_policy must be strict, reorder, or positional.")
        if self.classification_output not in {"probability", "label"}:
            raise ValueError("inference.classification_output must be probability or label.")
        if self.top_k is not None:
            if (
                isinstance(self.top_k, bool)
                or not isinstance(self.top_k, Integral)
                or int(self.top_k) < 1
            ):
                raise ValueError("inference.top_k must be a positive integer or None.")
            if self.return_logits or self.classification_output != "probability":
                raise ValueError(
                    "inference.top_k requires probability output and return_logits=False."
                )
            object.__setattr__(self, "top_k", int(self.top_k))
        if self.device_transfer not in {"per_batch", "full_batch"}:
            raise ValueError("inference.device_transfer must be per_batch or full_batch.")
        from .accelerators import canonical_dtype

        dtype = canonical_dtype(self.dtype)
        if dtype != "float32":
            raise ValueError("Stable inference supports dtype='float32' only.")
        object.__setattr__(self, "dtype", dtype)
        if self.fallback_policy not in {"warn", "error"}:
            raise ValueError("inference.fallback_policy must be 'warn' or 'error'.")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "batch_size": self.batch_size,
            "device": self.device,
            "dtype": self.dtype,
            "feature_policy": self.feature_policy,
            "return_logits": self.return_logits,
            "classification_output": self.classification_output,
            "top_k": self.top_k,
            "device_transfer": self.device_transfer,
            "fallback_policy": self.fallback_policy,
        }
        _json_safe(result, field_name="inference")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InferenceConfig":
        return cls(**dict(value))


def save_spec(spec: ModelSpec | TrainingConfig | InferenceConfig, path: str | Path) -> None:
    """Write a human-readable JSON specification."""

    Path(path).write_text(
        json.dumps(spec.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_model_spec(path: str | Path) -> ModelSpec:
    """Load and validate a model specification from JSON."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A model specification file must contain a JSON object.")
    return ModelSpec.from_dict(value)


__all__ = [
    "ClassificationOutput",
    "DataFormat",
    "DataSchema",
    "DeviceTransferPolicy",
    "FeaturePolicy",
    "InferenceConfig",
    "ModelSpec",
    "TaskSpec",
    "TrainingConfig",
    "load_model_spec",
    "save_spec",
]
