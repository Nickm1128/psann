"""Strictly typed boundaries shared by future workplace-platform components.

These contracts describe interchange between model specifications, task adapters,
artifact code, and inference code. They do not implement the Phase 3 lifecycle API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    Mapping,
    NotRequired,
    Protocol,
    Sequence,
    TypedDict,
    runtime_checkable,
)

TaskKind = Literal["regression", "binary", "multiclass", "multilabel"]


@runtime_checkable
class ModelSpecContract(Protocol):
    """Minimum serializable model-specification behavior."""

    task: Any
    backbone: str

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-safe representation using canonical field names."""


class ArtifactPackageManifest(TypedDict):
    name: str
    version: str


class ArtifactRuntimeManifest(TypedDict):
    python: str
    numpy: str
    torch: str


class ArtifactRequirementManifest(TypedDict):
    python_min: str
    psann_min: str
    torch_min: str


class ArtifactModelManifest(TypedDict):
    backbone: str
    task: TaskKind
    plugin: Mapping[str, str] | None


class ArtifactTrainingManifest(TypedDict):
    run_id: str | None


class ArtifactManifest(TypedDict):
    """Typed required fields for the native `.psann` manifest."""

    artifact_format: str
    artifact_format_version: str
    manifest_schema_version: str
    artifact_id: str
    created_at: str
    package: ArtifactPackageManifest
    runtime: ArtifactRuntimeManifest
    requirements: ArtifactRequirementManifest
    model: ArtifactModelManifest
    training: ArtifactTrainingManifest
    capabilities: Sequence[str]
    experimental: bool
    files: Mapping[str, str]
    required_extensions: Sequence[Mapping[str, str]]
    metadata: NotRequired[Mapping[str, Any]]
    registry: NotRequired[Mapping[str, Any]]


@runtime_checkable
class TaskAdapter(Protocol):
    """Task-owned validation and prediction-conversion boundary."""

    kind: TaskKind

    def validate_targets(self, targets: Any) -> None:
        """Reject targets that do not satisfy the task contract."""

    def predictions_from_outputs(self, outputs: Any) -> Any:
        """Convert raw model outputs into task-level predictions."""


@runtime_checkable
class BackboneProtocol(Protocol):
    """Minimum in-process model behavior required by the lifecycle API."""

    def fit(self, X: Any, y: Any, **fit_params: Any) -> Any:
        """Fit the model and return an estimator-compatible result."""

    def predict(self, X: Any, **predict_params: Any) -> Any:
        """Return task-level predictions."""

    def score(self, X: Any, y: Any, **score_params: Any) -> float:
        """Return the task's primary scalar score."""


@dataclass(frozen=True)
class TopKResult:
    """Ranked multiclass labels and probabilities for one inference result."""

    labels: Any
    probabilities: Any
    indices: Any


@dataclass(frozen=True)
class InferenceResult:
    """Typed metadata envelope for schema-aware inference results."""

    values: Any
    task: TaskKind
    output_names: tuple[str, ...] = ()
    artifact_version: str | None = None
    model_id: str | None = None
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    top_k: TopKResult | None = None
