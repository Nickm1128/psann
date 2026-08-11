"""Identifier registries for serializable workplace configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterable, Literal, Mapping, TypeVar

from .contracts import TaskKind

_ValueT = TypeVar("_ValueT")
EstimatorFactory = Callable[[Mapping[str, Any]], Any]
MetricFactory = Callable[[TaskKind], Callable[..., Any]]
TransformFactory = Callable[[], Any]
BackboneFactoryKind = Literal["estimator", "torch_module"]

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]*$")


def normalise_identifier(value: str, *, field_name: str = "identifier") -> str:
    identifier = str(value).strip().lower()
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"{field_name} must match {_IDENTIFIER.pattern!r}; received {value!r}.")
    return identifier


class IdentifierRegistry(Generic[_ValueT]):
    """Small explicit registry whose specifications store identifiers, not callables."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._values: dict[str, _ValueT] = {}

    def register(self, identifier: str, value: _ValueT, *, replace: bool = False) -> str:
        key = normalise_identifier(identifier, field_name=f"{self.name} identifier")
        if key in self._values and not replace:
            raise ValueError(f"{self.name} {key!r} is already registered.")
        self._values[key] = value
        return key

    def resolve(self, identifier: str) -> _ValueT:
        key = normalise_identifier(identifier, field_name=f"{self.name} identifier")
        try:
            return self._values[key]
        except KeyError as exc:
            supported = ", ".join(self.names()) or "<none>"
            raise ValueError(
                f"Unknown {self.name} {identifier!r}. Registered values: {supported}."
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))

    def __contains__(self, identifier: object) -> bool:
        return isinstance(identifier, str) and identifier.strip().lower() in self._values


@dataclass(frozen=True)
class ValueRegistration:
    """Metadata for a stable serializable configuration identifier."""

    identifier: str
    description: str
    plugin: str | None = None
    experimental: bool = False


@dataclass(frozen=True)
class BackboneRegistration:
    """Runtime factory and declared capability matrix for one backbone."""

    identifier: str
    factory: EstimatorFactory
    supported_tasks: frozenset[TaskKind]
    input_ranks: frozenset[int]
    activations: frozenset[str]
    normalizations: frozenset[str] = field(default_factory=lambda: frozenset({"none"}))
    supports_dropout: bool = False
    factory_kind: BackboneFactoryKind = "estimator"
    experimental: bool = False
    plugin: str | None = None
    plugin_version: str | None = None


BACKBONES: IdentifierRegistry[BackboneRegistration] = IdentifierRegistry("backbone")
ACTIVATIONS: IdentifierRegistry[ValueRegistration] = IdentifierRegistry("activation")
NORMALIZATIONS: IdentifierRegistry[ValueRegistration] = IdentifierRegistry("normalization")
DROPOUTS: IdentifierRegistry[ValueRegistration] = IdentifierRegistry("dropout strategy")
OPTIMIZERS: IdentifierRegistry[ValueRegistration] = IdentifierRegistry("optimizer")
SCHEDULERS: IdentifierRegistry[ValueRegistration] = IdentifierRegistry("scheduler")
LOSSES: IdentifierRegistry[ValueRegistration] = IdentifierRegistry("loss")
METRICS: IdentifierRegistry[MetricFactory] = IdentifierRegistry("metric")
CATEGORICAL_ENCODERS: IdentifierRegistry[TransformFactory] = IdentifierRegistry(
    "categorical encoder"
)
MISSING_VALUE_IMPUTERS: IdentifierRegistry[TransformFactory] = IdentifierRegistry(
    "missing-value imputer"
)


def _psann_estimator(parameters: Mapping[str, Any]) -> Any:
    from ..sklearn import PSANNRegressor

    return PSANNRegressor(**dict(parameters))


def _respsann_estimator(parameters: Mapping[str, Any]) -> Any:
    from ..sklearn import ResPSANNRegressor

    return ResPSANNRegressor(**dict(parameters))


def _conv_estimator(parameters: Mapping[str, Any]) -> Any:
    from ..sklearn import PSANNRegressor

    return PSANNRegressor.with_conv_stem(**dict(parameters))


def _resconv2d_estimator(parameters: Mapping[str, Any]) -> Any:
    from ..sklearn import ResConvPSANNRegressor

    return ResConvPSANNRegressor(**dict(parameters))


def _wave_estimator(parameters: Mapping[str, Any]) -> Any:
    from ..sklearn import WaveResNetRegressor

    return WaveResNetRegressor(**dict(parameters))


def _sgr_estimator(parameters: Mapping[str, Any]) -> Any:
    from ..sklearn import SGRPSANNRegressor

    return SGRPSANNRegressor(**dict(parameters))


def _metric_from_adapter(name: str) -> MetricFactory:
    def factory(kind: TaskKind) -> Callable[..., Any]:
        from .specs import TaskSpec
        from .tasks import create_task_adapter

        metrics = create_task_adapter(TaskSpec(kind=kind)).training_metrics()
        if name not in metrics:
            raise ValueError(f"Metric {name!r} is not defined for task {kind!r}.")
        return metrics[name]

    return factory


def register_backbone(
    identifier: str,
    factory: EstimatorFactory,
    *,
    supported_tasks: Iterable[TaskKind],
    input_ranks: Iterable[int],
    activations: Iterable[str],
    normalizations: Iterable[str] = ("none",),
    supports_dropout: bool = False,
    factory_kind: BackboneFactoryKind = "estimator",
    experimental: bool = True,
    plugin: str | None = None,
    plugin_version: str | None = None,
    replace: bool = False,
) -> BackboneRegistration:
    """Register a runtime backbone factory under a serialization-safe identifier."""

    if not callable(factory):
        raise TypeError("backbone factory must be callable.")
    key = normalise_identifier(identifier, field_name="backbone identifier")
    tasks = frozenset(supported_tasks)
    if not tasks or not tasks <= {"regression", "binary", "multiclass", "multilabel"}:
        raise ValueError("supported_tasks must contain valid TaskKind values.")
    if factory_kind not in {"estimator", "torch_module"}:
        raise ValueError("factory_kind must be 'estimator' or 'torch_module'.")
    if factory_kind == "torch_module" and tasks != {"regression"}:
        raise ValueError(
            "Registered torch_module factories currently support regression only; "
            "use adapt_module(...) for experimental in-process classification."
        )
    ranks = frozenset(int(rank) for rank in input_ranks)
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("input_ranks must contain positive non-batch ranks.")
    activation_ids = frozenset(
        normalise_identifier(item, field_name="activation identifier") for item in activations
    )
    normalization_ids = frozenset(
        normalise_identifier(item, field_name="normalization identifier") for item in normalizations
    )
    registration = BackboneRegistration(
        identifier=key,
        factory=factory,
        supported_tasks=tasks,
        input_ranks=ranks,
        activations=activation_ids,
        normalizations=normalization_ids,
        supports_dropout=bool(supports_dropout),
        factory_kind=factory_kind,
        experimental=bool(experimental),
        plugin=plugin,
        plugin_version=plugin_version,
    )
    BACKBONES.register(key, registration, replace=replace)
    return registration


def register_metric(
    identifier: str,
    factory: MetricFactory,
    *,
    replace: bool = False,
) -> str:
    """Register a task-aware metric factory without placing it in a ModelSpec."""

    if not callable(factory):
        raise TypeError("metric factory must be callable.")
    return METRICS.register(identifier, factory, replace=replace)


def register_schema_transform(
    kind: str,
    identifier: str,
    factory: TransformFactory,
    *,
    replace: bool = False,
) -> str:
    """Register an optional categorical encoder or missing-value imputer."""

    if not callable(factory):
        raise TypeError("schema transform factory must be callable.")
    if kind == "categorical_encoder":
        return CATEGORICAL_ENCODERS.register(identifier, factory, replace=replace)
    if kind == "missing_value_imputer":
        return MISSING_VALUE_IMPUTERS.register(identifier, factory, replace=replace)
    raise ValueError("kind must be 'categorical_encoder' or 'missing_value_imputer'.")


def _register_value_set(
    registry: IdentifierRegistry[ValueRegistration],
    values: Mapping[str, str],
) -> None:
    for identifier, description in values.items():
        registry.register(
            identifier,
            ValueRegistration(identifier=identifier, description=description),
        )


def _register_core() -> None:
    _register_value_set(
        ACTIVATIONS,
        {
            "relu": "Rectified linear unit",
            "tanh": "Hyperbolic tangent",
            "sigmoid": "Parameterized sigmoid",
            "gelu": "Gaussian error linear unit",
            "silu": "Sigmoid linear unit",
            "psann": "Parameterized sine activation",
            "relu_sigmoid_psann": "Hybrid PSANN activation",
        },
    )
    _register_value_set(
        NORMALIZATIONS,
        {
            "none": "No normalization",
            "layer": "Layer or channel-group normalization",
            "rms": "Root-mean-square normalization",
            "weight": "Weight normalization",
        },
    )
    _register_value_set(
        DROPOUTS,
        {
            "none": "No dropout",
            "standard": "Standard activation dropout",
        },
    )
    _register_value_set(
        OPTIMIZERS,
        {
            "adam": "Adam",
            "adamw": "AdamW",
            "sgd": "Momentum SGD",
        },
    )
    _register_value_set(
        SCHEDULERS,
        {
            "none": "No scheduler",
            "step": "Step decay",
            "cosine": "Cosine annealing",
        },
    )
    _register_value_set(
        LOSSES,
        {
            "mse": "Mean squared error",
            "l1": "Mean absolute error",
            "smooth_l1": "Smooth L1",
            "huber": "Huber",
            "binary_cross_entropy_with_logits": "Binary or multilabel logit loss",
            "cross_entropy": "Multiclass logit loss",
        },
    )
    all_tasks: tuple[TaskKind, ...] = (
        "regression",
        "binary",
        "multiclass",
        "multilabel",
    )
    standard_activations = ACTIVATIONS.names()
    register_backbone(
        "psann_mlp",
        _psann_estimator,
        supported_tasks=all_tasks,
        input_ranks=(1,),
        activations=standard_activations,
        experimental=False,
    )
    register_backbone(
        "respsann_mlp",
        _respsann_estimator,
        supported_tasks=all_tasks,
        input_ranks=(1,),
        activations=standard_activations,
        normalizations=("none", "layer", "rms"),
        experimental=False,
    )
    for identifier, rank in (
        ("psann_conv1d", 2),
        ("psann_conv2d", 3),
        ("psann_conv3d", 4),
    ):
        register_backbone(
            identifier,
            _conv_estimator,
            supported_tasks=all_tasks,
            input_ranks=(rank,),
            activations=standard_activations,
            experimental=False,
        )
    register_backbone(
        "respsann_conv2d",
        _resconv2d_estimator,
        supported_tasks=all_tasks,
        input_ranks=(3,),
        activations=standard_activations,
        normalizations=("none", "layer", "rms"),
        experimental=False,
    )
    register_backbone(
        "wave_resnet",
        _wave_estimator,
        supported_tasks=all_tasks,
        input_ranks=(1, 2, 3, 4),
        activations=("psann",),
        normalizations=("none", "rms", "weight"),
        supports_dropout=True,
        experimental=False,
    )
    register_backbone(
        "sgr_psann",
        _sgr_estimator,
        supported_tasks=all_tasks,
        input_ranks=(1, 2, 3),
        activations=("psann",),
        experimental=False,
    )
    for metric in ("mae", "mse", "accuracy", "subset_accuracy"):
        register_metric(metric, _metric_from_adapter(metric))


_register_core()


__all__ = [
    "ACTIVATIONS",
    "BACKBONES",
    "CATEGORICAL_ENCODERS",
    "DROPOUTS",
    "LOSSES",
    "METRICS",
    "MISSING_VALUE_IMPUTERS",
    "NORMALIZATIONS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "BackboneRegistration",
    "BackboneFactoryKind",
    "IdentifierRegistry",
    "ValueRegistration",
    "normalise_identifier",
    "register_backbone",
    "register_metric",
    "register_schema_transform",
]
