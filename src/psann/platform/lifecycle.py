"""Unified creation, training, and evaluation orchestration."""

from __future__ import annotations

import logging
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import torch

from .._sklearn.classifier import PSANNClassifier
from ..training_events import TrainingEventCallback
from .contracts import BackboneProtocol, TaskKind
from .module_adapter import TorchModuleAdapter
from .operations import (
    OperationalEvent,
    OperationalHooks,
    fingerprint_data,
    fingerprint_model,
    validate_no_secrets,
)
from .registry import (
    ACTIVATIONS,
    BACKBONES,
    CATEGORICAL_ENCODERS,
    LOSSES,
    METRICS,
    MISSING_VALUE_IMPUTERS,
    NORMALIZATIONS,
    OPTIMIZERS,
    SCHEDULERS,
)
from .specs import DataSchema, ModelSpec, TaskSpec, TrainingConfig
from .tasks import create_task_adapter


@dataclass(frozen=True)
class SupervisedData:
    """Explicit high-level container for supervised arrays and optional context."""

    inputs: Any
    targets: Any
    context: Any | None = None


def _coerce_model_spec(value: ModelSpec | Mapping[str, Any]) -> ModelSpec:
    return value if isinstance(value, ModelSpec) else ModelSpec.from_dict(value)


def _validate_schema_extensions(schema: DataSchema) -> None:
    if schema.categorical_encoder is not None:
        CATEGORICAL_ENCODERS.resolve(schema.categorical_encoder)
    if schema.missing_value_imputer is not None:
        MISSING_VALUE_IMPUTERS.resolve(schema.missing_value_imputer)


def _estimator_parameters(spec: ModelSpec) -> dict[str, Any]:
    parameters = dict(spec.parameters)
    forbidden = sorted(
        set(parameters)
        & {
            "activation_type",
            "data_format",
            "dropout",
            "hidden_width",
            "norm",
            "normalization",
            "preserve_shape",
        }
    )
    if forbidden:
        raise ValueError(
            "ModelSpec.parameters uses canonical fields owned by ModelSpec; remove: "
            f"{', '.join(forbidden)}."
        )
    parameters["activation_type"] = spec.activation
    parameters["data_format"] = spec.input_schema.data_format
    return parameters


def _validate_model_capabilities(spec: ModelSpec) -> Any:
    registration = BACKBONES.resolve(spec.backbone)
    ACTIVATIONS.resolve(spec.activation)
    NORMALIZATIONS.resolve(spec.normalization)
    if spec.task.kind not in registration.supported_tasks:
        raise ValueError(
            f"Backbone {registration.identifier!r} does not support task {spec.task.kind!r}."
        )
    if spec.activation not in registration.activations:
        raise ValueError(
            f"Backbone {registration.identifier!r} does not support activation "
            f"{spec.activation!r}; supported={sorted(registration.activations)!r}."
        )
    if spec.normalization not in registration.normalizations:
        raise ValueError(
            f"Backbone {registration.identifier!r} does not support normalization "
            f"{spec.normalization!r}; supported={sorted(registration.normalizations)!r}."
        )
    if spec.dropout and not registration.supports_dropout:
        raise ValueError(f"Backbone {registration.identifier!r} does not support standard dropout.")
    if spec.input_schema.input_shape:
        rank = len(spec.input_schema.input_shape)
        if rank not in registration.input_ranks:
            raise ValueError(
                f"Backbone {registration.identifier!r} expects non-batch input rank in "
                f"{sorted(registration.input_ranks)!r}; configured input_shape has rank "
                f"{rank}."
            )
    _validate_schema_extensions(spec.input_schema)
    return registration


def create_model(spec: ModelSpec | Mapping[str, Any]) -> BackboneProtocol:
    """Create a registered task-aware model from a serializable specification."""

    model_spec = _coerce_model_spec(spec)
    registration = _validate_model_capabilities(model_spec)
    parameters = _estimator_parameters(model_spec)
    if len(registration.normalizations) > 1:
        parameters["norm"] = model_spec.normalization
    if registration.supports_dropout:
        parameters["dropout"] = float(model_spec.dropout)
    if registration.identifier.startswith("psann_conv"):
        parameters["preserve_shape"] = True

    if model_spec.task.kind == "regression":
        model = registration.factory(parameters)
    else:
        model = PSANNClassifier(
            backbone=registration.identifier,
            task=model_spec.task.kind,
            threshold=model_spec.task.threshold,
            class_names=tuple(model_spec.task.class_names),
            positive_label=model_spec.task.positive_label,
            estimator_params=parameters,
            schema=model_spec.input_schema,
        )
    if isinstance(model, torch.nn.Module):
        if registration.factory_kind != "torch_module":
            raise TypeError(
                f"Backbone factory {registration.identifier!r} returned torch.nn.Module "
                "but was not registered with factory_kind='torch_module'."
            )
        model = TorchModuleAdapter(
            model,
            task=model_spec.task.kind,
            threshold=model_spec.task.threshold,
        )
        model.artifact_capabilities_ = (
            "in_process_training",
            "in_process_inference",
            "registered_native_artifact",
        )
    elif registration.factory_kind == "torch_module":
        raise TypeError(
            f"Backbone factory {registration.identifier!r} was registered as a "
            f"torch_module but returned {type(model).__name__}."
        )
    if not isinstance(model, BackboneProtocol):
        raise TypeError(
            f"Backbone factory {registration.identifier!r} returned "
            f"{type(model).__name__}, which does not implement fit/predict/score."
        )
    model._platform_model_spec_dict_ = model_spec.to_dict()  # type: ignore[attr-defined]
    model._platform_data_schema_ = model_spec.input_schema  # type: ignore[attr-defined]
    model._platform_task_spec_ = model_spec.task  # type: ignore[attr-defined]
    model.backbone_id_ = registration.identifier  # type: ignore[attr-defined]
    model.experimental_ = registration.experimental  # type: ignore[attr-defined]
    return model


def adapt_module(
    module: torch.nn.Module,
    *,
    task: TaskSpec | TaskKind | str = "regression",
    **training_parameters: Any,
) -> TorchModuleAdapter:
    """Wrap an arbitrary module for in-process training and inference only."""

    task_spec = task if isinstance(task, TaskSpec) else TaskSpec(kind=task)  # type: ignore[arg-type]
    return TorchModuleAdapter(
        module,
        task=task_spec.kind,
        threshold=task_spec.threshold,
        **training_parameters,
    )


def _supervised_data(value: SupervisedData | Sequence[Any] | Mapping[str, Any]) -> SupervisedData:
    if isinstance(value, SupervisedData):
        return value
    if isinstance(value, Mapping):
        inputs = value.get("inputs", value.get("X"))
        targets = value.get("targets", value.get("y"))
        if inputs is None or targets is None:
            raise ValueError("Training data mapping must contain inputs/X and targets/y.")
        return SupervisedData(inputs, targets, value.get("context"))
    values = tuple(value)
    if len(values) not in {2, 3}:
        raise ValueError("Training data must contain (inputs, targets[, context]).")
    return SupervisedData(
        inputs=values[0],
        targets=values[1],
        context=values[2] if len(values) == 3 else None,
    )


def _model_spec_from_instance(model: BackboneProtocol) -> ModelSpec:
    value = getattr(model, "_platform_model_spec_dict_", None)
    if value is not None:
        return ModelSpec.from_dict(value)
    task_value = getattr(model, "_platform_task_spec_", TaskSpec())
    task = (
        task_value
        if isinstance(task_value, TaskSpec)
        else (
            TaskSpec.from_dict(task_value)
            if isinstance(task_value, Mapping)
            else TaskSpec(kind=cast(TaskKind, str(task_value)))
        )
    )
    return ModelSpec(
        task=task,
        backbone=str(getattr(model, "backbone_id_", "arbitrary_module")),
    )


def _apply_training_config(
    model: BackboneProtocol,
    config: TrainingConfig,
    task: TaskSpec,
) -> None:
    requested_device = str(config.device).split(":", 1)[0].strip().lower()
    if requested_device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(
            f"Training device {requested_device!r} is outside the workplace support matrix."
        )
    OPTIMIZERS.resolve(config.optimizer)
    SCHEDULERS.resolve(config.scheduler)
    if config.loss is not None:
        LOSSES.resolve(config.loss)
        if task.kind != "regression":
            raise ValueError(
                "Classification losses are task-owned; leave TrainingConfig.loss unset."
            )
    parameter_values = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.learning_rate,
        "optimizer": config.optimizer,
        "weight_decay": config.weight_decay,
        "early_stopping": config.early_stopping,
        "patience": config.patience,
        "device": config.device,
        "amp": config.amp,
        "amp_dtype": config.amp_dtype,
        "compile": config.compile,
        "loss_reduction": config.loss_reduction,
    }
    if config.loss is not None:
        parameter_values["loss"] = config.loss
        parameter_values["loss_params"] = dict(config.loss_params)
    if isinstance(model, PSANNClassifier):
        merged = dict(model.estimator_params or {})
        merged.update(parameter_values)
        model.set_params(estimator_params=merged)
    elif isinstance(model, TorchModuleAdapter):
        unsupported_runtime_features = [
            name for name, enabled in (("amp", config.amp), ("compile", config.compile)) if enabled
        ]
        if unsupported_runtime_features:
            features = ", ".join(unsupported_runtime_features)
            message = (
                f"TorchModuleAdapter does not support {features}; "
                "the requested features would be disabled."
            )
            if config.fallback_policy == "error":
                raise RuntimeError(
                    f"{message} Set fallback_policy='warn' to permit this explicit fallback."
                )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        model.set_params(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer=config.optimizer,
            weight_decay=config.weight_decay,
            device=config.device,
        )
    else:
        available = set(model.get_params(deep=False))  # type: ignore[attr-defined]
        constructor_values = {
            name: value for name, value in parameter_values.items() if name in available
        }
        model.set_params(**constructor_values)  # type: ignore[attr-defined]
        for name, value in parameter_values.items():
            if name not in available and hasattr(model, name):
                setattr(model, name, value)


def _training_metrics(config: TrainingConfig, task: TaskSpec) -> dict[str, Any]:
    adapter = create_task_adapter(task)
    if not config.metrics:
        return dict(adapter.training_metrics())
    metrics: dict[str, Any] = {}
    for identifier in config.metrics:
        metrics[identifier] = METRICS.resolve(identifier)(task.kind)
    return metrics


@dataclass
class TrainingRun:
    """Completed high-level training result and evaluation boundary."""

    model: BackboneProtocol
    model_spec: ModelSpec
    training_config: TrainingConfig
    run_id: str
    history: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    operational_hooks: OperationalHooks | None = field(default=None, repr=False)

    def operational_event(
        self,
        kind: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        model_id: str | None = None,
    ) -> OperationalEvent:
        """Create a redacted event tied to this run."""

        return OperationalEvent.create(
            kind,
            run_id=self.run_id,
            model_id=model_id,
            metadata=metadata,
        )

    def evaluate(
        self,
        data: SupervisedData | Sequence[Any] | Mapping[str, Any],
    ) -> Mapping[str, float]:
        dataset = _supervised_data(data)
        task_adapter = getattr(self.model, "task_adapter_", None)
        if task_adapter is None:
            task_adapter = create_task_adapter(self.model_spec.task)
            task_adapter.fit_targets(dataset.targets)
        if self.model_spec.task.kind == "regression":
            output = self.model.predict(
                dataset.inputs,
                **({"context": dataset.context} if dataset.context is not None else {}),
            )
        else:
            decision = getattr(self.model, "decision_function")
            output = decision(
                dataset.inputs,
                **({"context": dataset.context} if dataset.context is not None else {}),
            )
        return task_adapter.evaluate(dataset.targets, output)

    def export(
        self,
        path: str | Path,
        *,
        model_card: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        registry: Mapping[str, Any] | None = None,
    ) -> Path:
        """Export the completed run as an atomic, safe native `.psann` artifact."""

        from .artifacts import export_model

        reviewed_metadata = dict(metadata or {})
        fingerprints = dict(reviewed_metadata.get("fingerprints", {}))
        fingerprints.setdefault(
            "model",
            self.metadata.get("model_fingerprint", fingerprint_model(self.model)),
        )
        if self.metadata.get("data_fingerprint") is not None:
            fingerprints.setdefault("data", self.metadata["data_fingerprint"])
        reviewed_metadata["fingerprints"] = fingerprints
        validate_no_secrets(reviewed_metadata, field="metadata")
        artifact = export_model(
            self.model,
            path,
            model_spec=self.model_spec,
            run_id=self.run_id,
            model_card=model_card,
            metadata=reviewed_metadata,
            registry=registry,
        )
        if self.operational_hooks is not None:
            from .artifacts import inspect_artifact

            info = inspect_artifact(artifact)
            self.operational_hooks.emit(
                self.operational_event(
                    "artifact_exported",
                    model_id=info.artifact_id,
                    metadata={
                        "artifact_format_version": info.artifact_format_version,
                        "backbone": info.backbone,
                        "task": info.task,
                        "fingerprints": fingerprints,
                    },
                ),
                registry=True,
            )
        return artifact


def train(
    model: BackboneProtocol,
    train_data: SupervisedData | Sequence[Any] | Mapping[str, Any],
    *,
    validation_data: SupervisedData | Sequence[Any] | Mapping[str, Any] | None = None,
    config: TrainingConfig | Mapping[str, Any] | None = None,
    callbacks: Sequence[TrainingEventCallback] | None = None,
    logger: logging.Logger | None = None,
    operational_hooks: OperationalHooks | None = None,
) -> TrainingRun:
    """Train a created model through the shared estimator training implementation."""

    training_config = (
        config
        if isinstance(config, TrainingConfig)
        else TrainingConfig.from_dict(config) if config is not None else TrainingConfig()
    )
    model_spec = _model_spec_from_instance(model)
    run_id = str(uuid.uuid4())
    if operational_hooks is not None:
        operational_hooks.emit(
            OperationalEvent.create(
                "training_started",
                run_id=run_id,
                metadata={
                    "backbone": model_spec.backbone,
                    "task": model_spec.task.kind,
                    "device": training_config.device,
                    "amp": training_config.amp,
                    "amp_dtype": training_config.amp_dtype,
                    "compile": training_config.compile,
                },
            )
        )
    _apply_training_config(model, training_config, model_spec.task)
    dataset = _supervised_data(train_data)
    validation = _supervised_data(validation_data) if validation_data is not None else None

    fit_parameters: dict[str, Any] = {}
    if dataset.context is not None:
        fit_parameters["context"] = dataset.context
    if validation is not None:
        fit_parameters["validation_data"] = (
            (validation.inputs, validation.targets)
            if validation.context is None
            else (validation.inputs, validation.targets, validation.context)
        )
    if not isinstance(model, TorchModuleAdapter):
        fit_parameters.update(
            {
                "scheduler": training_config.scheduler,
                "scheduler_params": dict(training_config.scheduler_params),
                "nonfinite_policy": training_config.nonfinite_policy,
                "fallback_policy": training_config.fallback_policy,
                "callback_error_policy": training_config.callback_error_policy,
                "deterministic": training_config.deterministic,
                "metrics": _training_metrics(training_config, model_spec.task),
                "callbacks": callbacks,
                "logger": logger,
                "resume_from": training_config.resume_from,
                "checkpoint_dir": training_config.checkpoint_dir,
                "checkpoint_every": training_config.checkpoint_every,
                "checkpoint_keep": training_config.checkpoint_keep,
            }
        )
    try:
        model.fit(dataset.inputs, dataset.targets, **fit_parameters)
    except Exception as exc:
        if operational_hooks is not None:
            operational_hooks.emit(
                OperationalEvent.create(
                    "training_failed",
                    run_id=run_id,
                    metadata={"error_type": type(exc).__name__},
                )
            )
        raise
    actual_task = getattr(model, "task_spec_", model_spec.task)
    if isinstance(actual_task, TaskSpec) and actual_task != model_spec.task:
        model_spec = ModelSpec(
            task=actual_task,
            backbone=model_spec.backbone,
            input_schema=model_spec.input_schema,
            activation=model_spec.activation,
            normalization=model_spec.normalization,
            dropout=model_spec.dropout,
            parameters=model_spec.parameters,
        )
        model._platform_model_spec_dict_ = model_spec.to_dict()  # type: ignore[attr-defined]
    run = TrainingRun(
        model=model,
        model_spec=model_spec,
        training_config=training_config,
        run_id=run_id,
        history=tuple(dict(entry) for entry in getattr(model, "history_", ())),
        metrics={},
        metadata={
            "backbone": model_spec.backbone,
            "task": model_spec.task.kind,
            "experimental": bool(getattr(model, "experimental_", False)),
            "training": dict(getattr(model, "training_metadata_", {})),
            "preprocessing": dict(getattr(model, "preprocessing_contract_", {})),
            "data_fingerprint": fingerprint_data(
                dataset.inputs,
                dataset.targets,
                dataset.context,
            ),
            "model_fingerprint": fingerprint_model(model),
        },
        operational_hooks=operational_hooks,
    )
    run.metrics = dict(run.evaluate(dataset))
    if operational_hooks is not None:
        operational_hooks.emit(
            run.operational_event(
                "training_completed",
                metadata={
                    "backbone": model_spec.backbone,
                    "task": model_spec.task.kind,
                    "metrics": run.metrics,
                    "data_fingerprint": run.metadata["data_fingerprint"],
                    "model_fingerprint": run.metadata["model_fingerprint"],
                },
            )
        )
    return run


__all__ = [
    "SupervisedData",
    "TrainingRun",
    "adapt_module",
    "create_model",
    "train",
]
