"""Safe native model artifacts and explicitly trusted legacy migration."""

from __future__ import annotations

import os
import platform as runtime_platform
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from packaging.version import InvalidVersion, Version

from .._sklearn.classifier import PSANNClassifier
from .._version import __version__
from ..legacy import LEGACY_CHECKPOINT_MESSAGE, LegacyCheckpointWarning
from ..training_checkpoint import is_training_checkpoint
from ..utils import choose_device
from .artifact_io import (
    ValidatedArtifact,
    all_payloads,
    inspect_bundle,
    json_bytes,
    parse_json,
    write_bundle,
)
from .artifact_models import (
    actual_model_spec as _actual_model_spec,
)
from .artifact_models import (
    deserialize_weights as _deserialize_weights,
)
from .artifact_models import (
    fitted_core as _fitted_core,
)
from .artifact_models import (
    fitted_metadata as _fitted_metadata,
)
from .artifact_models import (
    json_value as _json_value,
)
from .artifact_models import (
    plugin_version as _plugin_version,
)
from .artifact_models import (
    preprocessing_metadata as _preprocessing_metadata,
)
from .artifact_models import (
    required_extensions as _required_extensions,
)
from .artifact_models import (
    restore_classifier as _restore_classifier,
)
from .artifact_models import (
    restore_core as _restore_core,
)
from .artifact_models import (
    restore_module_adapter as _restore_module_adapter,
)
from .artifact_models import (
    serialize_weights as _serialize_weights,
)
from .artifact_schema import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    FITTED_CONFIG_PATH,
    INPUT_SCHEMA_PATH,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    MODEL_CARD_PATH,
    MODEL_CONFIG_PATH,
    OUTPUT_SCHEMA_PATH,
    PREPROCESSING_PATH,
    WEIGHTS_PATH,
    ArtifactError,
    ArtifactExtensionError,
    ArtifactFormatError,
    ArtifactVersionError,
    LegacyCheckpointTrustError,
    validate_manifest,
)
from .lifecycle import create_model
from .module_adapter import TorchModuleAdapter
from .operations import fingerprint_model, validate_no_secrets
from .registry import (
    BACKBONES,
    CATEGORICAL_ENCODERS,
    MISSING_VALUE_IMPUTERS,
)
from .specs import DataSchema, ModelSpec, TaskSpec


@dataclass(frozen=True)
class ArtifactInfo:
    """Inspectable artifact metadata returned without deserializing model weights."""

    path: Path
    artifact_id: str
    artifact_format_version: str
    original_format_version: str
    package_version: str
    backbone: str
    task: str
    run_id: str | None
    capabilities: tuple[str, ...]
    experimental: bool
    migrations: tuple[str, ...]
    manifest: Mapping[str, Any]


def export_model(
    model: Any,
    path: str | os.PathLike[str],
    *,
    model_spec: ModelSpec | Mapping[str, Any] | None = None,
    run_id: str | None = None,
    model_card: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    registry: Mapping[str, Any] | None = None,
) -> Path:
    """Write a fitted registered model as an atomic, restricted-load `.psann` bundle."""

    core = _fitted_core(model)
    validate_no_secrets(dict(metadata or {}), field="metadata")
    validate_no_secrets(dict(registry or {}), field="registry")
    if model_card is not None:
        validate_no_secrets(str(model_card), field="model_card")
    spec = _actual_model_spec(model, model_spec)
    try:
        registration = BACKBONES.resolve(spec.backbone)
    except ValueError as exc:
        raise ArtifactExtensionError(
            f"Backbone {spec.backbone!r} is not registered for artifact export."
        ) from exc
    if getattr(core, "lsm", None) is not None:
        raise ArtifactExtensionError(
            "LSM preprocessors require an approved artifact registration and cannot be "
            "embedded as arbitrary module objects."
        )
    if callable(getattr(core, "context_builder", None)):
        raise ArtifactExtensionError(
            "Callable context builders cannot be embedded in a safe artifact; use a "
            "registered string identifier."
        )

    preprocessing = _preprocessing_metadata(core)
    fitted = _fitted_metadata(core)
    output_schema = {
        "names": list(spec.input_schema.output_names),
        "shape": fitted["output_shape"],
        "task": spec.task.to_dict(),
    }
    payloads: dict[str, bytes] = {
        MODEL_CONFIG_PATH: json_bytes(spec.to_dict()),
        FITTED_CONFIG_PATH: json_bytes(fitted),
        INPUT_SCHEMA_PATH: json_bytes(spec.input_schema.to_dict()),
        OUTPUT_SCHEMA_PATH: json_bytes(output_schema),
        PREPROCESSING_PATH: json_bytes(preprocessing),
        WEIGHTS_PATH: _serialize_weights(core.model_),
    }
    file_roles: dict[str, str] = {
        MODEL_CONFIG_PATH: "model_configuration",
        FITTED_CONFIG_PATH: "fitted_metadata",
        INPUT_SCHEMA_PATH: "input_schema",
        OUTPUT_SCHEMA_PATH: "output_schema",
        PREPROCESSING_PATH: "preprocessing_state",
        WEIGHTS_PATH: "model_state_dict",
    }
    if model_card is not None:
        payloads[MODEL_CARD_PATH] = str(model_card).encode("utf-8")
        file_roles[MODEL_CARD_PATH] = "model_card"

    plugin = None
    if registration.plugin is not None:
        plugin = {
            "identifier": registration.plugin,
            "version": _plugin_version(
                registration.plugin,
                getattr(registration, "plugin_version", None),
            ),
        }
        plugin = {key: value for key, value in plugin.items() if value is not None}
    capabilities = [
        "native_inference",
        "batched_inference",
        "stateless_inference",
        "model_agnostic_explanations",
        "restricted_weights",
        "schema_validation",
    ]
    if spec.task.kind != "regression":
        capabilities.append("task_probabilities")
    context_builder = getattr(core, "context_builder", None)
    differentiable_context = context_builder is None or context_builder == "cosine"
    requires_explicit_context = (
        getattr(core, "_context_dim_", None) not in {None, 0} and context_builder is None
    )
    if (
        not isinstance(core, TorchModuleAdapter)
        and getattr(core, "_scaler_kind_", None) != "custom"
        and getattr(core, "_target_scaler_kind_", None) != "custom"
        and differentiable_context
        and not requires_explicit_context
        and not bool(getattr(core, "per_element", False))
        and spec.input_schema.categorical_encoder is None
        and spec.input_schema.missing_value_imputer is None
    ):
        capabilities.append("gradient_explanations")
    manifest = {
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "package": {"name": "psann", "version": __version__},
        "runtime": {
            "python": runtime_platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "requirements": {
            "python_min": "3.11",
            "psann_min": __version__,
            "torch_min": "2.4",
        },
        "model": {
            "backbone": spec.backbone,
            "task": spec.task.kind,
            "plugin": plugin,
        },
        "training": {"run_id": run_id},
        "capabilities": capabilities,
        "experimental": bool(registration.experimental),
        "files": file_roles,
        "required_extensions": _required_extensions(spec, registration),
        "metadata": _json_value(
            {
                **dict(metadata or {}),
                "fingerprints": {
                    **dict(dict(metadata or {}).get("fingerprints", {})),
                    "model": fingerprint_model(model),
                },
            },
            field="metadata",
        ),
        "registry": _json_value(dict(registry or {}), field="registry"),
    }
    validate_manifest(manifest)
    payloads[MANIFEST_PATH] = json_bytes(manifest)
    return write_bundle(path, payloads)


def inspect_artifact(path: str | os.PathLike[str]) -> ArtifactInfo:
    """Validate and inspect a `.psann` bundle without deserializing tensor state."""

    artifact = inspect_bundle(path)
    manifest = artifact.manifest
    package = manifest["package"]
    model = manifest["model"]
    training = manifest["training"]
    return ArtifactInfo(
        path=artifact.path,
        artifact_id=str(manifest["artifact_id"]),
        artifact_format_version=str(manifest["artifact_format_version"]),
        original_format_version=artifact.original_version,
        package_version=str(package["version"]),
        backbone=str(model["backbone"]),
        task=str(model["task"]),
        run_id=(str(training["run_id"]) if training.get("run_id") is not None else None),
        capabilities=tuple(str(item) for item in manifest["capabilities"]),
        experimental=bool(manifest["experimental"]),
        migrations=artifact.migrations,
        manifest=manifest,
    )


def is_model_artifact(path: str | os.PathLike[str]) -> bool:
    """Return whether a file is a fully valid native deployment artifact."""

    try:
        inspect_bundle(path)
    except ArtifactError:
        return False
    return True


def _parsed_version(value: str, *, field: str) -> Version:
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise ArtifactVersionError(
            f"{field} must contain a valid PEP 440 version; received {value!r}."
        ) from exc


def _validate_runtime(manifest: Mapping[str, Any]) -> None:
    requirements = manifest["requirements"]
    current = {
        "python_min": runtime_platform.python_version(),
        "psann_min": __version__,
        "torch_min": torch.__version__,
    }
    labels = {
        "python_min": "Python",
        "psann_min": "PSANN",
        "torch_min": "PyTorch",
    }
    for requirement, available in current.items():
        required = str(requirements[requirement])
        if _parsed_version(available, field=labels[requirement]) < _parsed_version(
            required,
            field=f"manifest.requirements.{requirement}",
        ):
            raise ArtifactVersionError(
                f"Artifact requires {labels[requirement]} >= {required}, but "
                f"{available} is installed. Upgrade the runtime before loading."
            )


def _validate_extensions(manifest: Mapping[str, Any], spec: ModelSpec) -> Any:
    try:
        registration = BACKBONES.resolve(spec.backbone)
    except ValueError as exc:
        plugin = manifest["model"].get("plugin")
        hint = (
            f" Install and register plugin {plugin.get('identifier')!r} first."
            if isinstance(plugin, Mapping)
            else ""
        )
        raise ArtifactExtensionError(
            f"Artifact requires unregistered backbone {spec.backbone!r}.{hint}"
        ) from exc
    declared_plugin = manifest["model"].get("plugin")
    if isinstance(declared_plugin, Mapping):
        identifier = str(declared_plugin.get("identifier"))
        if registration.plugin != identifier:
            raise ArtifactExtensionError(
                f"Artifact requires backbone plugin {identifier!r}, but the registered "
                f"backbone declares {registration.plugin!r}."
            )
        required_version = declared_plugin.get("version")
        available_version = _plugin_version(
            registration.plugin,
            getattr(registration, "plugin_version", None),
        )
        if (
            required_version is not None
            and available_version is not None
            and _parsed_version(str(available_version), field="plugin version")
            < _parsed_version(str(required_version), field="required plugin version")
        ):
            raise ArtifactExtensionError(
                f"Artifact requires plugin {identifier!r} >= {required_version}, "
                f"but {available_version} is registered."
            )
    for extension in manifest["required_extensions"]:
        kind = extension["kind"]
        identifier = extension["identifier"]
        try:
            if kind == "categorical_encoder":
                CATEGORICAL_ENCODERS.resolve(identifier)
            elif kind == "missing_value_imputer":
                MISSING_VALUE_IMPUTERS.resolve(identifier)
        except ValueError as exc:
            raise ArtifactExtensionError(
                f"Artifact requires registered {kind} {identifier!r}; register the "
                "extension before loading."
            ) from exc
    return registration


def _load_native(artifact: ValidatedArtifact, *, device: str | torch.device) -> Any:
    manifest = artifact.manifest
    _validate_runtime(manifest)
    payloads = all_payloads(artifact)
    spec = ModelSpec.from_dict(parse_json(payloads[MODEL_CONFIG_PATH], member=MODEL_CONFIG_PATH))
    input_schema = DataSchema.from_dict(
        parse_json(payloads[INPUT_SCHEMA_PATH], member=INPUT_SCHEMA_PATH)
    )
    if input_schema != spec.input_schema:
        raise ArtifactFormatError("Artifact input schema does not match config/model.json.")
    if (
        manifest["model"]["backbone"] != spec.backbone
        or manifest["model"]["task"] != spec.task.kind
    ):
        raise ArtifactFormatError(
            "Artifact manifest model identity does not match config/model.json."
        )
    registration = _validate_extensions(manifest, spec)
    fitted = parse_json(payloads[FITTED_CONFIG_PATH], member=FITTED_CONFIG_PATH)
    preprocessing = parse_json(
        payloads[PREPROCESSING_PATH],
        member=PREPROCESSING_PATH,
    )
    output_schema = parse_json(payloads[OUTPUT_SCHEMA_PATH], member=OUTPUT_SCHEMA_PATH)
    if output_schema.get("task") != spec.task.to_dict():
        raise ArtifactFormatError("Artifact output schema task does not match config/model.json.")
    resolved_device = choose_device(device)
    state_dict = _deserialize_weights(payloads[WEIGHTS_PATH], device=resolved_device)
    model = create_model(spec)
    if isinstance(model, TorchModuleAdapter):
        loaded = _restore_module_adapter(
            model,
            spec=spec,
            fitted=fitted,
            preprocessing=preprocessing,
            state_dict=state_dict,
            device=resolved_device,
        )
    elif isinstance(model, PSANNClassifier):
        loaded = _restore_classifier(
            model,
            spec=spec,
            registration=registration,
            fitted=fitted,
            preprocessing=preprocessing,
            state_dict=state_dict,
            device=resolved_device,
        )
    else:
        loaded = _restore_core(
            model,
            spec=spec,
            fitted=fitted,
            preprocessing=preprocessing,
            state_dict=state_dict,
            device=resolved_device,
        )
    loaded.artifact_info_ = inspect_artifact(artifact.path)
    loaded.artifact_id_ = str(manifest["artifact_id"])
    loaded.run_id_ = manifest["training"].get("run_id")
    return loaded


def _load_trusted_legacy(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device,
) -> Any:
    resolved_device = choose_device(device)
    warnings.warn(
        LEGACY_CHECKPOINT_MESSAGE,
        LegacyCheckpointWarning,
        stacklevel=3,
    )
    try:
        value = torch.load(
            Path(path),
            map_location=resolved_device,
            weights_only=False,
        )
    except TypeError:  # pragma: no cover - older supported Torch compatibility
        value = torch.load(Path(path), map_location=resolved_device)
    if isinstance(value, PSANNClassifier):
        value.estimator_.device = str(resolved_device)
        value.estimator_._resolved_training_device_ = resolved_device
        value.estimator_._ensure_model_device(resolved_device)
        value.model_ = value.estimator_.model_
        return value
    if isinstance(value, Mapping):
        class_name = value.get("class")
        from ..sklearn import (
            PSANNRegressor,
            ResConvPSANNRegressor,
            ResPSANNRegressor,
            SGRPSANNRegressor,
            WaveResNetRegressor,
        )

        classes = {
            cls.__name__: cls
            for cls in (
                PSANNRegressor,
                ResPSANNRegressor,
                ResConvPSANNRegressor,
                WaveResNetRegressor,
                SGRPSANNRegressor,
            )
        }
        estimator_class = classes.get(str(class_name))
        if estimator_class is None:
            raise ArtifactFormatError(f"Unsupported trusted legacy estimator class {class_name!r}.")
        return estimator_class.load(str(path), map_location=resolved_device)
    if hasattr(value, "predict") and isinstance(getattr(value, "model_", None), torch.nn.Module):
        return value
    raise ArtifactFormatError(
        f"Trusted legacy checkpoint contains unsupported object {type(value).__name__}."
    )


def load_model(
    path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
    trusted_legacy_checkpoint: bool = False,
) -> Any:
    """Load a safe native artifact or an explicitly trusted legacy checkpoint."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ArtifactFormatError(f"Model artifact does not exist: {source}")
    if is_training_checkpoint(source):
        raise ArtifactFormatError(
            "This file is a resumable `.psann-train` checkpoint, not a deployment "
            "artifact. Pass it to train(..., resume_from=...) instead."
        )
    try:
        artifact = inspect_bundle(source)
    except ArtifactError:
        if source.suffix.lower() == ".psann":
            raise
        if not trusted_legacy_checkpoint:
            raise LegacyCheckpointTrustError(
                "The file is not a valid native `.psann` artifact. Legacy checkpoint "
                "loading may execute arbitrary Python; set "
                "trusted_legacy_checkpoint=True only after verifying its source."
            )
        return _load_trusted_legacy(source, device=device)
    return _load_native(artifact, device=device)


_LEGACY_TRAINING_FIELDS = {
    "amp",
    "amp_dtype",
    "batch_size",
    "compile",
    "compile_backend",
    "compile_dynamic",
    "compile_fullgraph",
    "compile_mode",
    "device",
    "early_stopping",
    "epochs",
    "loss",
    "loss_params",
    "loss_reduction",
    "lr",
    "num_workers",
    "optimizer",
    "patience",
    "random_state",
    "stream_lr",
    "warm_start",
    "weight_decay",
}


def _infer_legacy_spec(model: Any) -> ModelSpec:
    existing = getattr(model, "_platform_model_spec_dict_", None)
    if existing is not None:
        return _actual_model_spec(model, ModelSpec.from_dict(existing))
    core = _fitted_core(model)
    class_to_backbone = {
        "PSANNRegressor": "psann_mlp",
        "ResPSANNRegressor": "respsann_mlp",
        "ResConvPSANNRegressor": "respsann_conv2d",
        "WaveResNetRegressor": "wave_resnet",
        "SGRPSANNRegressor": "sgr_psann",
    }
    backbone = class_to_backbone.get(type(core).__name__)
    if type(core).__name__ == "PSANNRegressor" and bool(getattr(core, "preserve_shape", False)):
        input_rank = len(tuple(getattr(core, "input_shape_", ())))
        backbone = {
            2: "psann_conv1d",
            3: "psann_conv2d",
            4: "psann_conv3d",
        }.get(input_rank)
    if backbone is None:
        raise ArtifactExtensionError(
            f"Cannot infer a registered backbone for legacy class {type(core).__name__}; "
            "pass model_spec explicitly."
        )
    parameters: dict[str, Any] = {}
    for name, value in core.get_params(deep=False).items():
        if name in _LEGACY_TRAINING_FIELDS or name in {
            "activation_type",
            "data_format",
            "dropout",
            "hidden_width",
            "norm",
            "normalization",
            "preserve_shape",
        }:
            continue
        if callable(value):
            raise ArtifactExtensionError(
                f"Legacy parameter {name!r} is callable and cannot be migrated safely."
            )
        try:
            parameters[name] = _json_value(value, field=f"legacy.parameters.{name}")
        except ArtifactFormatError:
            if value is not None:
                raise
    if "conv" not in backbone:
        for name in ("conv_channels", "conv_kernel_size", "per_element"):
            parameters.pop(name, None)
    schema = DataSchema(
        feature_names=tuple(str(item) for item in getattr(core, "feature_names_in_", ())),
        output_names=tuple(str(item) for item in getattr(core, "output_names_", ())),
        input_shape=tuple(int(item) for item in getattr(core, "input_shape_", ())),
        data_format=str(getattr(core, "data_format", "channels_first")),  # type: ignore[arg-type]
        dtype=str(getattr(core, "input_dtype_", "float32") or "float32"),
        feature_policy=str(getattr(core, "feature_schema_policy_", "strict")),  # type: ignore[arg-type]
    )
    spec = ModelSpec(
        task=TaskSpec(kind="regression"),
        backbone=backbone,
        input_schema=schema,
        activation=str(getattr(core, "activation_type", "psann")),
        normalization=str(getattr(core, "norm", "none")),
        dropout=float(getattr(core, "dropout", 0.0)),
        parameters=parameters,
    )
    return _actual_model_spec(model, spec)


def migrate_legacy_checkpoint(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    trusted_legacy_checkpoint: bool = False,
    model_spec: ModelSpec | Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
    model_card: str | None = None,
) -> Path:
    """Convert an explicitly trusted whole-object checkpoint into a safe artifact."""

    if not trusted_legacy_checkpoint:
        raise LegacyCheckpointTrustError(
            "Legacy migration requires trusted_legacy_checkpoint=True because reading "
            "the source may execute arbitrary Python."
        )
    model = _load_trusted_legacy(source, device=device)
    spec = (
        model_spec
        if isinstance(model_spec, ModelSpec)
        else (
            ModelSpec.from_dict(model_spec) if model_spec is not None else _infer_legacy_spec(model)
        )
    )
    return export_model(
        model,
        destination,
        model_spec=spec,
        model_card=model_card,
        metadata={"migrated_from_legacy_checkpoint": True},
    )


def migrate_artifact(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> Path:
    """Rewrite a supported historical artifact using the current manifest schema."""

    artifact = inspect_bundle(source)
    payloads = all_payloads(artifact)
    manifest = dict(artifact.manifest)
    metadata = dict(manifest.get("metadata", {}))
    metadata["migrated_from_artifact_format_version"] = artifact.original_version
    manifest["metadata"] = metadata
    manifest["artifact_format_version"] = ARTIFACT_FORMAT_VERSION
    manifest["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    validate_manifest(manifest)
    payloads[MANIFEST_PATH] = json_bytes(manifest)
    return write_bundle(destination, payloads)


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ArtifactError",
    "ArtifactExtensionError",
    "ArtifactFormatError",
    "ArtifactInfo",
    "ArtifactVersionError",
    "LegacyCheckpointTrustError",
    "LegacyCheckpointWarning",
    "export_model",
    "inspect_artifact",
    "is_model_artifact",
    "load_model",
    "migrate_artifact",
    "migrate_legacy_checkpoint",
]
