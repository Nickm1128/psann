"""Model-state conversion and registered estimator reconstruction helpers."""

from __future__ import annotations

import importlib.metadata
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._sklearn.classifier import PSANNClassifier
from ..estimators._fit_contracts import build_model_signature
from ..estimators._fit_types import ModelBuildRequest, PreparedInputState
from ..estimators._fit_utils import build_model_from_hooks
from .artifact_schema import (
    ArtifactExtensionError,
    ArtifactFormatError,
)
from .module_adapter import TorchModuleAdapter
from .specs import DataSchema, ModelSpec, TaskSpec
from .tasks import create_task_adapter


def json_value(value: Any, *, field: str) -> Any:
    """Convert supported metadata values into finite JSON-safe structures."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value), field=field)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactFormatError(f"{field} mapping keys must be strings.")
            result[key] = json_value(item, field=f"{field}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [json_value(item, field=field) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactFormatError(f"{field} cannot contain NaN or infinity.")
        return value
    raise ArtifactFormatError(
        f"{field} cannot contain runtime object {type(value).__module__}."
        f"{type(value).__qualname__}."
    )


def _safe_weight_value(value: Any, *, field: str) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactFormatError(f"{field} state keys must be strings.")
            result[key] = _safe_weight_value(item, field=f"{field}.{key}")
        return result
    if isinstance(value, tuple):
        return tuple(_safe_weight_value(item, field=field) for item in value)
    if isinstance(value, list):
        return [_safe_weight_value(item, field=field) for item in value]
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    raise ArtifactFormatError(
        f"{field} contains unsupported state value {type(value).__module__}."
        f"{type(value).__qualname__}; artifacts only permit tensors and primitives."
    )


def serialize_weights(model: torch.nn.Module) -> bytes:
    state = {
        str(name): _safe_weight_value(value, field=f"state_dict.{name}")
        for name, value in model.state_dict().items()
    }
    buffer = io.BytesIO()
    torch.save({"state_dict": state}, buffer)
    return buffer.getvalue()


def deserialize_weights(
    payload: bytes,
    *,
    device: torch.device,
) -> Mapping[str, Any]:
    try:
        value = torch.load(
            io.BytesIO(payload),
            map_location=device,
            weights_only=True,
        )
    except Exception as exc:
        raise ArtifactFormatError(
            f"Artifact weights could not be restricted-loaded: {exc}"
        ) from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("state_dict"), Mapping):
        raise ArtifactFormatError(
            "Artifact weights must contain a primitive/tensor state_dict mapping."
        )
    return value["state_dict"]


def fitted_core(model: Any) -> Any:
    if isinstance(model, TorchModuleAdapter):
        identifier = str(getattr(model, "backbone_id_", "arbitrary_module"))
        if identifier == "arbitrary_module" or not hasattr(model, "_platform_model_spec_dict_"):
            raise ArtifactExtensionError(
                "Arbitrary TorchModuleAdapter instances do not have a portable artifact "
                "guarantee. Register a reconstructable backbone first."
            )
        if not isinstance(getattr(model, "model_", None), torch.nn.Module):
            raise ArtifactFormatError("Cannot export an unfitted model; train it first.")
        return model
    core = model.estimator_ if isinstance(model, PSANNClassifier) else model
    if not isinstance(getattr(core, "model_", None), torch.nn.Module):
        raise ArtifactFormatError("Cannot export an unfitted model; train it first.")
    required = ("_make_fit_hooks", "_resolve_lsm_module", "_ensure_model_device")
    if any(not hasattr(core, name) for name in required):
        raise ArtifactExtensionError(
            "The registered backbone does not implement the PSANN artifact reconstruction protocol."
        )
    return core


def actual_model_spec(
    model: Any,
    supplied: ModelSpec | Mapping[str, Any] | None,
) -> ModelSpec:
    value = supplied or getattr(model, "_platform_model_spec_dict_", None)
    if value is None:
        raise ArtifactFormatError(
            "Portable export requires a ModelSpec. Create the model with psann.create_model "
            "or pass model_spec explicitly."
        )
    spec = value if isinstance(value, ModelSpec) else ModelSpec.from_dict(value)
    core = fitted_core(model)
    task = getattr(model, "task_spec_", spec.task)
    if not isinstance(task, TaskSpec):
        task = TaskSpec.from_dict(task)
    source_schema = spec.input_schema
    feature_names = tuple(
        str(item)
        for item in getattr(
            model,
            "feature_names_in_",
            getattr(core, "feature_names_in_", source_schema.feature_names),
        )
    )
    output_names = tuple(
        str(item)
        for item in getattr(
            model,
            "output_names_",
            getattr(core, "output_names_", source_schema.output_names),
        )
    )
    input_shape = tuple(
        int(item)
        for item in getattr(
            core,
            "input_shape_contract_",
            getattr(core, "input_shape_", source_schema.input_shape),
        )
    )
    schema = DataSchema(
        feature_names=feature_names,
        output_names=output_names,
        input_shape=input_shape,
        data_format=str(getattr(core, "data_format_", source_schema.data_format)),  # type: ignore[arg-type]
        dtype=str(getattr(core, "input_dtype_", source_schema.dtype) or source_schema.dtype),
        feature_policy=str(getattr(core, "feature_schema_policy_", source_schema.feature_policy)),  # type: ignore[arg-type]
        preprocessing=source_schema.preprocessing,
        target_scaling=source_schema.target_scaling,
        categorical_encoder=source_schema.categorical_encoder,
        missing_value_imputer=source_schema.missing_value_imputer,
    )
    return ModelSpec(
        task=task,
        backbone=spec.backbone,
        input_schema=schema,
        activation=spec.activation,
        normalization=spec.normalization,
        dropout=spec.dropout,
        parameters=spec.parameters,
    )


def fitted_metadata(core: Any) -> Mapping[str, Any]:
    input_shape = tuple(int(item) for item in getattr(core, "input_shape_", ()))
    if not input_shape:
        raise ArtifactFormatError("Fitted model is missing input_shape_ metadata.")
    return json_value(
        {
            "input_shape": input_shape,
            "internal_input_shape_cf": getattr(core, "_internal_input_shape_cf_", None),
            "primary_dim": getattr(core, "_primary_dim_", None),
            "output_dim": getattr(core, "_output_dim_", None),
            "keep_column_output": bool(getattr(core, "_keep_column_output_", False)),
            "train_layout": getattr(core, "_train_inputs_layout_", "flat"),
            "target_cf_shape": getattr(core, "_target_cf_shape_", None),
            "target_vector_dim": getattr(core, "_target_vector_dim_", None),
            "output_shape": getattr(core, "_output_shape_tuple_", None),
            "context_dim": getattr(core, "_context_dim_", None),
            "n_features_in": getattr(core, "n_features_in_", input_shape[0]),
            "feature_names": tuple(str(item) for item in getattr(core, "feature_names_in_", ())),
            "output_names": tuple(str(item) for item in getattr(core, "output_names_", ())),
            "input_dtype": getattr(core, "input_dtype_", "float32"),
            "feature_policy": getattr(core, "feature_schema_policy_", "strict"),
            "data_format": getattr(core, "data_format_", core.data_format),
            "progressive_depth_current": getattr(core, "_progressive_depth_current", None),
            "w0_schedule_step": getattr(core, "_w0_schedule_step", None),
        },
        field="fitted",
    )


def preprocessing_metadata(core: Any) -> Mapping[str, Any]:
    contract = dict(getattr(core, "preprocessing_contract_", {}))
    for label, kind in (
        ("input", getattr(core, "_scaler_kind_", None)),
        ("target", getattr(core, "_target_scaler_kind_", None)),
    ):
        if kind == "custom":
            raise ArtifactExtensionError(
                f"Custom {label} scaler objects cannot be embedded in a safe artifact. "
                "Use a built-in scaler or an approved identifier-based transform."
            )
        if kind not in {None, "standard", "minmax"}:
            raise ArtifactExtensionError(f"Unsupported fitted {label} scaler kind {kind!r}.")
    return json_value(contract, field="preprocessing")


def plugin_version(identifier: str | None, explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    if identifier is None:
        return None
    try:
        return importlib.metadata.version(identifier)
    except importlib.metadata.PackageNotFoundError:
        return None


def required_extensions(spec: ModelSpec, registration: Any) -> list[dict[str, Any]]:
    extensions: list[dict[str, Any]] = []
    if registration.plugin is not None:
        extension: dict[str, Any] = {
            "kind": "backbone_plugin",
            "identifier": registration.plugin,
        }
        version = plugin_version(
            registration.plugin,
            getattr(registration, "plugin_version", None),
        )
        if version is not None:
            extension["version"] = version
        extensions.append(extension)
    for kind, identifier in (
        ("categorical_encoder", spec.input_schema.categorical_encoder),
        ("missing_value_imputer", spec.input_schema.missing_value_imputer),
    ):
        if identifier is not None:
            extensions.append({"kind": kind, "identifier": identifier})
    return extensions


def _array_state(state: Any) -> Any:
    if not isinstance(state, Mapping):
        return state
    result = dict(state)
    for key in ("mean", "M2", "min", "max"):
        if key in result:
            result[key] = np.asarray(result[key], dtype=np.float64)
    return result


def _restore_preprocessing(core: Any, preprocessing: Mapping[str, Any]) -> None:
    input_scaler = preprocessing.get("input_scaler", {})
    target_scaler = preprocessing.get("target_scaler", {})
    if not isinstance(input_scaler, Mapping) or not isinstance(target_scaler, Mapping):
        raise ArtifactFormatError("Artifact preprocessing scaler entries must be objects.")
    core._scaler_kind_ = input_scaler.get("kind")
    core._scaler_state_ = _array_state(input_scaler.get("state"))
    core._scaler_fitted_ = core._scaler_kind_ is not None
    core._target_scaler_kind_ = target_scaler.get("kind")
    core._target_scaler_state_ = _array_state(target_scaler.get("state"))
    core._target_scaler_fitted_ = core._target_scaler_kind_ is not None
    core.preprocessing_contract_ = dict(preprocessing)


def restore_core(
    core: Any,
    *,
    spec: ModelSpec,
    fitted: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
) -> Any:
    input_shape = tuple(int(item) for item in fitted.get("input_shape", ()))
    if not input_shape:
        raise ArtifactFormatError("Artifact fitted.input_shape cannot be empty.")
    primary_dim = int(fitted.get("primary_dim") or 0)
    output_dim = int(fitted.get("output_dim") or primary_dim)
    if primary_dim < 1 or output_dim < 1:
        raise ArtifactFormatError("Artifact fitted output dimensions must be positive.")
    internal_value = fitted.get("internal_input_shape_cf")
    internal_shape = (
        tuple(int(item) for item in internal_value)
        if isinstance(internal_value, Sequence)
        else None
    )
    if core.preserve_shape and internal_shape is None:
        internal_shape = (
            (input_shape[-1], *input_shape[:-1])
            if core.data_format == "channels_last"
            else input_shape
        )
    batch = 2
    flat: np.ndarray = np.zeros((batch, int(np.prod(input_shape))), dtype=np.float32)
    channel_first: np.ndarray | None = (
        np.zeros((batch, *internal_shape), dtype=np.float32) if internal_shape is not None else None
    )
    layout = str(fitted.get("train_layout", "flat"))
    train_inputs = channel_first if layout == "cf" else flat
    if train_inputs is None:
        raise ArtifactFormatError("Artifact requests channel-first layout without its shape.")
    context_dim = fitted.get("context_dim")
    context_width = int(context_dim) if context_dim is not None else None
    context: np.ndarray | None = (
        np.zeros((batch, context_width), dtype=np.float32)
        if context_width not in {None, 0}
        else None
    )
    target_cf_value = fitted.get("target_cf_shape")
    target_cf_shape = (
        tuple(int(item) for item in target_cf_value)
        if isinstance(target_cf_value, Sequence)
        else None
    )
    target_cf: np.ndarray | None = (
        np.zeros((batch, *target_cf_shape), dtype=np.float32)
        if target_cf_shape is not None
        else None
    )
    target_vector_dim = int(fitted.get("target_vector_dim") or primary_dim)
    target_vector: np.ndarray = np.zeros((batch, target_vector_dim), dtype=np.float32)
    prepared = PreparedInputState(
        X_flat=flat,
        X_cf=channel_first,
        context=context,
        input_shape=input_shape,
        internal_shape_cf=internal_shape,
        scaler_transform=None,
        train_inputs=train_inputs,
        train_context=context,
        train_targets=target_cf if target_cf is not None and layout == "cf" else target_vector,
        y_vector=target_vector,
        y_cf=target_cf,
        context_dim=context_width,
        primary_dim=primary_dim,
        output_dim=output_dim,
    )

    core.input_shape_ = input_shape
    core._internal_input_shape_cf_ = internal_shape
    core._primary_dim_ = primary_dim
    core._output_dim_ = output_dim
    core._keep_column_output_ = bool(fitted.get("keep_column_output", False))
    core._train_inputs_layout_ = layout
    core._target_cf_shape_ = target_cf_shape
    core._target_vector_dim_ = target_vector_dim
    output_shape = fitted.get("output_shape")
    core._output_shape_tuple_ = (
        tuple(int(item) for item in output_shape) if isinstance(output_shape, Sequence) else None
    )
    core._context_dim_ = context_width
    if hasattr(core, "context_dim"):
        core.context_dim = context_width
    progressive_depth = fitted.get("progressive_depth_current")
    if progressive_depth is not None and hasattr(core, "progressive_depth_initial"):
        core.progressive_depth_initial = int(progressive_depth)
    if getattr(core, "lsm", None) is not None:
        raise ArtifactExtensionError("Artifact reconstruction does not permit opaque LSM modules.")

    lsm_model, lsm_dim = core._resolve_lsm_module(
        train_inputs,
        preserve_shape=bool(core.preserve_shape and core.per_element),
    )
    hooks = core._make_fit_hooks(prepared=prepared, verbose=0)
    request = ModelBuildRequest(
        estimator=core,
        prepared=prepared,
        primary_dim=primary_dim,
        lsm_module=lsm_model,
        lsm_output_dim=lsm_dim,
        preserve_shape=bool(core.preserve_shape),
    )
    core.model_ = build_model_from_hooks(hooks, request)
    core._model_signature_ = build_model_signature(core, prepared)
    core._model_rebuilt_ = True
    core.device = str(device)
    core._resolved_training_device_ = device
    core._model_device_ = None
    core._ensure_model_device(device)
    core._after_model_built()
    try:
        core.model_.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as exc:
        raise ArtifactFormatError(
            f"Artifact weights do not match the registered model configuration: {exc}"
        ) from exc
    w0_schedule_step = fitted.get("w0_schedule_step")
    if (
        w0_schedule_step is not None
        and hasattr(core, "_current_w0_values")
        and hasattr(core, "_apply_w0_values")
    ):
        core._w0_schedule_step = int(w0_schedule_step)
        first_w0, hidden_w0 = core._current_w0_values()
        core._apply_w0_values(first_w0, hidden_w0)
        core._w0_schedule_active = False
    core.model_.eval()
    core._optimizer_ = None
    core._lr_scheduler_ = None
    core._amp_scaler_ = None
    core.history_ = []
    core.training_events_ = []
    core.training_metadata_ = {"loaded_from_artifact": True}
    _restore_preprocessing(core, preprocessing)

    core.n_features_in_ = int(fitted.get("n_features_in", input_shape[0]))
    feature_names = tuple(str(item) for item in fitted.get("feature_names", ()))
    if feature_names:
        core.feature_names_in_ = np.asarray(feature_names, dtype=object)
    output_names = tuple(str(item) for item in fitted.get("output_names", ()))
    if output_names:
        core.output_names_ = np.asarray(output_names, dtype=object)
    core.input_shape_contract_ = input_shape
    core.input_dtype_ = str(fitted.get("input_dtype", spec.input_schema.dtype))
    core.feature_schema_policy_ = str(
        fitted.get("feature_policy", spec.input_schema.feature_policy)
    )
    core.data_format_ = str(fitted.get("data_format", spec.input_schema.data_format))
    core.task_metadata_ = spec.task.to_dict()
    core._platform_model_spec_dict_ = spec.to_dict()
    core._platform_data_schema_ = spec.input_schema
    core._platform_task_spec_ = spec.task
    core.backbone_id_ = spec.backbone
    return core


def restore_classifier(
    wrapper: PSANNClassifier,
    *,
    spec: ModelSpec,
    registration: Any,
    fitted: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
) -> PSANNClassifier:
    classes = tuple(spec.task.class_names)
    if spec.task.kind in {"binary", "multiclass"} and len(classes) < 2:
        raise ArtifactFormatError("Classification artifact is missing fitted class names.")
    if spec.task.kind == "multilabel" and not classes:
        raise ArtifactFormatError("Multilabel artifact is missing fitted label names.")
    output_width = 1 if spec.task.kind == "binary" else len(classes)
    parameters = dict(wrapper.estimator_params or {})
    parameters["output_shape"] = (output_width,)
    parameters["target_scaler"] = None
    core = registration.factory(parameters)
    if not hasattr(core, "_make_fit_hooks"):
        raise ArtifactExtensionError(
            "Registered classifier backbone does not implement artifact reconstruction."
        )
    core._platform_data_schema_ = spec.input_schema
    core._platform_task_spec_ = spec.task
    core = restore_core(
        core,
        spec=spec,
        fitted=fitted,
        preprocessing=preprocessing,
        state_dict=state_dict,
        device=device,
    )
    adapter = create_task_adapter(spec.task)
    adapter.classes = classes
    adapter.output_names = (
        (str(classes[1]),) if spec.task.kind == "binary" else tuple(str(item) for item in classes)
    )
    wrapper.estimator_ = core
    wrapper.task_adapter_ = adapter
    wrapper.task_spec_ = spec.task
    wrapper.classes_ = np.asarray(classes, dtype=object)
    wrapper.n_outputs_ = output_width
    wrapper.output_names_ = np.asarray(adapter.output_names, dtype=object)
    wrapper.n_features_in_ = core.n_features_in_
    if hasattr(core, "feature_names_in_"):
        wrapper.feature_names_in_ = core.feature_names_in_.copy()
    wrapper.input_shape_ = core.input_shape_
    wrapper.history_ = []
    wrapper.training_events_ = []
    wrapper.training_metadata_ = dict(core.training_metadata_)
    wrapper.preprocessing_contract_ = dict(core.preprocessing_contract_)
    wrapper.model_ = core.model_
    return wrapper


def restore_module_adapter(
    adapter: TorchModuleAdapter,
    *,
    spec: ModelSpec,
    fitted: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
) -> TorchModuleAdapter:
    """Restore a registered Torch module without deserializing its Python class."""

    module = adapter.module.to(device)
    try:
        module.load_state_dict(dict(state_dict), strict=True)
    except RuntimeError as exc:
        raise ArtifactFormatError(
            f"Artifact weights do not match the registered module factory: {exc}"
        ) from exc
    module.eval()
    input_shape = tuple(int(item) for item in fitted.get("input_shape", ()))
    if not input_shape:
        raise ArtifactFormatError("Artifact fitted.input_shape cannot be empty.")
    output_shape_value = fitted.get("output_shape")
    output_shape = (
        tuple(int(item) for item in output_shape_value)
        if isinstance(output_shape_value, Sequence)
        else ()
    )
    task_adapter = create_task_adapter(spec.task)
    task_adapter.classes = tuple(spec.task.class_names)
    task_adapter.output_names = tuple(spec.input_schema.output_names)

    adapter.model_ = module
    adapter.model = module
    adapter.task_adapter_ = task_adapter
    adapter.task_spec_ = spec.task
    adapter.classes_ = np.asarray(task_adapter.classes, dtype=object)
    adapter.output_names_ = np.asarray(task_adapter.output_names, dtype=object)
    adapter.input_shape_ = input_shape
    adapter.input_shape_contract_ = input_shape
    adapter.n_features_in_ = int(fitted.get("n_features_in", input_shape[0]))
    adapter.n_outputs_ = int(np.prod(output_shape)) if output_shape else 1
    adapter._primary_dim_ = adapter.n_outputs_
    adapter._output_dim_ = adapter.n_outputs_
    adapter._output_shape_tuple_ = output_shape or (adapter.n_outputs_,)
    adapter._device_ = device
    adapter.device = str(device)
    adapter.input_dtype_ = str(fitted.get("input_dtype", spec.input_schema.dtype))
    adapter.feature_schema_policy_ = str(
        fitted.get("feature_policy", spec.input_schema.feature_policy)
    )
    adapter.data_format_ = str(fitted.get("data_format", spec.input_schema.data_format))
    adapter.history_ = []
    adapter.training_events_ = []
    adapter.training_metadata_ = {
        "loaded_from_artifact": True,
        "adapter": "registered_torch_module",
    }
    adapter.preprocessing_contract_ = dict(preprocessing)
    adapter._platform_model_spec_dict_ = spec.to_dict()
    adapter._platform_data_schema_ = spec.input_schema
    adapter._platform_task_spec_ = spec.task
    adapter.backbone_id_ = spec.backbone
    feature_names = tuple(str(item) for item in fitted.get("feature_names", ()))
    if feature_names:
        adapter.feature_names_in_ = np.asarray(feature_names, dtype=object)
    return adapter


__all__ = [
    "actual_model_spec",
    "deserialize_weights",
    "fitted_core",
    "fitted_metadata",
    "json_value",
    "plugin_version",
    "preprocessing_metadata",
    "required_extensions",
    "restore_classifier",
    "restore_core",
    "restore_module_adapter",
    "serialize_weights",
]
