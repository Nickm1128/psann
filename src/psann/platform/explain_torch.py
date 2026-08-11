"""Differentiable raw-input inference for capability-gated SHAP explainers."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np
import torch

from .inference import InferenceRuntime, _core_estimator

EXPLAINABLE_LAYERS: dict[str, dict[str, str]] = {
    "psann_mlp": {"hidden_0": "body.0.linear", "output": "head"},
    "respsann_mlp": {"input": "in_linear", "output": "head"},
    "psann_conv1d": {"conv_0": "core.body.0.conv", "output": "core.fc"},
    "psann_conv2d": {"conv_0": "core.body.0.conv", "output": "core.fc"},
    "psann_conv3d": {"conv_0": "core.body.0.conv", "output": "core.fc"},
    "respsann_conv2d": {"input": "core.in_proj", "output": "core.fc"},
    "wave_resnet": {"input": "stem", "output": "head"},
    "sgr_psann": {"hidden_0": "body.0.linear", "output": "head"},
}


def _model_spec(runtime: InferenceRuntime) -> Mapping[str, Any]:
    value = getattr(runtime.model, "_platform_model_spec_dict_", None)
    if value is None:
        value = getattr(_core_estimator(runtime.model), "_platform_model_spec_dict_", None)
    return value if isinstance(value, Mapping) else {}


def _backbone(runtime: InferenceRuntime) -> str:
    return str(_model_spec(runtime).get("backbone", "unknown"))


def register_explainable_layer(backbone: str, name: str, module_path: str) -> None:
    """Register a public layer alias without exposing private module traversal."""

    backbone_key = str(backbone).strip()
    name_key = str(name).strip()
    path_value = str(module_path).strip()
    if not backbone_key or not name_key or not path_value:
        raise ValueError("backbone, name, and module_path must be non-empty.")
    EXPLAINABLE_LAYERS.setdefault(backbone_key, {})[name_key] = path_value


def list_explainable_layers(model: Any) -> tuple[str, ...]:
    """Return registered public aliases that resolve for a fitted model."""

    runtime = model if isinstance(model, InferenceRuntime) else InferenceRuntime(model)
    module = _core_estimator(runtime.model).model_
    available: list[str] = []
    for name, path in EXPLAINABLE_LAYERS.get(_backbone(runtime), {}).items():
        try:
            module.get_submodule(path)
        except (AttributeError, KeyError):
            continue
        available.append(name)
    return tuple(available)


def differentiable_inference_reasons(
    runtime: InferenceRuntime,
    *,
    algorithm: str,
    layer: str | None,
) -> tuple[str, ...]:
    """Return every reason a raw-input gradient adapter is not certified."""

    core = _core_estimator(runtime.model)
    spec = _model_spec(runtime)
    reasons: list[str] = []
    if getattr(core, "_scaler_kind_", None) == "custom":
        reasons.append("custom input scaler is not differentiable through the raw-input adapter")
    if getattr(core, "_target_scaler_kind_", None) == "custom":
        reasons.append("custom target scaler is not differentiable through the output adapter")
    context_builder = getattr(core, "context_builder", None)
    if callable(context_builder):
        reasons.append("custom context builder is not registered as differentiable")
    if (
        context_builder is not None
        and context_builder != "cosine"
        and not callable(context_builder)
    ):
        reasons.append(f"context builder {context_builder!r} is not differentiable")
    if getattr(core, "_context_dim_", None) not in {None, 0} and context_builder is None:
        reasons.append("model requires explicit context that is not part of the raw-input game")
    schema = getattr(core, "_platform_data_schema_", None)
    for field in ("categorical_encoder", "missing_value_imputer"):
        if getattr(schema, field, None) is not None:
            reasons.append(f"schema transform {field!r} has no differentiable Torch adapter")
    if bool(getattr(core, "per_element", False)):
        reasons.append("per-element outputs are not certified for gradient explainers")
    if algorithm == "deep":
        if str(spec.get("backbone", "")) not in {"psann_mlp", "respsann_mlp"}:
            reasons.append("DeepExplainer is certified only for dense PSANN backbones")
        if str(spec.get("activation", "")) != "relu":
            reasons.append("DeepExplainer is certified only for the ReLU activation")
        if bool(getattr(core, "stateful", False)):
            reasons.append("DeepExplainer is not certified for stateful models")
    if layer is not None:
        registered = EXPLAINABLE_LAYERS.get(_backbone(runtime), {})
        if layer not in registered:
            reasons.append(f"layer {layer!r} is not registered; available={sorted(registered)!r}")
        else:
            try:
                core.model_.get_submodule(registered[layer])
            except (AttributeError, KeyError):
                reasons.append(f"registered layer {layer!r} is absent from this fitted model")
    return tuple(reasons)


def _state_tensor(
    state: Mapping[str, Any] | None,
    key: str,
    *,
    default: float = 0.0,
) -> torch.Tensor:
    if state is None or state.get(key) is None:
        return torch.as_tensor([default], dtype=torch.float32)
    return torch.as_tensor(np.asarray(state[key]), dtype=torch.float32)


class DifferentiableInferenceAdapter(torch.nn.Module):
    """Frozen Torch module that maps raw flattened inputs to selected task outputs."""

    def __init__(
        self,
        runtime: InferenceRuntime,
        *,
        output_kind: str,
        output_indices: tuple[int, ...],
    ) -> None:
        super().__init__()
        reasons = differentiable_inference_reasons(
            runtime,
            algorithm="gradient",
            layer=None,
        )
        if reasons:
            raise ValueError("; ".join(reasons))
        core = _core_estimator(runtime.model)
        self.model = copy.deepcopy(core.model_).to(runtime.device).eval()
        self.model.requires_grad_(False)
        if hasattr(self.model, "set_state_updates"):
            self.model.set_state_updates(False)
        self.input_shape = tuple(int(item) for item in core.input_shape_)
        self.preserve_shape = bool(core.preserve_shape)
        self.per_element = bool(core.per_element)
        self.data_format = str(core.data_format)
        self.use_channel_first = bool(
            self.preserve_shape
            and (self.per_element or getattr(core, "_use_channel_first_train_inputs_", False))
        )
        self.task = runtime.task
        self.output_kind = str(output_kind)
        self.output_indices = tuple(int(item) for item in output_indices)
        self.input_scaler_kind = getattr(core, "_scaler_kind_", None)
        self.target_scaler_kind = getattr(core, "_target_scaler_kind_", None)
        input_state = getattr(core, "_scaler_state_", None)
        target_state = getattr(core, "_target_scaler_state_", None)
        self.register_buffer("input_mean", _state_tensor(input_state, "mean"))
        self.register_buffer("input_m2", _state_tensor(input_state, "M2"))
        self.register_buffer("input_min", _state_tensor(input_state, "min"))
        self.register_buffer("input_max", _state_tensor(input_state, "max", default=1.0))
        self.register_buffer("target_mean", _state_tensor(target_state, "mean"))
        self.register_buffer("target_m2", _state_tensor(target_state, "M2"))
        self.register_buffer("target_min", _state_tensor(target_state, "min"))
        self.register_buffer("target_max", _state_tensor(target_state, "max", default=1.0))
        self.input_count = max(int((input_state or {}).get("n", 0)), 1)
        self.target_count = max(int((target_state or {}).get("n", 0)), 1)
        self.context_builder = getattr(core, "context_builder", None)
        self.context_params = dict(getattr(core, "context_builder_params", {}))

    def registered_layer(self, runtime: InferenceRuntime, name: str) -> torch.nn.Module:
        """Resolve a registered alias against this adapter's independent model clone."""

        path = EXPLAINABLE_LAYERS[_backbone(runtime)][name]
        return self.model.get_submodule(path)

    @staticmethod
    def _scale(
        values: torch.Tensor,
        kind: str | None,
        mean: torch.Tensor,
        m2: torch.Tensor,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        if kind is None:
            return values
        if kind == "standard":
            variance = m2 / float(count)
            scale = torch.sqrt(torch.clamp(variance, min=1e-8))
            return (values - mean) / scale
        if kind == "minmax":
            width = maximum - minimum
            scale = torch.where(width > 1e-8, width, torch.ones_like(width))
            return (values - minimum) / scale
        raise RuntimeError(f"Unsupported differentiable scaler kind {kind!r}.")

    def _prepare(self, flattened: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = int(flattened.shape[0])
        raw = flattened.reshape((batch, *self.input_shape))
        if not self.preserve_shape:
            prepared = raw.reshape(batch, -1)
            prepared = self._scale(
                prepared,
                self.input_scaler_kind,
                self.input_mean,
                self.input_m2,
                self.input_min,
                self.input_max,
                self.input_count,
            )
            return prepared, prepared

        channel_axis = -1 if self.data_format == "channels_last" else 1
        channel_first = torch.movedim(raw, channel_axis, 1) if channel_axis == -1 else raw
        channels = int(channel_first.shape[1])
        position_major = (
            channel_first.reshape(batch, channels, -1).transpose(1, 2).reshape(-1, channels)
        )
        position_major = self._scale(
            position_major,
            self.input_scaler_kind,
            self.input_mean,
            self.input_m2,
            self.input_min,
            self.input_max,
            self.input_count,
        )
        restored = (
            position_major.reshape(batch, -1, channels).transpose(1, 2).reshape(channel_first.shape)
        )
        if self.use_channel_first:
            prepared = restored
        else:
            prepared = position_major.reshape(batch, -1)
        raw_order = (
            torch.movedim(restored, 1, -1) if self.data_format == "channels_last" else restored
        )
        return prepared, raw_order.reshape(batch, -1)

    def _context(self, flattened: torch.Tensor) -> torch.Tensor | None:
        if self.context_builder is None:
            return None
        if self.context_builder != "cosine":
            raise RuntimeError("Only the built-in cosine context builder is differentiable.")
        frequencies = self.context_params.get("frequencies")
        frequency_values: tuple[float, ...]
        if frequencies is None:
            frequency_values = (1.0,)
        elif isinstance(frequencies, int):
            frequency_values = tuple(float(index) for index in range(1, frequencies + 1))
        else:
            frequency_values = tuple(float(value) for value in frequencies)
        basis = flattened
        if bool(self.context_params.get("normalise_input", False)):
            basis = basis / torch.clamp(
                torch.linalg.vector_norm(basis, dim=1, keepdim=True),
                min=1e-6,
            )
        features: list[torch.Tensor] = []
        for frequency in frequency_values:
            scaled = basis * frequency
            if bool(self.context_params.get("include_sin", True)):
                features.append(torch.sin(scaled))
            if bool(self.context_params.get("include_cos", True)):
                features.append(torch.cos(scaled))
        return torch.cat(features, dim=1)

    def _inverse_targets(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.target_scaler_kind is None:
            return outputs
        if self.target_scaler_kind == "standard":
            variance = self.target_m2 / float(self.target_count)
            scale = torch.sqrt(torch.clamp(variance, min=1e-8))
            return outputs * scale + self.target_mean
        if self.target_scaler_kind == "minmax":
            width = self.target_max - self.target_min
            scale = torch.where(width > 1e-8, width, torch.ones_like(width))
            return outputs * scale + self.target_min
        raise RuntimeError(f"Unsupported differentiable target scaler {self.target_scaler_kind!r}.")

    def _task_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        values = outputs.reshape(int(outputs.shape[0]), -1)
        if self.task == "regression":
            values = self._inverse_targets(values)
        elif self.output_kind == "probability":
            if self.task == "binary":
                positive = torch.sigmoid(values)
                values = torch.cat((1.0 - positive, positive), dim=1)
            elif self.task == "multiclass":
                values = torch.softmax(values, dim=1)
            else:
                values = torch.sigmoid(values)
        elif self.output_kind != "logit":
            raise RuntimeError(f"Classification gradient adapter cannot emit {self.output_kind!r}.")
        return values[:, list(self.output_indices)]

    def forward(self, flattened: torch.Tensor) -> torch.Tensor:
        if flattened.ndim != 2 or int(flattened.shape[1]) != math.prod(self.input_shape):
            raise ValueError(
                f"Expected flattened raw inputs shaped (batch, {math.prod(self.input_shape)})."
            )
        prepared, context_source = self._prepare(flattened.to(dtype=torch.float32))
        context = self._context(context_source)
        outputs = self.model(prepared, context) if context is not None else self.model(prepared)
        return self._task_outputs(outputs)


__all__ = [
    "DifferentiableInferenceAdapter",
    "EXPLAINABLE_LAYERS",
    "differentiable_inference_reasons",
    "list_explainable_layers",
    "register_explainable_layer",
]
