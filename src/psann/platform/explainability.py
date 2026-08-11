"""Optional raw-input SHAP explainability for fitted and deployed PSANN models."""

from __future__ import annotations

import math
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .._sklearn.classifier import PSANNClassifier
from .explain_contracts import (
    BackgroundPolicyError,
    BackgroundSummary,
    ExplainabilityError,
    ExplainabilityUnavailableError,
    ExplainerConfig,
    ExplanationCapabilityError,
    ExplanationDriftSummary,
    ExplanationResult,
    FeatureGroup,
    _atomic_json,
    load_explainer_config,
    save_explainer_config,
    summarize_background,
)
from .explain_groups import (
    domain_linkage,
    feature_groups,
    feature_names,
    resolved_group_strategy,
)
from .explain_torch import (
    DifferentiableInferenceAdapter,
    differentiable_inference_reasons,
)
from .inference import (
    InferenceRuntime,
    _apply_feature_policy,
    _artifact_fields,
    _core_estimator,
    _normalise_batch,
    _output_names,
    create_inference_runtime,
)


def _import_shap() -> Any:
    try:
        import shap
    except ImportError as exc:
        raise ExplainabilityUnavailableError(
            "SHAP explainability is optional. Install it with `pip install psann[explain]`."
        ) from exc
    return shap


def _as_config(value: ExplainerConfig | Mapping[str, Any] | None) -> ExplainerConfig:
    if isinstance(value, ExplainerConfig):
        return value
    return ExplainerConfig.from_dict(value) if value is not None else ExplainerConfig()


def _background_summary(
    runtime: InferenceRuntime,
    config: ExplainerConfig,
    *,
    background: Any | None,
    reference_data: Any | None,
    summary: BackgroundSummary | None,
) -> tuple[BackgroundSummary, str]:
    supplied = sum(value is not None for value in (background, reference_data, summary))
    if supplied != 1:
        raise BackgroundPolicyError(
            "Provide exactly one explicit background source: background, reference_data, "
            "or summary. Training data is never selected implicitly."
        )
    core = _core_estimator(runtime.model)
    input_shape = tuple(int(item) for item in core.input_shape_)
    names = feature_names(runtime, input_shape)
    if summary is not None:
        if summary.input_shape != input_shape:
            raise BackgroundPolicyError(
                f"Background summary shape {summary.input_shape!r} does not match "
                f"model input shape {input_shape!r}."
            )
        if int(summary.values.shape[0]) > config.max_background_samples:
            raise BackgroundPolicyError("Background summary exceeds max_background_samples.")
        return summary, "persisted_summary"
    source = reference_data if reference_data is not None else background
    ordered = _apply_feature_policy(runtime.model, source, runtime.config.feature_policy)
    values = np.asarray(_normalise_batch(runtime.model, ordered), dtype=np.float32)
    if tuple(values.shape[1:]) != input_shape:
        raise BackgroundPolicyError(
            f"Background shape {values.shape[1:]!r} does not match {input_shape!r}."
        )
    if reference_data is None and int(values.shape[0]) > config.max_background_samples:
        raise BackgroundPolicyError(
            "Explicit background exceeds max_background_samples; use reference_data to "
            "request deterministic sampling."
        )
    selected = summarize_background(
        values,
        input_shape=input_shape,
        feature_names=names,
        data_format=str(core.data_format),
        max_samples=config.background_size if reference_data is not None else values.shape[0],
        seed=config.seed,
    )
    return selected, "sampled_reference" if reference_data is not None else "explicit"


def _output_contract(
    runtime: InferenceRuntime,
    config: ExplainerConfig,
) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    task = runtime.task
    model = runtime.model
    core = _core_estimator(model)
    if task == "regression":
        if config.output_kind not in {"auto", "prediction"}:
            raise ValueError("Regression explanations require output_kind='prediction'.")
        kind = "prediction"
        names = _output_names(model)
        width = int(getattr(core, "_output_dim_", 0) or getattr(core, "_primary_dim_", 1))
        if len(names) != width:
            names = tuple(f"output_{index}" for index in range(width))
    else:
        if config.output_kind == "prediction":
            raise ValueError("Class labels are not numeric and cannot be explained.")
        kind = "probability" if config.output_kind == "auto" else config.output_kind
        if kind not in {"probability", "logit"}:
            raise ValueError("Classification explanations require probability or logit output.")
        assert isinstance(model, PSANNClassifier)
        classes = tuple(str(item) for item in model.classes_.tolist())
        if task == "binary":
            names = classes if kind == "probability" else (classes[1],)
        else:
            names = tuple(str(item) for item in model.output_names_.tolist())
    if config.output is None:
        indices = tuple(range(len(names)))
    elif isinstance(config.output, int):
        if config.output >= len(names):
            raise ValueError(f"output index {config.output} is outside the {len(names)} outputs.")
        indices = (config.output,)
    else:
        requested = str(config.output)
        if requested not in names:
            raise ValueError(f"Unknown output {requested!r}; available={list(names)!r}.")
        indices = (names.index(requested),)
    return kind, tuple(names[index] for index in indices), indices


class PSANNExplainer:
    """Bounded SHAP explainer over one schema-aware PSANN inference runtime."""

    def __init__(
        self,
        runtime: InferenceRuntime,
        background_summary: BackgroundSummary,
        *,
        background_policy: str,
        config: ExplainerConfig,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.background_summary = background_summary
        self.background_policy = background_policy
        core = _core_estimator(runtime.model)
        if (
            getattr(core, "_context_dim_", None) not in {None, 0}
            and getattr(core, "context_builder", None) is None
        ):
            raise ExplanationCapabilityError(
                "This model requires explicit context, which is not part of the Phase 6 "
                "raw-input explanation game."
            )
        self.input_shape = background_summary.input_shape
        self.feature_names = feature_names(runtime, self.input_shape)
        self.group_strategy = resolved_group_strategy(
            runtime,
            self.input_shape,
            config.group_strategy,
        )
        self.feature_groups = feature_groups(
            runtime,
            self.input_shape,
            self.group_strategy,
            self.feature_names,
        )
        self.output_kind, self.output_names, self.output_indices = _output_contract(
            runtime,
            config,
        )
        self.requested_algorithm = config.algorithm
        self.fallback_reason: str | None = None
        self.algorithm = self._resolve_algorithm()
        self.masker_kind = self._resolve_masker()
        self._lock = threading.RLock()
        self._shap = _import_shap()
        self._background_flat = background_summary.values.reshape(
            int(background_summary.values.shape[0]),
            -1,
        )
        self._inference_runtime = self._configured_runtime()
        self._torch_adapter: DifferentiableInferenceAdapter | None = None
        self._explainer = self._build_explainer()

    def _configured_runtime(self) -> InferenceRuntime:
        inference_config = replace(
            self.runtime.config,
            classification_output="probability",
            return_logits=self.output_kind == "logit",
        )
        return create_inference_runtime(self.runtime.model, config=inference_config)

    def _resolve_algorithm(self) -> str:
        if self.config.algorithm in {"gradient", "deep"}:
            reasons = differentiable_inference_reasons(
                self.runtime,
                algorithm=self.config.algorithm,
                layer=self.config.layer,
            )
            if reasons:
                reason = "; ".join(reasons)
                if self.config.fallback == "error" or self.config.layer is not None:
                    raise ExplanationCapabilityError(reason)
                self.fallback_reason = reason
                return "partition" if len(self.input_shape) > 1 else "permutation"
            return self.config.algorithm
        if self.config.algorithm == "partition":
            return "partition"
        if self.config.algorithm == "permutation":
            return "permutation"
        return "partition" if len(self.input_shape) > 1 else "permutation"

    def _resolve_masker(self) -> str:
        if self.algorithm in {"gradient", "deep"}:
            return "none"
        if self.config.masker != "auto":
            masker = self.config.masker
        elif len(self.input_shape) > 1:
            masker = "domain"
        else:
            masker = "independent"
        if self.algorithm == "partition" and masker == "independent":
            masker = "partition"
        if self.algorithm == "permutation" and masker == "partition":
            return masker
        return masker

    def _predict_flat(self, values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        raw = array.reshape((int(array.shape[0]), *self.input_shape))
        result = self._inference_runtime.predict(
            raw,
            return_logits=self.output_kind == "logit",
        )
        outputs = np.asarray(result.values, dtype=np.float32)
        if outputs.ndim == 1:
            outputs = outputs.reshape(-1, 1)
        else:
            outputs = outputs.reshape(int(outputs.shape[0]), -1)
        return outputs[:, list(self.output_indices)]

    def _build_masker(self) -> Any:
        if self.masker_kind == "independent":
            return self._shap.maskers.Independent(
                self._background_flat,
                max_samples=self.config.max_background_samples,
            )
        if self.masker_kind == "partition":
            return self._shap.maskers.Partition(
                self._background_flat,
                max_samples=self.config.max_background_samples,
                clustering="correlation",
            )
        if self.masker_kind == "domain":
            linkage = domain_linkage(self.feature_groups, self._background_flat.shape[1])
            return self._shap.maskers.Partition(
                self._background_flat,
                max_samples=self.config.max_background_samples,
                clustering=linkage,
            )
        raise ValueError(f"Unsupported resolved masker {self.masker_kind!r}.")

    def _build_explainer(self) -> Any:
        if self.algorithm in {"permutation", "partition"}:
            if (
                self.algorithm == "permutation"
                and self.config.max_evaluations < 2 * self._background_flat.shape[1] + 1
            ):
                raise ExplanationCapabilityError(
                    "Permutation explanations require max_evaluations >= "
                    f"{2 * self._background_flat.shape[1] + 1} for this input."
                )
            return self._shap.Explainer(
                self._predict_flat,
                self._build_masker(),
                algorithm=self.algorithm,
                output_names=list(self.output_names),
                feature_names=list(self.feature_names),
                seed=self.config.seed,
            )
        adapter = DifferentiableInferenceAdapter(
            self.runtime,
            output_kind=self.output_kind,
            output_indices=self.output_indices,
        )
        self._torch_adapter = adapter
        background = torch.as_tensor(
            self._background_flat,
            device=self.runtime.device,
            dtype=torch.float32,
        )
        model: Any = adapter
        if self.config.layer is not None:
            model = (adapter, adapter.registered_layer(self.runtime, self.config.layer))
        if self.algorithm == "deep":
            return self._shap.DeepExplainer(model, background)
        return self._shap.GradientExplainer(
            model,
            background,
            batch_size=self.config.batch_size,
            local_smoothing=self.config.local_smoothing,
        )

    def _normalise_values(self, value: Any, sample_count: int) -> np.ndarray:
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, list):
            array = np.stack([np.asarray(item) for item in value], axis=-1)
        else:
            array = np.asarray(value)
        if array.shape[0] != sample_count:
            raise ExplainabilityError("SHAP returned an unexpected sample dimension.")
        if self.config.layer is None:
            feature_count = math.prod(self.input_shape)
            if array.ndim == 2:
                array = array[..., None]
            if array.shape[1] != feature_count:
                raise ExplainabilityError(
                    f"SHAP returned {array.shape[1]} features; expected {feature_count}."
                )
            if array.shape[-1] != len(self.output_names):
                if len(self.output_names) == 1:
                    array = array.reshape(sample_count, feature_count, 1)
                else:
                    raise ExplainabilityError("SHAP returned an unexpected output dimension.")
            return array.reshape((sample_count, *self.input_shape, len(self.output_names)))
        if array.ndim == 2:
            array = array[..., None]
        return array

    def _normalise_base_values(self, value: Any, sample_count: int) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        output_count = len(self.output_names)
        if array.ndim == 0:
            array = np.full((sample_count, output_count), float(array), dtype=np.float32)
        elif array.ndim == 1:
            if array.shape[0] == output_count:
                array = np.broadcast_to(array.reshape(1, -1), (sample_count, output_count))
            elif output_count == 1 and array.shape[0] == sample_count:
                array = array.reshape(-1, 1)
            else:
                raise ExplainabilityError("SHAP returned unexpected base-value dimensions.")
        elif array.shape[0] == 1 and sample_count > 1:
            array = np.broadcast_to(array, (sample_count, *array.shape[1:]))
        return np.asarray(array, dtype=np.float32).reshape(sample_count, output_count)

    def _explain_model_agnostic(self, flattened: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        numpy_state = np.random.get_state()
        try:
            np.random.seed(self.config.seed)
            explanation = self._explainer(
                flattened,
                max_evals=self.config.max_evaluations,
                batch_size=self.config.batch_size,
                silent=True,
            )
        finally:
            np.random.set_state(numpy_state)
        values = self._normalise_values(explanation.values, flattened.shape[0])
        bases = self._normalise_base_values(explanation.base_values, flattened.shape[0])
        return values, bases

    def _explain_gradient(self, flattened: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._torch_adapter is not None
        tensor = torch.as_tensor(
            flattened,
            device=self.runtime.device,
            dtype=torch.float32,
        )
        numpy_state = np.random.get_state()
        try:
            np.random.seed(self.config.seed)
            with torch.random.fork_rng(
                devices=[self.runtime.device] if self.runtime.device.type == "cuda" else []
            ):
                torch.manual_seed(self.config.seed)
                if self.algorithm == "deep":
                    raw_values = self._explainer.shap_values(
                        tensor,
                        check_additivity=False,
                    )
                    raw_bases = self._explainer.expected_value
                else:
                    raw_values = self._explainer.shap_values(
                        tensor,
                        nsamples=self.config.gradient_samples,
                        rseed=self.config.seed,
                    )
                    background = torch.as_tensor(
                        self._background_flat,
                        device=self.runtime.device,
                        dtype=torch.float32,
                    )
                    with torch.no_grad():
                        raw_bases = (
                            self._torch_adapter(background).mean(dim=0).detach().cpu().numpy()
                        )
        finally:
            np.random.set_state(numpy_state)
        values = self._normalise_values(raw_values, flattened.shape[0])
        bases = self._normalise_base_values(raw_bases, flattened.shape[0])
        return values, bases

    def explain(self, inputs: Any) -> ExplanationResult:
        """Explain bounded raw-input samples without changing shared model state."""

        ordered = _apply_feature_policy(
            self.runtime.model,
            inputs,
            self.runtime.config.feature_policy,
        )
        raw = np.asarray(_normalise_batch(self.runtime.model, ordered), dtype=np.float32)
        if tuple(raw.shape[1:]) != self.input_shape:
            raise ValueError(
                f"Explanation input shape {raw.shape[1:]!r} does not match {self.input_shape!r}."
            )
        if int(raw.shape[0]) > self.config.max_explanation_samples:
            raise ExplanationCapabilityError(
                f"Explanation request has {raw.shape[0]} samples; the configured maximum "
                f"is {self.config.max_explanation_samples}."
            )
        if not np.isfinite(raw).all():
            raise ValueError("Explanation inputs must contain only finite numeric values.")
        flattened = raw.reshape(int(raw.shape[0]), -1)
        with self._lock:
            if self.algorithm in {"gradient", "deep"}:
                values, bases = self._explain_gradient(flattened)
            else:
                values, bases = self._explain_model_agnostic(flattened)
        outputs = self._predict_flat(flattened)
        attribution_axes = tuple(range(1, values.ndim - 1))
        reconstructed = bases + values.sum(axis=attribution_axes)
        additivity_error = float(np.max(np.abs(reconstructed - outputs)))
        if self.config.layer is None:
            result_feature_names = self.feature_names
            result_groups = self.feature_groups
            explanation_data: Any = raw
            explanation_feature_names: Any = (
                list(self.feature_names)
                if len(self.input_shape) == 1
                else np.asarray(self.feature_names, dtype=object).reshape(self.input_shape).tolist()
            )
        else:
            attribution_shape = tuple(int(item) for item in values.shape[1:-1])
            result_feature_names = tuple(
                f"{self.config.layer}[" + ",".join(str(index) for index in coordinate) + "]"
                for coordinate in np.ndindex(attribution_shape)
            )
            result_groups = tuple(
                FeatureGroup(name=name, indices=(index,), strategy="layer")
                for index, name in enumerate(result_feature_names)
            )
            explanation_data = None
            explanation_feature_names = (
                list(result_feature_names)
                if len(attribution_shape) == 1
                else np.asarray(result_feature_names, dtype=object)
                .reshape(attribution_shape)
                .tolist()
            )
        shap_explanation = self._shap.Explanation(
            values=values,
            base_values=bases,
            data=explanation_data,
            feature_names=explanation_feature_names,
            output_names=list(self.output_names),
        )
        artifact_version, model_id, run_id = _artifact_fields(self.runtime.model)
        return ExplanationResult(
            explanation=shap_explanation,
            feature_groups=result_groups,
            task=self.runtime.task,
            output_names=self.output_names,
            artifact_version=artifact_version,
            model_id=model_id,
            run_id=run_id,
            metadata={
                "additivity_error": additivity_error,
                "algorithm": self.algorithm,
                "requested_algorithm": self.requested_algorithm,
                "fallback_reason": self.fallback_reason,
                "masker": self.masker_kind,
                "group_strategy": self.group_strategy,
                "background_policy": self.background_policy,
                "background_samples": int(self._background_flat.shape[0]),
                "batch_size": self.config.batch_size,
                "max_evaluations": self.config.max_evaluations,
                "gradient_samples": (
                    self.config.gradient_samples if self.algorithm in {"gradient", "deep"} else None
                ),
                "input_shape": list(self.input_shape),
                "data_format": str(_core_estimator(self.runtime.model).data_format),
                "feature_names": list(result_feature_names),
                "output_kind": self.output_kind,
                "layer": self.config.layer,
                "state_policy": (
                    "frozen_clone"
                    if self.algorithm in {"gradient", "deep"}
                    else "stateless_runtime"
                ),
            },
        )

    __call__ = explain


def make_explainer(
    model: Any,
    *,
    background: Any | None = None,
    reference_data: Any | None = None,
    summary: BackgroundSummary | None = None,
    config: ExplainerConfig | Mapping[str, Any] | None = None,
) -> PSANNExplainer:
    """Create a bounded explainer from an explicit background source."""

    runtime = model if isinstance(model, InferenceRuntime) else create_inference_runtime(model)
    resolved = _as_config(config)
    background_summary, policy = _background_summary(
        runtime,
        resolved,
        background=background,
        reference_data=reference_data,
        summary=summary,
    )
    return PSANNExplainer(
        runtime,
        background_summary,
        background_policy=policy,
        config=resolved,
    )


def explain(
    model: Any,
    inputs: Any,
    *,
    background: Any | None = None,
    reference_data: Any | None = None,
    summary: BackgroundSummary | None = None,
    config: ExplainerConfig | Mapping[str, Any] | None = None,
) -> ExplanationResult:
    """Create and execute a one-shot raw-input explanation."""

    return make_explainer(
        model,
        background=background,
        reference_data=reference_data,
        summary=summary,
        config=config,
    ).explain(inputs)


def _importance(result: ExplanationResult) -> np.ndarray:
    values = np.asarray(result.values, dtype=np.float64)
    if values.ndim < 3:
        raise ValueError("Explanation values must contain sample, feature, and output axes.")
    return np.abs(values).mean(axis=(0, values.ndim - 1)).reshape(-1)


def summarize_explanation_drift(
    reference: ExplanationResult,
    current: ExplanationResult,
) -> ExplanationDriftSummary:
    """Compare aggregate absolute attribution magnitudes across two cohorts."""

    reference_names = tuple(str(item) for item in reference.metadata["feature_names"])
    current_names = tuple(str(item) for item in current.metadata["feature_names"])
    if reference_names != current_names or reference.output_names != current.output_names:
        raise ValueError("Explanation drift inputs must share feature and output contracts.")
    reference_importance = _importance(reference)
    current_importance = _importance(current)
    shift = np.abs(current_importance - reference_importance)
    denominator = float(np.linalg.norm(reference_importance) * np.linalg.norm(current_importance))
    cosine = (
        1.0
        if denominator == 0.0 and np.array_equal(reference_importance, current_importance)
        else (
            0.0
            if denominator == 0.0
            else float(np.dot(reference_importance, current_importance) / denominator)
        )
    )
    return ExplanationDriftSummary(
        feature_names=reference_names,
        reference_importance=tuple(float(item) for item in reference_importance),
        current_importance=tuple(float(item) for item in current_importance),
        absolute_shift=tuple(float(item) for item in shift),
        mean_absolute_shift=float(shift.mean()),
        maximum_absolute_shift=float(shift.max(initial=0.0)),
        cosine_similarity=cosine,
    )


def write_explanation_report(
    result: ExplanationResult,
    path: str | os.PathLike[str],
) -> Path:
    """Write an aggregate offline report without raw inputs or row-level attributions."""

    importance = _importance(result)
    feature_names = tuple(str(item) for item in result.metadata["feature_names"])
    payload = {
        "format": "psann.explanation-summary",
        "version": "1.0",
        "task": result.task,
        "output_names": list(result.output_names),
        "artifact_version": result.artifact_version,
        "model_id": result.model_id,
        "run_id": result.run_id,
        "metadata": dict(result.metadata),
        "mean_absolute_attribution": {
            name: float(value) for name, value in zip(feature_names, importance)
        },
        "feature_groups": [group.to_dict() for group in result.feature_groups],
        "contains_raw_inputs": False,
        "contains_row_level_attributions": False,
    }
    destination = Path(path)
    _atomic_json(destination, payload)
    return destination


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
    "PSANNExplainer",
    "explain",
    "load_explainer_config",
    "make_explainer",
    "save_explainer_config",
    "summarize_background",
    "summarize_explanation_drift",
    "write_explanation_report",
]
