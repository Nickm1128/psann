"""Sklearn-compatible classifier using the shared PSANN estimator training path."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

try:
    from sklearn.exceptions import NotFittedError
except Exception:  # pragma: no cover - sklearn is optional
    NotFittedError = RuntimeError  # type: ignore[misc,assignment]

from ..legacy import LEGACY_CHECKPOINT_MESSAGE, LegacyCheckpointWarning
from ..platform.registry import BACKBONES
from ..platform.specs import DataSchema, TaskSpec
from ..platform.tasks import FittedTaskAdapter, create_task_adapter
from .shared import BaseEstimator, ClassifierMixin


class PSANNClassifier(ClassifierMixin, BaseEstimator):
    """Task-aware classifier over any compatible registered PSANN backbone.

    The wrapper owns label encoding and probability semantics. Its internal estimator
    is one of the existing regressors and therefore uses the same model builders,
    training events, fallback policies, and resumable training core.
    """

    def __init__(
        self,
        *,
        backbone: str = "psann_mlp",
        task: str = "auto",
        threshold: float | tuple[float, ...] = 0.5,
        class_names: tuple[Any, ...] | None = None,
        positive_label: Any | None = None,
        estimator_params: Mapping[str, Any] | None = None,
        schema: DataSchema | Mapping[str, Any] | None = None,
    ) -> None:
        self.backbone = backbone
        self.task = task
        self.threshold = threshold
        self.class_names = class_names
        self.positive_label = positive_label
        self.estimator_params = estimator_params
        self.schema = schema

    def _task_spec(self, targets: Any) -> TaskSpec:
        kind = str(self.task).strip().lower()
        target_array = np.asarray(targets)
        if kind == "auto":
            kind = (
                "multilabel"
                if target_array.ndim == 2 and target_array.shape[1] > 1
                else ("binary" if len(np.unique(target_array)) == 2 else "multiclass")
            )
        if kind not in {"binary", "multiclass", "multilabel"}:
            raise ValueError(
                "PSANNClassifier task must be auto, binary, multiclass, or multilabel; "
                f"received {self.task!r}."
            )
        threshold = self.threshold
        if isinstance(threshold, list):
            threshold = tuple(float(item) for item in threshold)
        class_names = tuple(self.class_names or ())
        if kind == "multilabel" and not class_names:
            class_names = self._schema_spec().output_names
        return TaskSpec(
            kind=kind,  # type: ignore[arg-type]
            class_names=class_names,
            threshold=threshold,
            positive_label=self.positive_label,
        )

    def _schema_spec(self) -> DataSchema:
        if self.schema is None:
            return DataSchema()
        if isinstance(self.schema, DataSchema):
            return self.schema
        return DataSchema.from_dict(self.schema)

    def _require_fitted(self) -> None:
        if not hasattr(self, "estimator_") or not hasattr(self, "task_adapter_"):
            raise NotFittedError("PSANNClassifier is not fitted; call fit before prediction.")

    def fit(self, X: Any, y: Any, **fit_params: Any) -> "PSANNClassifier":
        spec = self._task_spec(y)
        adapter = create_task_adapter(spec)
        encoded = adapter.fit_targets(y)
        fitted_spec = TaskSpec(
            kind=spec.kind,
            class_names=tuple(adapter.classes),
            threshold=spec.threshold,
            positive_label=spec.positive_label,
        )
        adapter.spec = fitted_spec
        registration = BACKBONES.resolve(self.backbone)
        if spec.kind not in registration.supported_tasks:
            raise ValueError(
                f"Backbone {registration.identifier!r} does not support task {spec.kind!r}."
            )
        input_rank = max(0, np.asarray(X).ndim - 1)
        if input_rank not in registration.input_ranks:
            raise ValueError(
                f"Backbone {registration.identifier!r} expects non-batch input rank in "
                f"{sorted(registration.input_ranks)!r}; received {input_rank}."
            )

        parameters = dict(self.estimator_params or {})
        runtime_overrides = {
            name: parameters.pop(name)
            for name in ("amp", "amp_dtype", "compile")
            if name in parameters
        }
        if parameters.get("target_scaler") is not None:
            raise ValueError("Classification does not support target_scaler.")
        output_width = int(encoded.shape[1]) if encoded.ndim == 2 else 1
        configured_output = parameters.get("output_shape")
        if configured_output is not None and tuple(configured_output) != (output_width,):
            raise ValueError(
                f"Configured output_shape {configured_output!r} does not match task output "
                f"width {output_width}."
            )
        parameters["output_shape"] = (output_width,)
        parameters["target_scaler"] = None
        estimator = registration.factory(parameters)
        for name, value in runtime_overrides.items():
            setattr(estimator, name, value)
        estimator._platform_data_schema_ = self._schema_spec()
        estimator._platform_task_spec_ = fitted_spec

        validation_data = fit_params.get("validation_data")
        if validation_data is not None:
            values = tuple(validation_data)
            if len(values) not in {2, 3}:
                raise ValueError("validation_data must contain (X, y) or (X, y, context).")
            encoded_validation = adapter.transform_targets(values[1])
            fit_params["validation_data"] = (
                (values[0], encoded_validation)
                if len(values) == 2
                else (values[0], encoded_validation, values[2])
            )

        supplied_metrics = fit_params.get("metrics")
        metrics = dict(adapter.training_metrics())
        if supplied_metrics is not None:
            metrics.update(dict(supplied_metrics))
        fit_params["metrics"] = metrics

        original_loss = estimator.loss
        original_output_shape = estimator.output_shape
        estimator.loss = adapter.loss()
        estimator.output_shape = (output_width,)
        try:
            estimator.fit(X, encoded, **fit_params)
        finally:
            estimator.loss = original_loss
            estimator.output_shape = original_output_shape

        self.estimator_ = estimator
        self.task_adapter_: FittedTaskAdapter = adapter
        self.task_spec_ = fitted_spec
        self.classes_ = np.asarray(adapter.classes, dtype=object)
        self.n_outputs_ = output_width
        self.output_names_ = np.asarray(adapter.output_names, dtype=object)
        self.n_features_in_ = estimator.n_features_in_
        if hasattr(estimator, "feature_names_in_"):
            self.feature_names_in_ = estimator.feature_names_in_.copy()
        self.input_shape_ = tuple(estimator.input_shape_)
        self.history_ = list(getattr(estimator, "history_", ()))
        self.training_events_ = list(getattr(estimator, "training_events_", ()))
        self.training_metadata_ = dict(getattr(estimator, "training_metadata_", {}))
        self.preprocessing_contract_ = dict(getattr(estimator, "preprocessing_contract_", {}))
        self.model_ = estimator.model_
        return self

    def decision_function(self, X: Any, *, context: Any = None) -> np.ndarray:
        """Return raw logits from the fitted backbone."""

        self._require_fitted()
        output = self.estimator_.predict(X, context=context)
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def predict_proba(self, X: Any, *, context: Any = None) -> np.ndarray:
        """Return positive-class, class-column, or per-label probabilities."""

        self._require_fitted()
        return self.task_adapter_.probabilities(self.decision_function(X, context=context))

    def predict(self, X: Any, *, context: Any = None) -> np.ndarray:
        self._require_fitted()
        return self.task_adapter_.predictions_from_outputs(
            self.decision_function(X, context=context)
        )

    def score(self, X: Any, y: Any, *, context: Any = None) -> float:
        prediction = self.predict(X, context=context)
        truth = np.asarray(y)
        if self.task_spec_.kind == "multilabel":
            expected = self.task_adapter_.transform_targets(truth).astype(bool)
            return float((np.asarray(prediction, dtype=bool) == expected).all(axis=1).mean())
        return float((prediction == truth).mean())

    def set_feature_schema_policy(self, policy: str) -> "PSANNClassifier":
        """Change the fitted named-feature policy and return ``self``."""

        self._require_fitted()
        self.estimator_.set_feature_schema_policy(policy)
        return self

    def save(self, path: str | Path) -> None:
        """Save a deprecated trusted snapshot; prefer ``TrainingRun.export``."""

        warnings.warn(
            LEGACY_CHECKPOINT_MESSAGE,
            LegacyCheckpointWarning,
            stacklevel=2,
        )
        self._require_fitted()
        torch.save(self, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "PSANNClassifier":
        """Load a deprecated trusted snapshot created by :meth:`save`."""

        warnings.warn(
            LEGACY_CHECKPOINT_MESSAGE,
            LegacyCheckpointWarning,
            stacklevel=2,
        )
        try:
            value = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            value = torch.load(path, map_location=map_location)
        if not isinstance(value, cls):
            raise TypeError(f"Snapshot contains {type(value).__name__}, not {cls.__name__}.")
        return value


__all__ = ["PSANNClassifier"]
