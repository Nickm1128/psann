"""Task-owned target validation, losses, probabilities, and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as functional

from .contracts import TaskKind
from .specs import TaskSpec

TensorMetric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _as_array(targets: Any) -> np.ndarray:
    array = np.asarray(targets)
    if array.ndim < 1:
        raise ValueError("Targets must include a batch dimension.")
    if array.shape[0] < 1:
        raise ValueError("Targets must contain at least one sample.")
    return array


def _require_numeric_finite(targets: Any, *, task: str) -> np.ndarray:
    try:
        array = np.asarray(targets, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{task} targets must be numeric.") from exc
    if np.isnan(array).any() or np.isinf(array).any():
        raise ValueError(f"{task} targets must contain only finite values.")
    return array


def _require_labels(targets: Any, *, task: str) -> np.ndarray:
    array = _as_array(targets)
    if array.ndim != 1:
        raise ValueError(f"{task} targets must be one-dimensional class labels.")
    for value in array.tolist():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            raise ValueError(f"{task} targets cannot contain missing or infinite labels.")
    return array


def _bce_logits(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return functional.binary_cross_entropy_with_logits(prediction, target)


def _cross_entropy_one_hot(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return functional.cross_entropy(prediction, target.argmax(dim=1))


def _regression_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).abs().mean()


def _regression_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean()


def _multiclass_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.argmax(dim=1) == target.argmax(dim=1)).float().mean()


def _threshold_accuracy(
    threshold: float | tuple[float, ...],
    *,
    subset: bool,
) -> TensorMetric:
    """Build a detached logit metric that follows the task's threshold policy."""

    def metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(threshold, tuple):
            threshold_value: float | torch.Tensor = torch.as_tensor(
                threshold,
                dtype=prediction.dtype,
                device=prediction.device,
            ).reshape(1, -1)
        else:
            threshold_value = float(threshold)
        labels = (torch.sigmoid(prediction) >= threshold_value).to(target.dtype)
        matches = labels == target
        return matches.all(dim=1).float().mean() if subset else matches.float().mean()

    return metric


@dataclass
class FittedTaskAdapter:
    """Runtime task adapter fitted to class/output metadata."""

    spec: TaskSpec
    classes: tuple[Any, ...] = ()
    output_names: tuple[str, ...] = ()

    @property
    def kind(self) -> TaskKind:
        return self.spec.kind

    def fit_targets(self, targets: Any) -> np.ndarray:
        if self.kind == "regression":
            array = _require_numeric_finite(targets, task="Regression")
            if array.ndim > 2:
                raise ValueError("Regression targets must be a vector or 2D output matrix.")
            if not self.output_names:
                width = 1 if array.ndim == 1 else int(array.shape[1])
                self.output_names = tuple(f"output_{index}" for index in range(width))
            return array

        if self.kind in {"binary", "multiclass"}:
            labels = _require_labels(targets, task=self.kind.capitalize())
            discovered = tuple(np.unique(labels).tolist())
            configured = tuple(self.spec.class_names)
            classes = configured or discovered
            if configured and set(configured) != set(discovered):
                raise ValueError(
                    f"Configured class_names {configured!r} do not match fitted labels "
                    f"{discovered!r}."
                )
            if self.kind == "binary":
                if len(classes) != 2:
                    raise ValueError(
                        f"Binary classification requires exactly 2 classes; found {len(classes)}."
                    )
                if self.spec.positive_label is not None:
                    if self.spec.positive_label not in classes:
                        raise ValueError(
                            f"positive_label {self.spec.positive_label!r} is not present in targets."
                        )
                    negative = next(item for item in classes if item != self.spec.positive_label)
                    classes = (negative, self.spec.positive_label)
                self.classes = classes
                self.output_names = (str(classes[1]),)
                return (labels == classes[1]).astype(np.float32).reshape(-1, 1)
            if len(classes) < 2:
                raise ValueError(
                    f"Multiclass classification requires at least 2 classes; found {len(classes)}."
                )
            self.classes = classes
            self.output_names = tuple(str(item) for item in classes)
            indices = {label: index for index, label in enumerate(classes)}
            encoded = np.zeros((labels.shape[0], len(classes)), dtype=np.float32)
            for row, label in enumerate(labels.tolist()):
                if label not in indices:
                    raise ValueError(f"Unknown class label {label!r}.")
                encoded[row, indices[label]] = 1.0
            return encoded

        array = _require_numeric_finite(targets, task="Multilabel")
        if array.ndim != 2 or array.shape[1] < 1:
            raise ValueError("Multilabel targets must be a non-empty 2D indicator matrix.")
        if not np.isin(array, (0.0, 1.0)).all():
            raise ValueError("Multilabel targets must contain only 0/1 indicators.")
        names = tuple(self.spec.class_names) or tuple(
            f"label_{index}" for index in range(array.shape[1])
        )
        if len(names) != array.shape[1]:
            raise ValueError(
                "Multilabel class_names must match the target width; "
                f"received {len(names)} names for {array.shape[1]} outputs."
            )
        thresholds = self.spec.threshold
        if isinstance(thresholds, tuple) and len(thresholds) != array.shape[1]:
            raise ValueError(
                "Multilabel threshold tuple must match the target width; "
                f"received {len(thresholds)} thresholds for {array.shape[1]} outputs."
            )
        self.classes = names
        self.output_names = tuple(str(item) for item in names)
        return array.astype(np.float32, copy=False)

    def transform_targets(self, targets: Any) -> np.ndarray:
        if self.kind == "regression":
            return _require_numeric_finite(targets, task="Regression")
        if not self.classes and self.kind != "regression":
            raise RuntimeError("The task adapter must be fit before transforming targets.")
        if self.kind == "binary":
            labels = _require_labels(targets, task="Binary")
            unknown = set(np.unique(labels).tolist()) - set(self.classes)
            if unknown:
                raise ValueError(
                    f"Validation targets contain unknown classes: {sorted(unknown)!r}."
                )
            return (labels == self.classes[1]).astype(np.float32).reshape(-1, 1)
        if self.kind == "multiclass":
            labels = _require_labels(targets, task="Multiclass")
            indices = {label: index for index, label in enumerate(self.classes)}
            encoded = np.zeros((labels.shape[0], len(self.classes)), dtype=np.float32)
            for row, label in enumerate(labels.tolist()):
                if label not in indices:
                    raise ValueError(f"Validation targets contain unknown class {label!r}.")
                encoded[row, indices[label]] = 1.0
            return encoded
        array = _require_numeric_finite(targets, task="Multilabel")
        if array.ndim != 2 or array.shape[1] != len(self.classes):
            raise ValueError(f"Multilabel targets must have shape (samples, {len(self.classes)}).")
        if not np.isin(array, (0.0, 1.0)).all():
            raise ValueError("Multilabel targets must contain only 0/1 indicators.")
        return array

    def loss(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | str:
        if self.kind == "regression":
            return "mse"
        if self.kind == "multiclass":
            return _cross_entropy_one_hot
        return _bce_logits

    def probabilities(self, outputs: Any) -> np.ndarray:
        logits = np.asarray(outputs, dtype=np.float32)
        if self.kind == "regression":
            raise AttributeError("Regression tasks do not define class probabilities.")
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        if self.kind in {"binary", "multilabel"}:
            clipped = np.clip(logits, -80.0, 80.0)
            positive = 1.0 / (1.0 + np.exp(-clipped))
            if self.kind == "binary":
                return np.concatenate((1.0 - positive, positive), axis=1)
            return positive
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)

    def predictions_from_outputs(self, outputs: Any) -> np.ndarray:
        if self.kind == "regression":
            return np.asarray(outputs)
        probabilities = self.probabilities(outputs)
        if self.kind == "binary":
            threshold_value = self.spec.threshold
            if isinstance(threshold_value, tuple):
                raise RuntimeError("Binary task threshold cannot be a tuple.")
            threshold = float(threshold_value)
            indices = (probabilities[:, 1] >= threshold).astype(np.int64)
            return np.asarray(self.classes, dtype=object)[indices]
        if self.kind == "multiclass":
            indices = probabilities.argmax(axis=1)
            return np.asarray(self.classes, dtype=object)[indices]
        thresholds = self.spec.threshold
        if isinstance(thresholds, tuple):
            threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
        else:
            threshold_array = float(thresholds)
        return probabilities >= threshold_array

    def training_metrics(self) -> Mapping[str, TensorMetric]:
        if self.kind == "regression":
            return {"mae": _regression_mae, "mse": _regression_mse}
        if self.kind == "binary":
            return {"accuracy": _threshold_accuracy(self.spec.threshold, subset=False)}
        if self.kind == "multiclass":
            return {"accuracy": _multiclass_accuracy}
        return {"subset_accuracy": _threshold_accuracy(self.spec.threshold, subset=True)}

    def evaluate(self, targets: Any, outputs: Any) -> dict[str, float]:
        if self.kind == "regression":
            truth = _require_numeric_finite(targets, task="Regression")
            prediction = np.asarray(outputs, dtype=np.float32).reshape(truth.shape)
            residual = prediction - truth
            denominator = float(((truth - truth.mean(axis=0)) ** 2).sum())
            r2 = (
                float("nan")
                if denominator == 0.0
                else 1.0 - float((residual**2).sum()) / denominator
            )
            return {
                "mae": float(np.abs(residual).mean()),
                "mse": float((residual**2).mean()),
                "r2": r2,
            }
        prediction = self.predictions_from_outputs(outputs)
        if self.kind in {"binary", "multiclass"}:
            truth = _require_labels(targets, task=self.kind.capitalize())
            return {"accuracy": float(np.asarray(prediction == truth).mean())}
        truth_multi: np.ndarray = self.transform_targets(targets).astype(bool)
        predicted_multi = np.asarray(prediction, dtype=bool)
        return {
            "subset_accuracy": float((predicted_multi == truth_multi).all(axis=1).mean()),
            "hamming_loss": float((predicted_multi != truth_multi).mean()),
        }


def create_task_adapter(spec: TaskSpec | TaskKind | str) -> FittedTaskAdapter:
    """Create an unfitted runtime adapter from a serializable task specification."""

    task_spec = spec if isinstance(spec, TaskSpec) else TaskSpec(kind=spec)  # type: ignore[arg-type]
    return FittedTaskAdapter(spec=task_spec)


__all__ = [
    "FittedTaskAdapter",
    "TensorMetric",
    "create_task_adapter",
]
