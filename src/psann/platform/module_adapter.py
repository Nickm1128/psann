"""In-process adapter for advanced users supplying an arbitrary torch module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .._sklearn.schema import capture_fit_schema, validate_inference_schema
from .._sklearn.shared import BaseEstimator
from ..training import TrainingLoopConfig, run_training_loop
from ..utils import choose_device, seed_all
from .specs import TaskSpec
from .tasks import FittedTaskAdapter, create_task_adapter


class TorchModuleAdapter(BaseEstimator):
    """Train a user module in process with intentionally limited guarantees."""

    def __init__(
        self,
        module: torch.nn.Module,
        *,
        task: str = "regression",
        epochs: int = 20,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        threshold: float | tuple[float, ...] = 0.5,
        device: str = "auto",
        random_state: int | None = None,
    ) -> None:
        self.module = module
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.threshold = threshold
        self.device = device
        self.random_state = random_state
        self.data_format = "channels_first"
        self.artifact_capabilities_: tuple[str, ...] = (
            "in_process_training",
            "in_process_inference",
        )
        self.experimental_ = True

    def _optimizer(self) -> torch.optim.Optimizer:
        name = str(self.optimizer).strip().lower()
        parameters = self.model_.parameters()
        if name == "adam":
            return torch.optim.Adam(
                parameters,
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
        if name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
        if name == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=float(self.learning_rate),
                momentum=0.9,
                weight_decay=float(self.weight_decay),
            )
        raise ValueError("optimizer must be adam, adamw, or sgd.")

    def _task_spec(self, targets: Any) -> TaskSpec:
        kind = str(self.task).strip().lower()
        if kind == "auto":
            array = np.asarray(targets)
            kind = (
                "multilabel"
                if array.ndim == 2 and array.shape[1] > 1
                else ("binary" if len(np.unique(array)) == 2 else "multiclass")
            )
        return TaskSpec(kind=kind, threshold=self.threshold)  # type: ignore[arg-type]

    def fit(self, X: Any, y: Any) -> "TorchModuleAdapter":
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive.")
        X = capture_fit_schema(self, X, y)
        inputs = np.asarray(X, dtype=np.float32)
        if np.isnan(inputs).any() or np.isinf(inputs).any():
            raise ValueError("X must contain only finite values.")
        adapter = create_task_adapter(self._task_spec(y))
        targets: np.ndarray = adapter.fit_targets(y).astype(np.float32, copy=False)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        seed_all(self.random_state)
        device = choose_device(self.device)
        self.model_ = self.module.to(device)
        self.input_shape_ = tuple(int(item) for item in inputs.shape[1:])
        with torch.no_grad():
            sample = torch.from_numpy(inputs[:1]).to(device)
            output = self.model_(sample)
        if tuple(output.shape[1:]) != tuple(targets.shape[1:]):
            raise ValueError(
                f"Module output shape {tuple(output.shape[1:])!r} does not match target "
                f"shape {tuple(targets.shape[1:])!r}."
            )
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(int(self.random_state))
        loader = DataLoader(
            TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)),
            batch_size=int(self.batch_size),
            shuffle=True,
            generator=generator,
        )
        config = TrainingLoopConfig(
            epochs=int(self.epochs),
            patience=int(self.epochs),
            early_stopping=False,
            stateful=False,
            state_reset="batch",
            verbose=0,
            lr_max=None,
            lr_min=None,
            seed=self.random_state,
        )
        loss = adapter.loss()
        loss_fn = torch.nn.MSELoss() if loss == "mse" else loss
        history, _ = run_training_loop(
            self.model_,
            optimizer=self._optimizer(),
            loss_fn=loss_fn,
            train_loader=loader,
            device=device,
            cfg=config,
            metrics=adapter.training_metrics(),
            metadata={
                "adapter": "arbitrary_module",
                "artifact_capabilities": list(self.artifact_capabilities_),
            },
        )
        self.history_ = history
        self.task_adapter_: FittedTaskAdapter = adapter
        self.classes_ = np.asarray(adapter.classes, dtype=object)
        self.output_names_ = np.asarray(adapter.output_names, dtype=object)
        self.task_spec_ = adapter.spec
        self.n_outputs_ = int(targets.shape[1])
        self._output_dim_ = self.n_outputs_
        self._primary_dim_ = self.n_outputs_
        self._output_shape_tuple_: tuple[int, ...] = (self.n_outputs_,)
        self.input_dtype_ = "float32"
        self.feature_schema_policy_ = str(
            getattr(getattr(self, "_platform_data_schema_", None), "feature_policy", "strict")
        )
        self.data_format_ = self.data_format
        self.preprocessing_contract_ = {
            "input_scaler": {"kind": None, "state": None},
            "target_scaler": {"kind": None, "state": None},
        }
        self.training_metadata_: dict[str, Any] = {
            "adapter": "registered_torch_module",
            "artifact_capabilities": list(self.artifact_capabilities_),
        }
        self._device_ = device
        return self

    def decision_function(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchModuleAdapter is not fitted; call fit first.")
        X = validate_inference_schema(self, X)
        inputs = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            output = self.model_(torch.from_numpy(inputs).to(self._device_))
        return output.detach().cpu().numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.task_adapter_.probabilities(self.decision_function(X))

    def predict(self, X: Any) -> np.ndarray:
        output = self.decision_function(X)
        return self.task_adapter_.predictions_from_outputs(output)

    def score(self, X: Any, y: Any) -> float:
        metrics = self.task_adapter_.evaluate(y, self.decision_function(X))
        for name in ("r2", "accuracy", "subset_accuracy"):
            if name in metrics:
                return float(metrics[name])
        raise RuntimeError("Task adapter did not produce a score metric.")

    def export(self, path: str | Path, **kwargs: Any) -> Path:
        """Export only when the module came from a registered reconstructable factory."""

        identifier = str(getattr(self, "backbone_id_", "arbitrary_module"))
        if identifier == "arbitrary_module" or not hasattr(self, "_platform_model_spec_dict_"):
            raise NotImplementedError(
                "Arbitrary modules guarantee in-process training/inference only. Register a "
                "backbone with explicit artifact capabilities before exporting it."
            )
        from .artifacts import export_model

        return export_model(self, path, **kwargs)


__all__ = ["TorchModuleAdapter"]
