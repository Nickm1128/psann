"""Schema-aware, bounded-memory deployment inference."""

from __future__ import annotations

import copy
import itertools
import math
import threading
import uuid
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .._sklearn.classifier import PSANNClassifier
from .accelerators import resolve_workplace_device
from .contracts import InferenceResult, TaskKind, TopKResult
from .specs import InferenceConfig

_LOCKS_GUARD = threading.Lock()
_MODEL_LOCKS: weakref.WeakKeyDictionary[torch.nn.Module, threading.RLock] = (
    weakref.WeakKeyDictionary()
)


def _core_estimator(model: Any) -> Any:
    return model.estimator_ if isinstance(model, PSANNClassifier) else model


def _torch_model(model: Any) -> torch.nn.Module:
    core = _core_estimator(model)
    module = getattr(core, "model_", None)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("Inference requires a fitted PSANN estimator with a Torch model.")
    return module


def _model_lock(model: Any) -> threading.RLock:
    module = _torch_model(model)
    with _LOCKS_GUARD:
        lock = _MODEL_LOCKS.get(module)
        if lock is None:
            lock = threading.RLock()
            _MODEL_LOCKS[module] = lock
        return lock


def _task_kind(model: Any) -> TaskKind:
    task = getattr(model, "task_spec_", None)
    if task is None:
        task = getattr(_core_estimator(model), "_platform_task_spec_", None)
    kind = getattr(task, "kind", None)
    if kind is None:
        metadata = getattr(_core_estimator(model), "task_metadata_", {})
        kind = metadata.get("kind", "regression") if isinstance(metadata, Mapping) else "regression"
    value = str(kind)
    if value not in {"regression", "binary", "multiclass", "multilabel"}:
        raise ValueError(f"Unsupported fitted task {value!r}.")
    return value  # type: ignore[return-value]


def _output_names(model: Any) -> tuple[str, ...]:
    value = getattr(model, "output_names_", None)
    if value is None:
        value = getattr(_core_estimator(model), "output_names_", ())
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _column_names(value: Any) -> tuple[str, ...]:
    columns = getattr(value, "columns", None)
    if columns is None:
        return ()
    names = tuple(str(item) for item in list(columns))
    if len(set(names)) != len(names):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Named features contain duplicate columns: {duplicates!r}.")
    return names


def _reorder_columns(value: Any, names: tuple[str, ...]) -> Any:
    locator = getattr(value, "loc", None)
    if locator is not None:
        return locator[:, list(names)]
    try:
        return value[list(names)]
    except Exception as exc:
        raise TypeError(
            "feature_policy='reorder' requires a dataframe-like input supporting column selection."
        ) from exc


def _apply_feature_policy(model: Any, inputs: Any, policy: str) -> Any:
    core = _core_estimator(model)
    expected = tuple(str(item) for item in getattr(core, "feature_names_in_", ()))
    observed = _column_names(inputs)
    if not expected:
        return inputs
    if not observed:
        shape = getattr(inputs, "shape", np.asarray(inputs).shape)
        if len(shape) < 2 or int(shape[1]) != len(expected):
            received = int(shape[1]) if len(shape) >= 2 else None
            raise ValueError(
                f"predict expected {len(expected)} named features but received {received}."
            )
        return inputs
    if policy == "positional":
        return np.asarray(inputs)
    missing = [name for name in expected if name not in observed]
    unexpected = [name for name in observed if name not in expected]
    if missing or unexpected:
        raise ValueError(
            f"predict feature schema mismatch: missing={missing!r}, unexpected={unexpected!r}."
        )
    if observed == expected:
        return inputs
    if policy == "reorder":
        return _reorder_columns(inputs, expected)
    raise ValueError(
        f"predict feature order {observed!r} does not match fitted order {expected!r}. "
        "Use feature_policy='reorder' to opt into safe name-based reordering."
    )


def _normalise_batch(model: Any, inputs: Any) -> Any:
    core = _core_estimator(model)
    expected = tuple(int(item) for item in getattr(core, "input_shape_", ()))
    if getattr(inputs, "iloc", None) is not None:
        return inputs
    array = np.asarray(inputs)
    if expected and array.ndim == len(expected):
        return array.reshape((1,) + expected)
    return array


def _sample_count(value: Any) -> int:
    shape = getattr(value, "shape", ())
    if not shape:
        raise ValueError("Inference inputs must include a batch dimension.")
    count = int(shape[0])
    if count < 1:
        raise ValueError("Inference inputs must contain at least one sample.")
    return count


def _slice(value: Any, start: int, stop: int) -> Any:
    if value is None:
        return None
    iloc = getattr(value, "iloc", None)
    if iloc is not None:
        return iloc[start:stop]
    return value[start:stop]


def _artifact_fields(model: Any) -> tuple[str | None, str | None, str | None]:
    info = getattr(model, "artifact_info_", None)
    if info is None:
        info = getattr(_core_estimator(model), "artifact_info_", None)
    artifact_version = str(getattr(info, "artifact_format_version")) if info is not None else None
    model_id = getattr(model, "artifact_id_", None)
    run_id = getattr(model, "run_id_", None)
    return (
        str(artifact_version) if artifact_version is not None else None,
        str(model_id) if model_id is not None else None,
        str(run_id) if run_id is not None else None,
    )


def _classification_values(model: PSANNClassifier, inputs: Any, context: Any, config: Any) -> Any:
    kwargs = {"context": context} if context is not None else {}
    if config.return_logits:
        return model.decision_function(inputs, **kwargs)
    if config.classification_output == "label":
        return model.predict(inputs, **kwargs)
    return model.predict_proba(inputs, **kwargs)


def _top_k_result(model: PSANNClassifier, probabilities: Any, k: int) -> TopKResult:
    values = np.asarray(probabilities, dtype=np.float32)
    single = values.ndim == 1
    if single:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise RuntimeError(
            f"Multiclass top-k requires a 2D probability matrix; received {values.shape!r}."
        )
    if k > values.shape[1]:
        raise ValueError(f"top_k={k} exceeds the fitted multiclass width {values.shape[1]}.")
    indices = np.argsort(-values, axis=1, kind="stable")[:, :k]
    ranked_probabilities = np.take_along_axis(values, indices, axis=1)
    classes = np.asarray(model.classes_, dtype=object)
    labels = classes[indices]
    if single:
        labels = labels[0]
        ranked_probabilities = ranked_probabilities[0]
        indices = indices[0]
    return TopKResult(labels=labels, probabilities=ranked_probabilities, indices=indices)


class InferenceRuntime:
    """Thread-safe deployment adapter over one fitted PSANN estimator.

    Stateless calls share a model but serialize its small amount of mode/device
    bookkeeping. Stateful rollouts must use :meth:`create_session`, which deep-copies
    the estimator so request state and optional online updates cannot leak.
    """

    def __init__(
        self,
        model: Any,
        *,
        config: InferenceConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._model = model
        self.config = (
            config
            if isinstance(config, InferenceConfig)
            else InferenceConfig.from_dict(config) if config is not None else InferenceConfig()
        )
        self.device, self._device_fallback = resolve_workplace_device(
            self.config.device,
            fallback_policy=self.config.fallback_policy,
            operation="inference",
        )
        self._lock = _model_lock(model)
        self._configure_model()

    @property
    def model(self) -> Any:
        """Return the fitted estimator owned by this runtime."""

        return self._model

    @property
    def task(self) -> TaskKind:
        return _task_kind(self._model)

    @property
    def is_stateful(self) -> bool:
        return bool(getattr(_core_estimator(self._model), "stateful", False))

    def _configure_model(self) -> None:
        core = _core_estimator(self._model)
        with self._lock:
            core.device = self.device
            ensure_device = getattr(core, "_ensure_model_device", None)
            if callable(ensure_device):
                ensure_device(self.device)
            module = _torch_model(self._model)
            module.eval()
            if hasattr(module, "set_state_updates"):
                module.set_state_updates(False)

    def metadata(self) -> Mapping[str, Any]:
        """Return service-safe model metadata without the full artifact manifest."""

        artifact_version, model_id, run_id = _artifact_fields(self._model)
        core = _core_estimator(self._model)
        metadata: dict[str, Any] = {
            "task": self.task,
            "output_names": list(_output_names(self._model)),
            "input_shape": list(getattr(core, "input_shape_", ())),
            "artifact_version": artifact_version,
            "model_id": model_id,
            "run_id": run_id,
            "device": str(self.device),
            "dtype": self.config.dtype,
            "stateful": self.is_stateful,
        }
        if self._device_fallback is not None:
            metadata["device_fallback"] = self._device_fallback
        return metadata

    def predict(
        self,
        inputs: Any,
        *,
        context: Any | None = None,
        batch_size: int | None = None,
        return_logits: bool | None = None,
    ) -> InferenceResult:
        """Run bounded-memory inference under the fitted raw-input contract."""

        config = self.config
        if batch_size is not None:
            if int(batch_size) < 1:
                raise ValueError("batch_size must be >= 1.")
            config = replace(config, batch_size=int(batch_size))
        if return_logits is not None:
            config = replace(config, return_logits=bool(return_logits))
        if self.task == "regression" and config.return_logits:
            raise ValueError("return_logits is only valid for classification tasks.")
        if config.top_k is not None and self.task != "multiclass":
            raise ValueError("top_k is only valid for multiclass inference.")

        ordered = _apply_feature_policy(self._model, inputs, config.feature_policy)
        batched = _normalise_batch(self._model, ordered)
        count = _sample_count(batched)
        context_batch = None if context is None else np.asarray(context)
        if context_batch is not None:
            if context_batch.ndim == 1:
                context_batch = context_batch.reshape(-1, 1)
            if int(context_batch.shape[0]) != count:
                raise ValueError(
                    f"context has {context_batch.shape[0]} samples but inputs have {count}."
                )

        step = count if config.device_transfer == "full_batch" else config.batch_size
        chunks: list[np.ndarray] = []
        with self._lock, torch.inference_mode():
            module = _torch_model(self._model)
            module.eval()
            if hasattr(module, "set_state_updates"):
                module.set_state_updates(False)
            for start in range(0, count, step):
                stop = min(start + step, count)
                input_chunk = _slice(batched, start, stop)
                context_chunk = _slice(context_batch, start, stop)
                if isinstance(self._model, PSANNClassifier):
                    output = _classification_values(
                        self._model,
                        input_chunk,
                        context_chunk,
                        config,
                    )
                else:
                    kwargs = {"context": context_chunk} if context_chunk is not None else {}
                    output = self._model.predict(input_chunk, **kwargs)
                array = np.asarray(output)
                if array.ndim == 0:
                    array = array.reshape(1)
                chunks.append(array)

        values = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
        artifact_version, model_id, run_id = _artifact_fields(self._model)
        if self.task == "regression":
            output_kind = "prediction"
        elif config.return_logits:
            output_kind = "logit"
        elif config.classification_output == "label":
            output_kind = "prediction"
        else:
            output_kind = "probability"
        metadata: dict[str, Any] = {
            "batch_size": config.batch_size,
            "chunks": int(math.ceil(count / step)),
            "device": str(self.device),
            "device_transfer": config.device_transfer,
            "dtype": config.dtype,
            "num_samples": count,
            "output_kind": output_kind,
        }
        top_k = (
            _top_k_result(self._model, values, config.top_k)
            if config.top_k is not None and isinstance(self._model, PSANNClassifier)
            else None
        )
        if top_k is not None:
            metadata["top_k"] = config.top_k
        if self._device_fallback is not None:
            metadata["device_fallback"] = self._device_fallback
        return InferenceResult(
            values=values,
            task=self.task,
            output_names=_output_names(self._model),
            artifact_version=artifact_version,
            model_id=model_id,
            run_id=run_id,
            metadata=metadata,
            top_k=top_k,
        )

    def create_session(self, *, session_id: str | None = None) -> "InferenceSession":
        """Create isolated state for an explicit streaming request lifecycle."""

        if not self.is_stateful:
            raise RuntimeError(
                "Inference sessions are only available for models trained with stateful=True."
            )
        with self._lock:
            session_model = copy.deepcopy(self._model)
        return InferenceSession(
            session_model,
            config=self.config,
            session_id=session_id or str(uuid.uuid4()),
        )

    def make_explainer(self, **kwargs: Any) -> Any:
        """Create an optional SHAP explainer over this runtime's raw-input contract."""

        from .explainability import make_explainer

        return make_explainer(self, **kwargs)

    def explain(self, inputs: Any, **kwargs: Any) -> Any:
        """Create and execute a one-shot optional SHAP explanation."""

        from .explainability import explain

        return explain(self, inputs, **kwargs)


class InferenceSession:
    """Isolated mutable state for one streaming inference lifecycle."""

    def __init__(self, model: Any, *, config: InferenceConfig, session_id: str) -> None:
        self._model: Any | None = model
        self.config = config
        self.session_id = str(session_id)
        self._lock = threading.RLock()
        self._closed = False
        self.device, self._device_fallback = resolve_workplace_device(
            config.device,
            fallback_policy=config.fallback_policy,
            operation="inference",
        )
        core = _core_estimator(model)
        core.device = self.device
        core._ensure_model_device(core.device)
        core.model_.eval()
        core.reset_state()

    def _require_open(self) -> Any:
        if self._closed or self._model is None:
            raise RuntimeError("Inference session is closed.")
        return self._model

    def _result(self, values: Any, *, output_kind: str) -> InferenceResult:
        model = self._require_open()
        artifact_version, model_id, run_id = _artifact_fields(model)
        metadata: dict[str, Any] = {
            "device": str(self.device),
            "output_kind": output_kind,
            "session_id": self.session_id,
            "stateful": True,
        }
        top_k = (
            _top_k_result(model, values, self.config.top_k)
            if self.config.top_k is not None
            and output_kind == "probability"
            and isinstance(model, PSANNClassifier)
            and _task_kind(model) == "multiclass"
            else None
        )
        if top_k is not None:
            metadata["top_k"] = self.config.top_k
        if self._device_fallback is not None:
            metadata["device_fallback"] = self._device_fallback
        return InferenceResult(
            values=values,
            task=_task_kind(model),
            output_names=_output_names(model),
            artifact_version=artifact_version,
            model_id=model_id,
            run_id=run_id,
            metadata=metadata,
            top_k=top_k,
        )

    def _convert(self, raw: Any, *, single: bool) -> tuple[Any, str]:
        model = self._require_open()
        if not isinstance(model, PSANNClassifier):
            return raw, "prediction"
        array = np.asarray(raw)
        logits = array.reshape(1, -1) if single else array
        if self.config.return_logits:
            converted = logits
            kind = "logit"
        elif self.config.classification_output == "label":
            converted = model.task_adapter_.predictions_from_outputs(logits)
            kind = "prediction"
        else:
            converted = model.task_adapter_.probabilities(logits)
            kind = "probability"
        if single:
            converted = np.asarray(converted)[0]
        return converted, kind

    def step(
        self,
        inputs: Any,
        *,
        context: Any | None = None,
        target: Any | None = None,
        update_params: bool = False,
    ) -> InferenceResult:
        """Advance this session by one step without touching the shared runtime."""

        model = self._require_open()
        core = _core_estimator(model)
        with self._lock:
            raw = core.step(
                inputs,
                context=context,
                target=target,
                update_params=update_params,
                update_state=True,
            )
            if hasattr(core.model_, "commit_state_updates"):
                core.model_.commit_state_updates()
            values, kind = self._convert(raw, single=True)
        return self._result(values, output_kind=kind)

    def predict_sequence(
        self,
        inputs: Any,
        *,
        context: Any | None = None,
        reset_state: bool = False,
    ) -> InferenceResult:
        """Run an ordered rollout inside this isolated session."""

        model = self._require_open()
        core = _core_estimator(model)
        with self._lock:
            sequence = core._coerce_sequence_inputs(inputs)
            contexts = (
                core._coerce_sequence_context(context, int(sequence.shape[0]))
                if context is not None
                else None
            )
            if reset_state:
                core.reset_state()
            outputs: list[np.ndarray] = []
            kind = "prediction"
            for index in range(int(sequence.shape[0])):
                raw = core.step(
                    sequence[index],
                    context=None if contexts is None else contexts[index : index + 1],
                    update_params=False,
                    update_state=True,
                )
                if hasattr(core.model_, "commit_state_updates"):
                    core.model_.commit_state_updates()
                converted, kind = self._convert(raw, single=True)
                outputs.append(np.asarray(converted))
            values = np.stack(outputs, axis=0)
        return self._result(values, output_kind=kind)

    def reset(self) -> None:
        """Reset this session's state controller to its fitted initialization."""

        model = self._require_open()
        with self._lock:
            _core_estimator(model).reset_state()

    def close(self) -> None:
        """Release the isolated estimator and reject future session calls."""

        with self._lock:
            self._model = None
            self._closed = True

    def __enter__(self) -> "InferenceSession":
        self._require_open()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class InferencePool:
    """Round-robin pool of independently loaded single-device runtimes."""

    def __init__(self, runtimes: tuple[InferenceRuntime, ...]) -> None:
        if not runtimes:
            raise ValueError("InferencePool requires at least one runtime.")
        self.runtimes = runtimes
        self._counter = itertools.count()
        self._selection_lock = threading.Lock()

    def _select(self) -> tuple[int, InferenceRuntime]:
        with self._selection_lock:
            index = next(self._counter) % len(self.runtimes)
        return index, self.runtimes[index]

    def predict(self, inputs: Any, **kwargs: Any) -> InferenceResult:
        """Route one stateless request to the next independent device runtime."""

        index, runtime = self._select()
        result = runtime.predict(inputs, **kwargs)
        return replace(
            result,
            metadata={
                **dict(result.metadata),
                "pool_index": index,
                "pool_size": len(self.runtimes),
            },
        )

    def create_session(self, *, session_id: str | None = None) -> InferenceSession:
        """Pin a new explicit session to one runtime's device."""

        _, runtime = self._select()
        return runtime.create_session(session_id=session_id)

    def metadata(self) -> Mapping[str, Any]:
        """Return pool topology without exposing model internals."""

        return {
            "pool_size": len(self.runtimes),
            "devices": [str(runtime.device) for runtime in self.runtimes],
            "model": dict(self.runtimes[0].metadata()),
        }


def create_inference_runtime(
    model: Any,
    *,
    config: InferenceConfig | Mapping[str, Any] | None = None,
) -> InferenceRuntime:
    """Wrap a fitted estimator in the stable deployment inference adapter."""

    return InferenceRuntime(model, config=config)


def load_runtime(
    path: str | Path,
    *,
    config: InferenceConfig | Mapping[str, Any] | None = None,
) -> InferenceRuntime:
    """Restricted-load a native artifact directly into an inference runtime."""

    resolved = (
        config
        if isinstance(config, InferenceConfig)
        else InferenceConfig.from_dict(config) if config is not None else InferenceConfig()
    )
    from .artifacts import load_model

    model = load_model(path, device=resolved.device)
    return InferenceRuntime(model, config=resolved)


def load_runtime_pool(
    path: str | Path,
    *,
    devices: tuple[str | torch.device, ...],
    config: InferenceConfig | Mapping[str, Any] | None = None,
) -> InferencePool:
    """Load one independent artifact instance per configured serving device."""

    if not devices:
        raise ValueError("devices must contain at least one device.")
    base = (
        config
        if isinstance(config, InferenceConfig)
        else InferenceConfig.from_dict(config) if config is not None else InferenceConfig()
    )
    runtimes = tuple(
        load_runtime(path, config=replace(base, device=str(device))) for device in devices
    )
    return InferencePool(runtimes)


__all__ = [
    "InferencePool",
    "InferenceRuntime",
    "InferenceSession",
    "create_inference_runtime",
    "load_runtime",
    "load_runtime_pool",
]
