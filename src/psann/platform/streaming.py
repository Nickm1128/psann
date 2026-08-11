"""Bounded-memory supervised training over restartable batch streams."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .._sklearn.classifier import PSANNClassifier
from .contracts import BackboneProtocol
from .lifecycle import (
    SupervisedData,
    TrainingRun,
    _model_spec_from_instance,
    _supervised_data,
    train,
)
from .module_adapter import TorchModuleAdapter
from .operations import OperationalHooks, fingerprint_data, fingerprint_model
from .specs import TrainingConfig

BatchLike = SupervisedData | Sequence[Any] | Mapping[str, Any]
BatchFactory = Callable[[], Iterable[BatchLike]]


@dataclass(frozen=True)
class StreamingSupervisedData:
    """Restartable batch source that does not require a full in-memory dataset."""

    batch_factory: BatchFactory
    steps_per_epoch: int | None = None
    max_batch_samples: int = 65_536
    name: str = "stream"

    def __post_init__(self) -> None:
        if not callable(self.batch_factory):
            raise TypeError("batch_factory must be callable and return a fresh iterable.")
        if self.steps_per_epoch is not None and self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1 when provided.")
        if self.max_batch_samples < 1:
            raise ValueError("max_batch_samples must be >= 1.")
        if not self.name.strip():
            raise ValueError("stream name cannot be empty.")

    def batches(self) -> Iterator[SupervisedData]:
        count = 0
        for value in self.batch_factory():
            if self.steps_per_epoch is not None and count >= self.steps_per_epoch:
                break
            batch = _supervised_data(value)
            shape = getattr(batch.inputs, "shape", ())
            if not shape:
                raise ValueError("A streaming batch must include a batch dimension.")
            samples = int(shape[0])
            if samples < 1:
                raise ValueError("Streaming batches cannot be empty.")
            if samples > self.max_batch_samples:
                raise ValueError(
                    f"Streaming batch has {samples} samples; maximum is {self.max_batch_samples}."
                )
            target_shape = getattr(batch.targets, "shape", ())
            if not target_shape or int(target_shape[0]) != samples:
                raise ValueError("Streaming inputs and targets must have matching samples.")
            if batch.context is not None:
                context_shape = getattr(batch.context, "shape", ())
                if not context_shape or int(context_shape[0]) != samples:
                    raise ValueError("Streaming context and inputs must have matching samples.")
            count += 1
            yield batch
        if count == 0:
            raise ValueError("The streaming batch factory produced no batches.")
        if self.steps_per_epoch is not None and count < self.steps_per_epoch:
            raise ValueError(
                f"The streaming batch factory produced {count} batches; expected "
                f"steps_per_epoch={self.steps_per_epoch}."
            )


@dataclass(frozen=True)
class NumpyShard:
    """One memory-mapped set of input, target, and optional context `.npy` files."""

    inputs: str | os.PathLike[str]
    targets: str | os.PathLike[str]
    context: str | os.PathLike[str] | None = None


def numpy_shard_stream(
    shards: Sequence[NumpyShard],
    *,
    batch_size: int,
    drop_last: bool = False,
    name: str = "numpy_shards",
) -> StreamingSupervisedData:
    """Create a restartable memory-mapped stream over uncompressed `.npy` shards."""

    reviewed = tuple(shards)
    if not reviewed:
        raise ValueError("At least one NumPy shard is required.")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    for shard in reviewed:
        for path_value in (shard.inputs, shard.targets, shard.context):
            if path_value is None:
                continue
            path = Path(path_value)
            if path.suffix.lower() != ".npy":
                raise ValueError("NumPy shard files must use the uncompressed .npy format.")
            if not path.is_file():
                raise FileNotFoundError(path)

    def factory() -> Iterator[SupervisedData]:
        for shard in reviewed:
            inputs = np.load(shard.inputs, mmap_mode="r", allow_pickle=False)
            targets = np.load(shard.targets, mmap_mode="r", allow_pickle=False)
            context = (
                np.load(shard.context, mmap_mode="r", allow_pickle=False)
                if shard.context is not None
                else None
            )
            samples = int(inputs.shape[0])
            if int(targets.shape[0]) != samples:
                raise ValueError("NumPy shard inputs and targets have different lengths.")
            if context is not None and int(context.shape[0]) != samples:
                raise ValueError("NumPy shard context and inputs have different lengths.")
            for start in range(0, samples, batch_size):
                stop = min(start + batch_size, samples)
                if drop_last and stop - start < batch_size:
                    continue
                yield SupervisedData(
                    inputs=inputs[start:stop],
                    targets=targets[start:stop],
                    context=context[start:stop] if context is not None else None,
                )

    return StreamingSupervisedData(
        batch_factory=factory,
        max_batch_samples=batch_size,
        name=name,
    )


def train_streaming(
    model: BackboneProtocol,
    stream: StreamingSupervisedData,
    *,
    config: TrainingConfig | Mapping[str, Any] | None = None,
    operational_hooks: OperationalHooks | None = None,
) -> TrainingRun:
    """Train a regression estimator incrementally over bounded restartable batches.

    Each stream pass is one logical epoch. Model weights are warm-started between
    batches; optimizer state is intentionally batch-local in this first bounded
    contract. Classification and arbitrary-module streaming remain unsupported until
    they have class-vocabulary and checkpoint semantics.
    """

    if isinstance(model, (PSANNClassifier, TorchModuleAdapter)):
        raise NotImplementedError(
            "Streaming training currently supports registered regression estimators only."
        )
    training_config = (
        config
        if isinstance(config, TrainingConfig)
        else TrainingConfig.from_dict(config) if config is not None else TrainingConfig()
    )
    if training_config.resume_from is not None or training_config.checkpoint_dir is not None:
        raise ValueError(
            "Streaming resume/checkpoints require a stream cursor contract and are not "
            "silently mapped to array checkpoints."
        )
    if training_config.scheduler != "none" or training_config.early_stopping:
        raise ValueError(
            "Streaming training currently requires scheduler='none' and early_stopping=False."
        )
    if training_config.compile:
        raise ValueError("Streaming batch-local compilation is not supported.")

    original_warm_start = bool(getattr(model, "warm_start", False))
    per_batch_config = replace(
        training_config,
        epochs=1,
        early_stopping=False,
        checkpoint_every=0,
    )
    history: list[dict[str, Any]] = []
    metric_totals: dict[str, float] = {}
    total_samples = 0
    total_batches = 0
    stream_hasher = hashlib.sha256()
    stream_hasher.update(b"psann-stream-fingerprint-v1")
    final_run: TrainingRun | None = None
    try:
        setattr(model, "warm_start", True)
        for stream_epoch in range(1, training_config.epochs + 1):
            epoch_batches = 0
            for batch_index, batch in enumerate(stream.batches(), start=1):
                final_run = train(
                    model,
                    batch,
                    config=per_batch_config,
                    operational_hooks=operational_hooks,
                )
                samples = int(getattr(batch.inputs, "shape")[0])
                total_samples += samples
                total_batches += 1
                epoch_batches += 1
                batch_fingerprint = fingerprint_data(
                    batch.inputs,
                    batch.targets,
                    batch.context,
                )
                stream_hasher.update(batch_fingerprint.encode("ascii"))
                for entry in final_run.history:
                    history.append(
                        {
                            **dict(entry),
                            "stream_epoch": stream_epoch,
                            "stream_batch": batch_index,
                        }
                    )
                for name, value in final_run.metrics.items():
                    numeric = float(value)
                    if math.isfinite(numeric):
                        metric_totals[name] = metric_totals.get(name, 0.0) + numeric * samples
            if epoch_batches == 0:
                raise ValueError("The streaming source produced no batches for an epoch.")
    finally:
        setattr(model, "warm_start", original_warm_start)

    if final_run is None or total_samples == 0:
        raise ValueError("The streaming source produced no training samples.")
    metrics = {name: value / total_samples for name, value in metric_totals.items()}
    model_spec = _model_spec_from_instance(model)
    run_id = str(uuid.uuid4())
    result = TrainingRun(
        model=model,
        model_spec=model_spec,
        training_config=training_config,
        run_id=run_id,
        history=tuple(history),
        metrics=metrics,
        metadata={
            "backbone": model_spec.backbone,
            "task": model_spec.task.kind,
            "streaming": {
                "name": stream.name,
                "epochs": training_config.epochs,
                "batches": total_batches,
                "samples_seen": total_samples,
                "optimizer_state": "batch_local",
            },
            "data_fingerprint": f"sha256:{stream_hasher.hexdigest()}",
            "model_fingerprint": fingerprint_model(model),
        },
        operational_hooks=operational_hooks,
    )
    if operational_hooks is not None:
        operational_hooks.emit(
            result.operational_event(
                "stream_training_completed",
                metadata=result.metadata,
            )
        )
    return result


__all__ = [
    "NumpyShard",
    "StreamingSupervisedData",
    "numpy_shard_stream",
    "train_streaming",
]
