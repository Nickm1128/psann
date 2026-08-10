from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import psann


def _model(seed: int = 901):
    return psann.create_model(
        psann.ModelSpec(
            input_schema=psann.DataSchema(input_shape=(3,)),
            parameters={"hidden_layers": 1, "hidden_units": 6, "random_state": seed},
        )
    )


def test_restartable_stream_trains_without_materializing_all_batches():
    rng = np.random.default_rng(901)
    inputs = rng.normal(size=(18, 3)).astype(np.float32)
    targets = (inputs[:, 0] - inputs[:, 1]).astype(np.float32)
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        for start in range(0, len(inputs), 6):
            yield inputs[start : start + 6], targets[start : start + 6]

    stream = psann.StreamingSupervisedData(
        batch_factory=factory,
        steps_per_epoch=3,
        max_batch_samples=6,
        name="bounded-fixture",
    )
    run = psann.train_streaming(
        _model(),
        stream,
        config=psann.TrainingConfig(
            epochs=2,
            batch_size=6,
            learning_rate=1e-3,
            deterministic=True,
        ),
    )
    assert calls == 2
    assert run.metadata["streaming"] == {
        "name": "bounded-fixture",
        "epochs": 2,
        "batches": 6,
        "samples_seen": 36,
        "optimizer_state": "batch_local",
    }
    assert run.metadata["data_fingerprint"].startswith("sha256:")
    assert run.metadata["model_fingerprint"].startswith("sha256:")
    assert len(run.history) == 6
    assert run.model.predict(inputs[:2]).shape == (2,)


def test_numpy_shard_stream_uses_memory_mapped_bounded_batches(tmp_path: Path):
    rng = np.random.default_rng(902)
    inputs = rng.normal(size=(11, 3)).astype(np.float32)
    targets = inputs[:, 0].astype(np.float32)
    input_path = tmp_path / "inputs.npy"
    target_path = tmp_path / "targets.npy"
    np.save(input_path, inputs)
    np.save(target_path, targets)
    stream = psann.numpy_shard_stream(
        [psann.NumpyShard(inputs=input_path, targets=target_path)],
        batch_size=4,
    )
    batches = list(stream.batches())
    assert [batch.inputs.shape[0] for batch in batches] == [4, 4, 3]
    assert isinstance(batches[0].inputs.base, np.memmap)
    np.testing.assert_array_equal(
        np.concatenate([np.asarray(batch.inputs) for batch in batches]),
        inputs,
    )


def test_stream_rejects_empty_oversized_and_short_factories():
    empty = psann.StreamingSupervisedData(lambda: iter(()))
    with pytest.raises(ValueError, match="no batches"):
        list(empty.batches())

    oversized = psann.StreamingSupervisedData(
        lambda: iter([(np.zeros((3, 2)), np.zeros(3))]),
        max_batch_samples=2,
    )
    with pytest.raises(ValueError, match="maximum"):
        list(oversized.batches())

    short = psann.StreamingSupervisedData(
        lambda: iter([(np.zeros((1, 2)), np.zeros(1))]),
        steps_per_epoch=2,
    )
    with pytest.raises(ValueError, match="expected steps_per_epoch"):
        list(short.batches())


def test_streaming_resume_compile_and_classification_fail_explicitly(tmp_path: Path):
    inputs = np.zeros((4, 3), dtype=np.float32)
    targets = np.zeros(4, dtype=np.float32)
    stream = psann.StreamingSupervisedData(lambda: iter([(inputs, targets)]))
    with pytest.raises(ValueError, match="cursor contract"):
        psann.train_streaming(
            _model(),
            stream,
            config=psann.TrainingConfig(
                epochs=1,
                checkpoint_dir=str(tmp_path),
            ),
        )
    with pytest.raises(ValueError, match="compilation"):
        psann.train_streaming(
            _model(),
            stream,
            config=psann.TrainingConfig(epochs=1, compile=True),
        )
    classifier = psann.create_model(
        psann.ModelSpec(
            task=psann.TaskSpec(kind="binary"),
            input_schema=psann.DataSchema(input_shape=(3,)),
        )
    )
    with pytest.raises(NotImplementedError, match="regression"):
        psann.train_streaming(
            classifier,
            stream,
            config=psann.TrainingConfig(epochs=1),
        )
