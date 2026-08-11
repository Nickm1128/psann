from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import psann
from psann.platform import DataSchema, InferenceConfig, ModelSpec, TaskSpec, TrainingConfig


def _training_config() -> TrainingConfig:
    return TrainingConfig(epochs=1, batch_size=4, deterministic=True)


def _parameters(**extra: Any) -> dict[str, Any]:
    return {
        "hidden_layers": 1,
        "hidden_units": 5,
        "random_state": 17,
        **extra,
    }


@pytest.fixture(scope="module")
def regression_run():
    rng = np.random.default_rng(201)
    inputs = rng.normal(size=(17, 3)).astype(np.float32)
    targets = (inputs[:, 0] - 0.25 * inputs[:, 1]).astype(np.float32)
    spec = ModelSpec(
        input_schema=DataSchema(input_shape=(3,), output_names=("demand",)),
        parameters=_parameters(scaler="standard", target_scaler="standard"),
    )
    run = psann.train(
        psann.create_model(spec),
        (inputs, targets),
        config=_training_config(),
    )
    return run, inputs


def test_inference_config_validates_deployment_choices():
    config = InferenceConfig(
        batch_size=16,
        classification_output="label",
        device_transfer="full_batch",
    )
    assert InferenceConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="classification_output"):
        InferenceConfig(classification_output="scores")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive integer"):
        InferenceConfig(top_k=0)
    with pytest.raises(ValueError, match="positive integer"):
        InferenceConfig(top_k=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="probability output"):
        InferenceConfig(classification_output="label", top_k=1)
    with pytest.raises(ValueError, match="device_transfer"):
        InferenceConfig(device_transfer="unbounded")  # type: ignore[arg-type]


def test_runtime_batches_raw_inputs_and_returns_stable_metadata(
    regression_run,
    monkeypatch,
):
    run, inputs = regression_run
    expected = run.model.predict(inputs)
    calls: list[int] = []
    original = run.model.predict

    def recording_predict(batch, **kwargs):
        calls.append(int(batch.shape[0]))
        return original(batch, **kwargs)

    monkeypatch.setattr(run.model, "predict", recording_predict)
    runtime = psann.create_inference_runtime(
        run.model,
        config=InferenceConfig(batch_size=4, device="cpu"),
    )
    result = runtime.predict(inputs)

    assert calls == [4, 4, 4, 4, 1]
    np.testing.assert_allclose(result.values, expected, rtol=1e-6, atol=1e-6)
    assert result.task == "regression"
    assert result.output_names == ("demand",)
    assert result.metadata == {
        "batch_size": 4,
        "chunks": 5,
        "device": "cpu",
        "device_transfer": "per_batch",
        "dtype": "float32",
        "num_samples": 17,
        "output_kind": "prediction",
    }


def test_full_batch_transfer_policy_is_explicit(regression_run, monkeypatch):
    run, inputs = regression_run
    calls: list[int] = []
    original = run.model.predict

    def recording_predict(batch, **kwargs):
        calls.append(int(batch.shape[0]))
        return original(batch, **kwargs)

    monkeypatch.setattr(run.model, "predict", recording_predict)
    runtime = psann.create_inference_runtime(
        run.model,
        config=InferenceConfig(batch_size=2, device_transfer="full_batch"),
    )
    result = runtime.predict(inputs)

    assert calls == [17]
    assert result.metadata["chunks"] == 1


def test_runtime_executes_in_eval_and_inference_mode(regression_run):
    run, inputs = regression_run
    observed: list[tuple[bool, bool]] = []

    def record(module, _arguments):
        observed.append((module.training, torch.is_inference_mode_enabled()))

    handle = run.model.model_.register_forward_pre_hook(record)
    try:
        runtime = psann.create_inference_runtime(run.model)
        runtime.predict(inputs[:3])
    finally:
        handle.remove()

    assert observed
    assert all(not training and inference_mode for training, inference_mode in observed)
    assert run.model.model_.training is False


def test_concurrent_stateless_requests_are_deterministic_and_non_mutating(regression_run):
    run, inputs = regression_run
    runtime = psann.create_inference_runtime(run.model, config=InferenceConfig(batch_size=3))
    before = {
        name: tensor.detach().clone() for name, tensor in run.model.model_.state_dict().items()
    }

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(runtime.predict, inputs) for _ in range(12)]
    results = [future.result().values for future in futures]

    for values in results[1:]:
        np.testing.assert_array_equal(values, results[0])
    for name, tensor in run.model.model_.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("task", "targets", "width"),
    [
        (TaskSpec(kind="binary"), np.asarray([0, 1] * 6), 2),
        (TaskSpec(kind="multiclass"), np.asarray([0, 1, 2] * 4), 3),
        (
            TaskSpec(kind="multilabel", class_names=("risk", "review")),
            np.asarray([[0, 1], [1, 0], [1, 1]] * 4),
            2,
        ),
    ],
)
def test_classification_probability_logit_and_label_contracts(task, targets, width):
    inputs = np.random.default_rng(202).normal(size=(12, 4)).astype(np.float32)
    model = psann.create_model(
        ModelSpec(
            task=task,
            input_schema=DataSchema(input_shape=(4,)),
            parameters=_parameters(),
        )
    )
    psann.train(model, (inputs, targets), config=_training_config())
    runtime = psann.create_inference_runtime(model, config=InferenceConfig(batch_size=5))

    probabilities = runtime.predict(inputs)
    logits = runtime.predict(inputs, return_logits=True)
    labels = psann.create_inference_runtime(
        model,
        config=InferenceConfig(classification_output="label"),
    ).predict(inputs)

    assert probabilities.values.shape == (12, width)
    assert logits.values.shape == (12, width - 1 if task.kind == "binary" else width)
    assert labels.values.shape == ((12, width) if task.kind == "multilabel" else (12,))
    assert probabilities.metadata["output_kind"] == "probability"
    assert logits.metadata["output_kind"] == "logit"
    assert labels.metadata["output_kind"] == "prediction"


def test_multiclass_runtime_returns_ranked_top_k_labels_and_probabilities():
    inputs = np.random.default_rng(212).normal(size=(12, 4)).astype(np.float32)
    targets = np.asarray(["red", "green", "blue"] * 4)
    model = psann.create_model(
        ModelSpec(
            task=TaskSpec(kind="multiclass"),
            input_schema=DataSchema(input_shape=(4,)),
            parameters=_parameters(),
        )
    )
    psann.train(model, (inputs, targets), config=_training_config())
    result = psann.create_inference_runtime(
        model,
        config=InferenceConfig(batch_size=2, top_k=2),
    ).predict(inputs[:5])

    assert result.values.shape == (5, 3)
    assert result.top_k is not None
    assert result.top_k.labels.shape == (5, 2)
    assert result.top_k.probabilities.shape == (5, 2)
    assert result.top_k.indices.shape == (5, 2)
    np.testing.assert_allclose(
        result.top_k.probabilities,
        np.take_along_axis(result.values, result.top_k.indices, axis=1),
    )
    assert np.all(result.top_k.probabilities[:, 0] >= result.top_k.probabilities[:, 1])
    assert result.metadata["top_k"] == 2

    with pytest.raises(ValueError, match="fitted multiclass width"):
        psann.create_inference_runtime(
            model,
            config=InferenceConfig(top_k=4),
        ).predict(inputs[:1])


def test_runtime_applies_its_named_feature_policy():
    pandas = pytest.importorskip("pandas")
    inputs = pandas.DataFrame(
        np.random.default_rng(203).normal(size=(10, 2)),
        columns=["amount", "age"],
    )
    targets = inputs["amount"].to_numpy(dtype=np.float32)
    model = psann.create_model(
        ModelSpec(
            input_schema=DataSchema(
                input_shape=(2,),
                feature_names=("amount", "age"),
                feature_policy="strict",
            ),
            parameters=_parameters(),
        )
    )
    psann.train(model, (inputs, targets), config=_training_config())
    reordered = inputs[["age", "amount"]]

    strict = psann.create_inference_runtime(model)
    with pytest.raises(ValueError, match="feature order"):
        strict.predict(reordered)
    safe = psann.create_inference_runtime(
        model,
        config=InferenceConfig(feature_policy="reorder"),
    )
    np.testing.assert_allclose(
        safe.predict(reordered).values,
        model.predict(inputs),
        rtol=0.0,
        atol=0.0,
    )


def test_context_is_chunked_with_inputs():
    rng = np.random.default_rng(204)
    inputs = rng.normal(size=(10, 3)).astype(np.float32)
    context = rng.normal(size=(10, 2)).astype(np.float32)
    targets = (inputs[:, 0] + context[:, 0]).astype(np.float32)
    model = psann.create_model(
        ModelSpec(
            backbone="wave_resnet",
            input_schema=DataSchema(input_shape=(3,)),
            parameters=_parameters(context_dim=2),
        )
    )
    psann.train(
        model,
        psann.SupervisedData(inputs, targets, context),
        config=_training_config(),
    )
    runtime = psann.create_inference_runtime(model, config=InferenceConfig(batch_size=3))

    result = runtime.predict(inputs, context=context)

    np.testing.assert_allclose(
        result.values,
        model.predict(inputs, context=context),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="context has"):
        runtime.predict(inputs, context=context[:-1])


def test_stateful_behavior_requires_isolated_explicit_sessions():
    rng = np.random.default_rng(205)
    inputs = rng.normal(size=(10, 3)).astype(np.float32)
    targets = inputs[:, 0]
    model = psann.create_model(
        ModelSpec(
            input_schema=DataSchema(input_shape=(3,)),
            parameters=_parameters(
                stateful=True,
                state_reset="batch",
                state={
                    "rho": 0.5,
                    "beta": 1.0,
                    "init": 1.0,
                    "max_abs": 5.0,
                    "detach": True,
                },
            ),
        )
    )
    psann.train(model, (inputs, targets), config=_training_config())
    runtime = psann.create_inference_runtime(model)
    shared_before = {
        name: tensor.detach().clone() for name, tensor in model.model_.state_dict().items()
    }

    first = runtime.predict(inputs[:2]).values
    second = runtime.predict(inputs[:2]).values
    np.testing.assert_array_equal(first, second)

    session_a = runtime.create_session(session_id="request-a")
    session_b = runtime.create_session(session_id="request-b")
    a_first = session_a.step(inputs[0]).values
    a_second_result = session_a.step(inputs[0])
    a_second = a_second_result.values
    b_first = session_b.step(inputs[0]).values
    np.testing.assert_allclose(a_first, b_first, rtol=0.0, atol=0.0)
    assert not np.allclose(a_first, a_second)
    assert a_second_result.metadata["session_id"] == "request-a"
    session_a.close()
    with pytest.raises(RuntimeError, match="closed"):
        session_a.step(inputs[0])
    session_b.close()

    for name, tensor in model.model_.state_dict().items():
        torch.testing.assert_close(tensor, shared_before[name], rtol=0.0, atol=0.0)


def test_non_stateful_models_reject_sessions(regression_run):
    run, _ = regression_run
    runtime = psann.create_inference_runtime(run.model)
    with pytest.raises(RuntimeError, match="stateful=True"):
        runtime.create_session()


def test_native_artifact_loads_directly_as_runtime(regression_run, tmp_path: Path):
    run, inputs = regression_run
    artifact = run.export(tmp_path / "deployment.psann")

    runtime = psann.load_runtime(
        artifact,
        config=InferenceConfig(batch_size=4, device="cpu"),
    )
    result = runtime.predict(inputs)

    np.testing.assert_allclose(
        result.values,
        run.model.predict(inputs),
        rtol=1e-6,
        atol=1e-6,
    )
    assert result.artifact_version == "1.0"
    assert result.model_id
    assert result.run_id == run.run_id
    assert runtime.metadata()["stateful"] is False


def test_runtime_rejects_unsupported_dtype(regression_run):
    run, _ = regression_run
    with pytest.raises(ValueError, match="float32"):
        psann.create_inference_runtime(run.model, config=InferenceConfig(dtype="float64"))


def test_device_pool_uses_independent_round_robin_runtimes(regression_run, tmp_path: Path):
    run, inputs = regression_run
    artifact = run.export(tmp_path / "pool.psann")
    pool = psann.load_runtime_pool(
        artifact,
        devices=("cpu", "cpu"),
        config=InferenceConfig(batch_size=4),
    )

    first = pool.predict(inputs[:3])
    second = pool.predict(inputs[:3])
    third = pool.predict(inputs[:3])

    assert [first.metadata["pool_index"], second.metadata["pool_index"]] == [0, 1]
    assert third.metadata["pool_index"] == 0
    assert pool.metadata()["devices"] == ["cpu", "cpu"]
    assert pool.runtimes[0].model is not pool.runtimes[1].model
    np.testing.assert_allclose(first.values, second.values, rtol=0.0, atol=0.0)


def test_device_pool_rejects_empty_device_set(regression_run, tmp_path: Path):
    run, _ = regression_run
    artifact = run.export(tmp_path / "empty-pool.psann")
    with pytest.raises(ValueError, match="at least one"):
        psann.load_runtime_pool(artifact, devices=())


def test_explicit_internal_registry_resolver_loads_runtime(regression_run, tmp_path: Path):
    run, inputs = regression_run
    artifact = run.export(tmp_path / "registry.psann")
    identifier = f"testregistry{tmp_path.name.lower().replace('-', '')}"
    psann.register_artifact_resolver(
        identifier,
        lambda reference: (
            artifact
            if reference == f"{identifier}://models/demand/7"
            else tmp_path / "missing.psann"
        ),
    )

    runtime = psann.load_registry_runtime(
        f"{identifier}://models/demand/7",
        config=InferenceConfig(device="cpu"),
    )

    np.testing.assert_allclose(
        runtime.predict(inputs).values,
        run.model.predict(inputs),
        rtol=1e-6,
        atol=1e-6,
    )
    assert psann.resolve_artifact(artifact) == artifact.resolve()
