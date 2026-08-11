from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import psann


def _regression_model(seed: int = 801):
    inputs = np.random.default_rng(seed).normal(size=(12, 3)).astype(np.float32)
    targets = (inputs[:, 0] - inputs[:, 1]).astype(np.float32)
    model = psann.create_model(
        psann.ModelSpec(
            input_schema=psann.DataSchema(input_shape=(3,)),
            parameters={"hidden_layers": 1, "hidden_units": 4, "random_state": seed},
        )
    )
    return model, inputs, targets


def test_accelerator_support_matrix_is_explicit():
    matrix = psann.accelerator_support_matrix()
    stable = {
        (item.device, item.dtype, item.operation, item.amp)
        for item in matrix
        if item.status == "stable"
    }
    assert ("cpu", "float32", "training", False) in stable
    assert ("cuda", "float32", "training", False) in stable
    assert ("cuda", "float16", "training", True) in stable
    assert ("cuda", "bfloat16", "training", True) in stable
    assert (
        psann.accelerator_capability("cpu", "bf16", operation="training", amp=True).status
        == "unsupported"
    )
    assert (
        psann.accelerator_capability("mps", "float32", operation="inference").status
        == "experimental"
    )
    assert (
        psann.accelerator_capability("mps", "float32", operation="export").status == "unsupported"
    )
    assert (
        psann.accelerator_capability(
            "cuda",
            "float16",
            operation="training",
            amp=True,
            compile=True,
        ).status
        == "unsupported"
    )
    assert (
        psann.accelerator_capability("xpu", "float32", operation="training").status == "unsupported"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("fp32", "float32"),
        ("float16", "float16"),
        ("bf16", "bfloat16"),
        (torch.float32, "float32"),
    ],
)
def test_canonical_dtype_aliases(value, expected):
    assert psann.canonical_dtype(value) == expected


def test_training_and_inference_dtype_contracts_round_trip():
    config = psann.TrainingConfig(epochs=1, amp=True, amp_dtype="fp16")
    assert config.amp_dtype == "float16"
    assert psann.TrainingConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        psann.TrainingConfig(epochs=1, amp=True, amp_dtype="float32")
    with pytest.raises(ValueError, match="cannot both be enabled"):
        psann.TrainingConfig(epochs=1, amp=True, compile=True)
    assert psann.InferenceConfig(dtype="fp32").dtype == "float32"
    with pytest.raises(ValueError, match="float32"):
        psann.InferenceConfig(dtype="float16")


def test_unavailable_inference_device_obeys_fallback_policy(monkeypatch):
    model, inputs, targets = _regression_model()
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(epochs=1, batch_size=4),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="fallback_policy='error'"):
        psann.create_inference_runtime(
            run.model,
            config=psann.InferenceConfig(device="cuda", fallback_policy="error"),
        )
    with pytest.warns(RuntimeWarning, match="using CPU"):
        runtime = psann.create_inference_runtime(
            run.model,
            config=psann.InferenceConfig(device="cuda", fallback_policy="warn"),
        )
    assert runtime.device.type == "cpu"
    assert runtime.metadata()["device_fallback"]["requested"] == "cuda"


@pytest.mark.parametrize(
    "runtime_feature",
    ["amp", "compile"],
)
def test_cpu_amp_and_compile_follow_explicit_fallback_policy(runtime_feature: str):
    _, inputs, targets = _regression_model(802)
    warn_model, _, _ = _regression_model(802)
    runtime_options = {runtime_feature: True}
    with pytest.warns(RuntimeWarning, match="fallback"):
        run = psann.train(
            warn_model,
            (inputs, targets),
            config=psann.TrainingConfig(
                epochs=1,
                batch_size=4,
                device="cpu",
                amp_dtype="bf16",
                fallback_policy="warn",
                **runtime_options,
            ),
        )
    fallbacks = [
        event for event in getattr(run.model, "training_events_", ()) if event["name"] == "fallback"
    ]
    assert {event["data"]["component"] for event in fallbacks} == {runtime_feature}

    error_model, _, _ = _regression_model(803)
    with pytest.raises(RuntimeError, match="fallback_policy='error'"):
        psann.train(
            error_model,
            (inputs, targets),
            config=psann.TrainingConfig(
                epochs=1,
                batch_size=4,
                device="cpu",
                amp=True,
                fallback_policy="error",
            ),
        )


def test_workplace_rejects_unsupported_device_and_adapter_runtime_features():
    model = psann.adapt_module(torch.nn.Linear(2, 1), task="regression")
    inputs = np.ones((4, 2), dtype=np.float32)
    targets = np.ones((4, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="outside the workplace support matrix"):
        psann.train(
            model,
            (inputs, targets),
            config=psann.TrainingConfig(epochs=1, device="xpu"),
        )
    with pytest.raises(RuntimeError, match="does not support amp"):
        psann.train(
            model,
            (inputs, targets),
            config=psann.TrainingConfig(
                epochs=1,
                amp=True,
                fallback_policy="error",
            ),
        )


def test_data_and_model_fingerprints_are_deterministic_and_content_sensitive(tmp_path: Path):
    model, inputs, targets = _regression_model(804)
    first = psann.fingerprint_data(inputs, targets)
    second = psann.fingerprint_data(inputs.copy(), targets.copy())
    changed = inputs.copy()
    changed[0, 0] += 1
    assert first == second
    assert first != psann.fingerprint_data(changed, targets)
    assert first.startswith("sha256:")

    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(epochs=1, batch_size=4),
    )
    assert run.metadata["data_fingerprint"] == first
    model_fingerprint = psann.fingerprint_model(model)
    assert run.metadata["model_fingerprint"] == model_fingerprint
    artifact = run.export(tmp_path / "fingerprinted.psann")
    info = psann.inspect_artifact(artifact)
    assert info.manifest["metadata"]["fingerprints"] == {
        "data": first,
        "model": model_fingerprint,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("metadata", {"api_token": "not-even-a-real-secret"}),
        ("registry", {"credential": "secret"}),
        ("model_card", "credential: sk-abcdefghijklmnopqrstuvwxyz123456"),
        ("model_card", "api_key=not-even-a-real-secret"),
    ],
)
def test_artifact_export_rejects_secret_like_content(tmp_path: Path, field, value):
    model, inputs, targets = _regression_model(805)
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(epochs=1, batch_size=4),
    )
    kwargs = {field: value}
    with pytest.raises(psann.OperationalMetadataError, match="prohibited"):
        run.export(tmp_path / f"{field}.psann", **kwargs)


def test_redaction_and_retention_policy_are_serializable():
    value = {
        "owner": "risk",
        "api_key": "sensitive",
        "nested": {"authorization": "Bearer sensitive", "count": np.int64(2)},
    }
    redacted = psann.redact_sensitive(value)
    assert redacted == {
        "owner": "risk",
        "api_key": "[REDACTED]",
        "nested": {"authorization": "[REDACTED]", "count": 2},
    }
    policy = psann.RetentionPolicy(
        history_days=60,
        checkpoint_days=14,
        explanation_days=7,
        service_log_days=7,
    )
    assert psann.RetentionPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(ValueError, match="history_days"):
        psann.RetentionPolicy(history_days=-1)


def test_operational_hooks_receive_redacted_training_and_artifact_events(tmp_path: Path):
    events: list[psann.OperationalEvent] = []
    registry_events: list[psann.OperationalEvent] = []
    hooks = psann.OperationalHooks(
        experiment_tracker=events.append,
        monitor=events.append,
        registry_publisher=registry_events.append,
    )
    model, inputs, targets = _regression_model(806)
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(epochs=1, batch_size=4),
        operational_hooks=hooks,
    )
    run.export(tmp_path / "hooked.psann")
    kinds = [event.kind for event in events]
    assert kinds.count("training_started") == 2
    assert kinds.count("training_completed") == 2
    assert kinds.count("artifact_exported") == 2
    assert [event.kind for event in registry_events] == ["artifact_exported"]
    serialized = json.dumps(
        [
            {"kind": event.kind, "metadata": dict(event.metadata)}
            for event in events + registry_events
        ],
        sort_keys=True,
    )
    assert str(inputs[0, 0]) not in serialized
    assert "data_fingerprint" in serialized


def test_performance_regressions_are_warnings_but_correctness_is_blocking():
    baseline = psann.PerformanceBaseline(
        name="cpu-reference",
        metrics={
            "training_samples_per_second": 100.0,
            "inference_p95_ms": 10.0,
        },
        relative_tolerances={
            "training_samples_per_second": 0.10,
            "inference_p95_ms": 0.20,
        },
    )
    report = psann.compare_performance(
        baseline,
        {
            "training_samples_per_second": 80.0,
            "inference_p95_ms": 15.0,
        },
        correctness_passed=True,
    )
    assert report.blocking_status == "ok"
    assert report.performance_status == "warning"
    assert {item.metric for item in report.regressions} == {
        "training_samples_per_second",
        "inference_p95_ms",
    }
    failed = psann.compare_performance(
        baseline,
        baseline.metrics,
        correctness_passed=False,
    )
    assert failed.blocking_status == "failed"


def test_runtime_accelerator_evidence_contains_no_environment_secrets():
    evidence = psann.runtime_accelerator_evidence()
    assert "torch" in evidence
    assert "matrix" in evidence
    assert psann.sensitive_paths(evidence) == ()
