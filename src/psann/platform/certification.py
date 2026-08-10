"""Executable workplace release-candidate certification scenarios.

This module intentionally ships in the wheel so release automation can validate the
installed distribution rather than importing from a source checkout.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from .._version import __version__
from .artifact_schema import ARTIFACT_FORMAT_VERSION
from .artifacts import inspect_artifact, load_model
from .contracts import InferenceResult
from .explainability import ExplainerConfig, ExplanationCapabilityError, explain
from .inference import create_inference_runtime, load_runtime
from .lifecycle import SupervisedData, create_model, train
from .operations import validate_no_secrets
from .registry import register_backbone
from .specs import DataSchema, InferenceConfig, ModelSpec, TaskSpec, TrainingConfig

Scenario = Callable[[Path, str, int], Mapping[str, Any]]


@dataclass(frozen=True)
class ScenarioResult:
    """Privacy-safe evidence for one end-to-end scenario."""

    name: str
    status: str
    duration_seconds: float
    evidence: Mapping[str, Any]


def _parameters(seed: int, **extra: Any) -> dict[str, Any]:
    return {
        "hidden_layers": 1,
        "hidden_units": 4,
        "random_state": seed,
        **extra,
    }


def _training(
    device: str,
    *,
    epochs: int = 1,
    **extra: Any,
) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=4,
        device=device,
        deterministic=True,
        fallback_policy="error",
        **extra,
    )


def _maximum_error(observed: Any, expected: Any) -> float:
    left = np.asarray(observed, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise AssertionError(f"Parity shape mismatch: {left.shape!r} != {right.shape!r}.")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _assert_parity(observed: Any, expected: Any, *, atol: float = 1e-5) -> float:
    error = _maximum_error(observed, expected)
    if error > atol:
        raise AssertionError(f"Deployment parity error {error:.8g} exceeds {atol:.8g}.")
    return error


def _soak(runtime: Any, inputs: Any, iterations: int, **kwargs: Any) -> InferenceResult:
    reference = runtime.predict(inputs, **kwargs)
    for _ in range(max(0, iterations - 1)):
        observed = runtime.predict(inputs, **kwargs)
        _assert_parity(observed.values, reference.values, atol=0.0)
    return reference


def _artifact_evidence(path: Path) -> dict[str, Any]:
    info = inspect_artifact(path)
    return {
        "artifact_format_version": info.artifact_format_version,
        "backbone": info.backbone,
        "capabilities": list(info.capabilities),
        "experimental": info.experimental,
        "package_version": info.package_version,
        "task": info.task,
    }


def _tabular_regression(root: Path, device: str, soak_iterations: int) -> Mapping[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency-light installations
        raise RuntimeError("Tabular certification requires pandas.") from exc

    rng = np.random.default_rng(801)
    inputs = pd.DataFrame(
        rng.normal(size=(24, 3)).astype(np.float32),
        columns=["amount", "tenure", "utilization"],
    )
    targets = pd.Series(
        (1.5 * inputs["amount"] - 0.4 * inputs["utilization"]).to_numpy(np.float32),
        name="demand",
    )
    spec = ModelSpec(
        input_schema=DataSchema(
            feature_names=tuple(inputs.columns),
            output_names=("demand",),
            input_shape=(3,),
            target_scaling={"kind": "standard"},
        ),
        parameters=_parameters(801, scaler="standard", target_scaler="standard"),
    )
    checkpoint_dir = root / "regression-checkpoints"
    initial = train(
        create_model(spec),
        (inputs, targets),
        validation_data=(inputs.iloc[-6:], targets.iloc[-6:]),
        config=_training(
            device,
            epochs=1,
            early_stopping=True,
            patience=2,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=1,
        ),
    )
    checkpoint = checkpoint_dir / "latest.psann-train"
    if not checkpoint.is_file():
        raise AssertionError("Regression certification did not write a resume checkpoint.")
    resumed = train(
        create_model(spec),
        (inputs, targets),
        validation_data=(inputs.iloc[-6:], targets.iloc[-6:]),
        config=_training(
            device,
            epochs=2,
            early_stopping=True,
            patience=2,
            resume_from=str(checkpoint),
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=1,
        ),
    )
    artifact = resumed.export(root / "tabular-regression.psann")
    runtime = load_runtime(
        artifact,
        config=InferenceConfig(batch_size=5, device=device, fallback_policy="error"),
    )
    result = _soak(runtime, inputs, soak_iterations)
    parity = _assert_parity(result.values, resumed.model.predict(inputs))
    explanation = explain(
        runtime,
        inputs.iloc[:1],
        background=inputs.iloc[1:5],
        config=ExplainerConfig(max_evaluations=20, seed=801),
    )
    explanation_feature_names = tuple(explanation.metadata["feature_names"])
    if explanation_feature_names != tuple(inputs.columns):
        raise AssertionError("Regression explanation lost named pandas features.")
    return {
        **_artifact_evidence(artifact),
        "batch_chunks": result.metadata["chunks"],
        "checkpoint_epoch_count": len(resumed.history),
        "early_stopping_enabled": resumed.training_config.early_stopping,
        "explanation_shape": list(explanation.values.shape),
        "feature_names": list(explanation_feature_names),
        "initial_epoch_count": len(initial.history),
        "max_artifact_parity_error": parity,
        "resume_checkpoint_format": ".psann-train",
        "target_scaler": getattr(resumed.model, "preprocessing_contract_")["target_scaler"]["kind"],
    }


def _binary_classification(root: Path, device: str, soak_iterations: int) -> Mapping[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. "
                "Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead."
            ),
            category=DeprecationWarning,
        )
        from fastapi.testclient import TestClient

    from ..serving import create_app

    rng = np.random.default_rng(802)
    inputs = rng.normal(size=(24, 3)).astype(np.float32)
    targets = np.where(inputs[:, 0] - 0.3 * inputs[:, 1] > 0, "approved", "denied")
    spec = ModelSpec(
        task=TaskSpec(kind="binary", threshold=0.65, positive_label="approved"),
        input_schema=DataSchema(
            feature_names=("income", "risk", "tenure"),
            input_shape=(3,),
        ),
        parameters=_parameters(802),
    )
    run = train(create_model(spec), (inputs, targets), config=_training(device))
    artifact = run.export(root / "binary-classification.psann")
    runtime = load_runtime(
        artifact,
        config=InferenceConfig(batch_size=4, device=device, fallback_policy="error"),
    )
    result = _soak(runtime, inputs[:7], soak_iterations)
    probabilities = np.asarray(result.values)
    if probabilities.shape != (7, 2):
        raise AssertionError(f"Binary probabilities have shape {probabilities.shape!r}.")
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    labels = create_inference_runtime(
        load_model(artifact, device=device),
        config=InferenceConfig(
            classification_output="label",
            device=device,
            fallback_policy="error",
        ),
    ).predict(inputs[:7])
    client = TestClient(create_app(runtime=runtime))
    response = client.post("/predict", json={"inputs": inputs[:7].tolist(), "batch_size": 3})
    if response.status_code != 200:
        raise AssertionError(f"Binary service inference failed with HTTP {response.status_code}.")
    service_values = response.json()["values"]
    parity = _assert_parity(service_values, probabilities)
    explanation = explain(
        runtime,
        inputs[:1],
        background=inputs[1:5],
        config=ExplainerConfig(
            output_kind="probability",
            output="approved",
            max_evaluations=20,
            seed=802,
        ),
    )
    return {
        **_artifact_evidence(artifact),
        "classification_metrics": dict(run.metrics),
        "explanation_output_names": list(explanation.output_names),
        "explanation_shape": list(explanation.values.shape),
        "label_count": int(np.asarray(labels.values).shape[0]),
        "max_service_parity_error": parity,
        "probability_shape": list(probabilities.shape),
        "service_status": response.status_code,
        "threshold": 0.65,
    }


def _multiclass_classification(
    root: Path,
    device: str,
    soak_iterations: int,
) -> Mapping[str, Any]:
    rng = np.random.default_rng(803)
    inputs = rng.normal(size=(24, 4)).astype(np.float32)
    indices = np.argmax(inputs[:, :3], axis=1)
    classes = np.asarray(["bronze", "silver", "gold"], dtype=object)
    targets = classes[indices]
    spec = ModelSpec(
        task=TaskSpec(kind="multiclass"),
        input_schema=DataSchema(
            feature_names=("margin", "growth", "quality", "leverage"),
            input_shape=(4,),
        ),
        parameters=_parameters(803),
    )
    run = train(create_model(spec), (inputs, targets), config=_training(device))
    artifact = run.export(root / "multiclass-classification.psann")
    runtime = load_runtime(
        artifact,
        config=InferenceConfig(
            batch_size=3,
            device=device,
            top_k=2,
            fallback_policy="error",
        ),
    )
    result = _soak(runtime, inputs[:8], soak_iterations)
    if result.top_k is None:
        raise AssertionError("Multiclass certification did not return top-k output.")
    probabilities = np.asarray(result.values)
    parity = _assert_parity(
        probabilities,
        getattr(run.model, "predict_proba")(inputs[:8]),
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    explanation = explain(
        runtime,
        inputs[:1],
        background=inputs[1:5],
        config=ExplainerConfig(max_evaluations=24, seed=803),
    )
    if explanation.values.shape[-1] != 3:
        raise AssertionError("Multiclass explanation did not preserve every class output.")
    return {
        **_artifact_evidence(artifact),
        "class_labels": list(result.output_names),
        "explanation_output_names": list(explanation.output_names),
        "explanation_shape": list(explanation.values.shape),
        "max_artifact_parity_error": parity,
        "probability_shape": list(probabilities.shape),
        "top_k": 2,
        "top_k_shape": list(np.asarray(result.top_k.labels).shape),
    }


def _convolutional(root: Path, device: str, soak_iterations: int) -> Mapping[str, Any]:
    rng = np.random.default_rng(804)
    inputs = rng.normal(size=(15, 2, 3)).astype(np.float32)
    targets = (inputs[:, 0].mean(axis=1) - inputs[:, 1].mean(axis=1)).astype(np.float32)
    spec = ModelSpec(
        backbone="psann_conv1d",
        input_schema=DataSchema(input_shape=(2, 3), output_names=("signal",)),
        parameters=_parameters(804),
    )
    run = train(create_model(spec), (inputs, targets), config=_training(device))
    artifact = run.export(root / "convolutional.psann")
    runtime = load_runtime(
        artifact,
        config=InferenceConfig(batch_size=2, device=device, fallback_policy="error"),
    )
    result = _soak(runtime, inputs, soak_iterations)
    parity = _assert_parity(result.values, run.model.predict(inputs))
    if result.metadata["chunks"] != 8:
        raise AssertionError("Convolutional inference did not honor bounded batch size.")
    explanation = explain(
        runtime,
        inputs[:1],
        background=inputs[1:5],
        config=ExplainerConfig(max_evaluations=32, seed=804),
    )
    if explanation.values.shape[1:-1] != inputs.shape[1:]:
        raise AssertionError("Convolutional explanation did not preserve spatial shape.")
    return {
        **_artifact_evidence(artifact),
        "batch_chunks": result.metadata["chunks"],
        "explanation_group_strategy": explanation.metadata["group_strategy"],
        "explanation_groups": len(explanation.feature_groups),
        "explanation_shape": list(explanation.values.shape),
        "input_shape": list(inputs.shape[1:]),
        "max_artifact_parity_error": parity,
    }


def _sequence_context(root: Path, device: str, soak_iterations: int) -> Mapping[str, Any]:
    rng = np.random.default_rng(805)
    inputs = rng.normal(size=(16, 3)).astype(np.float32)
    context = rng.normal(size=(16, 2)).astype(np.float32)
    targets = (inputs[:, 0] + 0.2 * context[:, 0]).astype(np.float32)
    spec = ModelSpec(
        backbone="wave_resnet",
        input_schema=DataSchema(input_shape=(3,), output_names=("state",)),
        parameters=_parameters(805, context_dim=2),
    )
    run = train(
        create_model(spec),
        SupervisedData(inputs, targets, context),
        config=_training(device),
    )
    artifact = run.export(root / "sequence-context.psann")
    runtime = load_runtime(
        artifact,
        config=InferenceConfig(batch_size=3, device=device, fallback_policy="error"),
    )
    result = _soak(runtime, inputs[:7], soak_iterations, context=context[:7])
    parity = _assert_parity(
        result.values,
        run.model.predict(inputs[:7], context=context[:7]),
    )
    state_spec = ModelSpec(
        input_schema=DataSchema(input_shape=(3,), output_names=("state",)),
        parameters=_parameters(
            815,
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
    state_run = train(
        create_model(state_spec),
        (inputs, targets),
        config=_training(device),
    )
    state_runtime = create_inference_runtime(
        state_run.model,
        config=InferenceConfig(device=device, fallback_policy="error"),
    )
    first_session = state_runtime.create_session(session_id="cert-a")
    second_session = state_runtime.create_session(session_id="cert-b")
    try:
        first = first_session.step(inputs[0]).values
        second = second_session.step(inputs[0]).values
        isolation_error = _assert_parity(first, second, atol=0.0)
    finally:
        first_session.close()
        second_session.close()
    try:
        explain(
            runtime,
            inputs[:1],
            background=inputs[1:5],
            config=ExplainerConfig(max_evaluations=20, seed=805),
        )
    except ExplanationCapabilityError as exc:
        if "explicit context" not in str(exc):
            raise
        explanation_behavior = "explicit_capability_error"
    else:
        raise AssertionError(
            "Explicit-context explanation unexpectedly bypassed its capability gate."
        )
    return {
        **_artifact_evidence(artifact),
        "context_shape": [2],
        "explanation_behavior": explanation_behavior,
        "max_artifact_parity_error": parity,
        "session_isolation_error": isolation_error,
        "stateful": state_runtime.is_stateful,
    }


def _custom_registered(root: Path, device: str, soak_iterations: int) -> Mapping[str, Any]:
    class RegisteredLinear(torch.nn.Module):
        def __init__(self, input_dim: int, output_dim: int) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    def factory(parameters: Mapping[str, Any]) -> torch.nn.Module:
        return RegisteredLinear(
            int(parameters["input_dim"]),
            int(parameters["output_dim"]),
        )

    registration = register_backbone(
        "certification.registered_linear",
        factory,
        supported_tasks=("regression",),
        input_ranks=(1,),
        activations=("relu",),
        factory_kind="torch_module",
        experimental=True,
        plugin="psann-certification-fixture",
        plugin_version=__version__,
        replace=True,
    )
    rng = np.random.default_rng(806)
    inputs = rng.normal(size=(20, 2)).astype(np.float32)
    targets = (inputs[:, :1] - 0.5 * inputs[:, 1:2]).astype(np.float32)
    spec = ModelSpec(
        backbone=registration.identifier,
        activation="relu",
        input_schema=DataSchema(input_shape=(2,), output_names=("custom",)),
        parameters={"input_dim": 2, "output_dim": 1},
    )
    run = train(create_model(spec), (inputs, targets), config=_training(device))
    artifact = run.export(root / "custom-registered.psann")
    runtime = load_runtime(
        artifact,
        config=InferenceConfig(batch_size=4, device=device, fallback_policy="error"),
    )
    result = _soak(runtime, inputs, soak_iterations)
    parity = _assert_parity(result.values, run.model.predict(inputs))
    info = inspect_artifact(artifact)
    if not info.experimental:
        raise AssertionError("Custom registered backbone lost its experimental support tier.")
    if "gradient_explanations" in info.capabilities:
        raise AssertionError("Custom registered backbone overclaimed gradient explanations.")
    return {
        **_artifact_evidence(artifact),
        "derived_export_support": "not_guaranteed",
        "max_artifact_parity_error": parity,
        "native_artifact_support": "registered_factory_only",
        "support_tier": "experimental",
    }


SCENARIOS: Mapping[str, Scenario] = {
    "tabular_regression": _tabular_regression,
    "binary_classification": _binary_classification,
    "multiclass_classification": _multiclass_classification,
    "convolutional": _convolutional,
    "sequence_context": _sequence_context,
    "custom_registered_backbone": _custom_registered,
}


def run_certification(
    output_dir: str | Path,
    *,
    device: str = "cpu",
    scenarios: Sequence[str] | None = None,
    soak_iterations: int = 1,
) -> Mapping[str, Any]:
    """Run selected scenarios and write a privacy-safe JSON evidence report."""

    requested_device = str(device).strip().lower()
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("Certification device must be 'cpu' or 'cuda'.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA certification requires an available CUDA device.")
    if soak_iterations < 1:
        raise ValueError("soak_iterations must be >= 1.")
    selected = tuple(scenarios or SCENARIOS)
    unknown = sorted(set(selected) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown certification scenarios: {unknown!r}.")

    destination = Path(output_dir).resolve()
    artifact_dir = destination / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    results: list[ScenarioResult] = []
    for name in selected:
        scenario_started = time.perf_counter()
        evidence = SCENARIOS[name](artifact_dir, requested_device, soak_iterations)
        results.append(
            ScenarioResult(
                name=name,
                status="passed",
                duration_seconds=round(time.perf_counter() - scenario_started, 6),
                evidence=evidence,
            )
        )
    completed = datetime.now(timezone.utc)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "passed",
        "package_version": __version__,
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "device": requested_device,
        "soak_iterations": soak_iterations,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": round((completed - started).total_seconds(), 6),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "privacy": {
            "contains_raw_inputs": False,
            "contains_targets": False,
            "contains_row_level_attributions": False,
        },
        "scenarios": [asdict(result) for result in results],
    }
    validate_no_secrets(report, field="certification_report")
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / f"workplace-certification-{requested_device}.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, help="Directory for artifacts and JSON evidence."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIOS),
        help="Run only this scenario; repeat to select more than one.",
    )
    parser.add_argument("--soak-iterations", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = run_certification(
        arguments.output,
        device=arguments.device,
        scenarios=arguments.scenario,
        soak_iterations=arguments.soak_iterations,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "device": report["device"],
                "package_version": report["package_version"],
                "scenarios": [item["name"] for item in report["scenarios"]],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI execution
    raise SystemExit(main())


__all__ = ["SCENARIOS", "ScenarioResult", "main", "run_certification"]
