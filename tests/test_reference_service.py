from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

import psann
from psann.platform import DataSchema, InferenceConfig, ModelSpec, TaskSpec, TrainingConfig

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx2")
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def service_runtime():
    inputs = np.random.default_rng(401).normal(size=(8, 3)).astype(np.float32)
    targets = inputs[:, 0] - inputs[:, 1]
    model = psann.create_model(
        ModelSpec(
            input_schema=DataSchema(input_shape=(3,), output_names=("forecast",)),
            parameters={"hidden_layers": 1, "hidden_units": 4, "random_state": 23},
        )
    )
    psann.train(
        model,
        (inputs, targets),
        config=TrainingConfig(epochs=1, batch_size=4, deterministic=True),
    )
    return (
        psann.create_inference_runtime(
            model,
            config=InferenceConfig(batch_size=2, device="cpu"),
        ),
        inputs,
    )


def test_reference_service_health_ready_metadata_and_prediction(service_runtime):
    runtime, inputs = service_runtime
    client = TestClient(psann.create_app(runtime=runtime))

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}
    metadata = client.get("/metadata").json()
    assert metadata["task"] == "regression"
    assert metadata["output_names"] == ["forecast"]
    response = client.post("/predict", json={"inputs": inputs[:5].tolist()})

    assert response.status_code == 200
    payload = response.json()
    np.testing.assert_allclose(
        payload["values"],
        runtime.predict(inputs[:5]).values,
        rtol=1e-6,
        atol=1e-6,
    )
    assert payload["metadata"]["chunks"] == 3
    assert client.get("/metrics").json()["requests"] == 1
    assert client.get("/metrics").json()["samples"] == 5


def test_reference_service_is_live_but_not_ready_after_load_failure(tmp_path: Path):
    app = psann.create_app(artifact_path=tmp_path / "missing.psann")
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.get("/metadata").status_code == 503
    assert client.post("/predict", json={"inputs": [[1.0]]}).status_code == 503


def test_reference_service_rejects_invalid_request_without_logging_inputs(
    service_runtime,
    caplog,
):
    runtime, _ = service_runtime
    client = TestClient(psann.create_app(runtime=runtime))
    sensitive = [[999999.123]]

    with caplog.at_level(logging.INFO, logger="psann.serving"):
        response = client.post("/predict", json={"inputs": sensitive})

    assert response.status_code == 422
    event = json.loads(caplog.records[-1].message)
    assert event["status"] == "error"
    assert event["batch_size"] == 1
    assert "999999" not in caplog.records[-1].message


def test_reference_service_forbids_unknown_request_fields(service_runtime):
    runtime, inputs = service_runtime
    client = TestClient(psann.create_app(runtime=runtime))

    response = client.post(
        "/predict",
        json={"inputs": inputs[:1].tolist(), "raw_sensitive_payload": "do-not-accept"},
    )

    assert response.status_code == 422


def test_nondefault_binary_threshold_agrees_across_training_artifact_and_service(
    tmp_path: Path,
):
    inputs = np.random.default_rng(402).normal(size=(12, 3)).astype(np.float32)
    targets = np.asarray([0, 1] * 6)
    model = psann.create_model(
        ModelSpec(
            task=TaskSpec(kind="binary", class_names=(0, 1), threshold=0.9),
            input_schema=DataSchema(input_shape=(3,)),
            parameters={"hidden_layers": 1, "hidden_units": 4, "random_state": 29},
        )
    )
    run = psann.train(
        model,
        (inputs, targets),
        validation_data=(inputs, targets),
        config=TrainingConfig(epochs=1, batch_size=4, deterministic=True),
    )
    probabilities = model.predict_proba(inputs)[:, 1]
    expected = (probabilities >= 0.9).astype(np.int64)
    expected_accuracy = float((expected == targets).mean())

    assert run.history[-1]["val_accuracy"] == pytest.approx(expected_accuracy)
    assert run.evaluate((inputs, targets))["accuracy"] == pytest.approx(expected_accuracy)
    assert model.score(inputs, targets) == pytest.approx(expected_accuracy)

    artifact = run.export(tmp_path / "threshold.psann")
    runtime = psann.load_runtime(
        artifact,
        config=InferenceConfig(classification_output="label"),
    )
    np.testing.assert_array_equal(runtime.predict(inputs).values, expected)

    client = TestClient(psann.create_app(runtime=runtime))
    response = client.post("/predict", json={"inputs": inputs.tolist()})
    assert response.status_code == 200
    np.testing.assert_array_equal(response.json()["values"], expected)


def test_container_recipe_uses_non_root_healthcheck_and_mounted_artifact():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "deploy" / "Dockerfile").read_text(encoding="utf-8")
    lock = (root / "constraints" / "deployment-py311.txt").read_text(encoding="utf-8")

    assert "USER psann" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "PSANN_ARTIFACT_PATH=/artifacts/model.psann" in dockerfile
    assert "torch==2.13.0" in dockerfile
    assert "fastapi==0.139.2" in lock
    assert "starlette==1.3.1" in lock
    assert "uvicorn==0.40.0" in lock


def test_release_certification_uses_modern_test_client_in_every_environment():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release-certification.yml").read_text(
        encoding="utf-8"
    )

    assert workflow.count('"httpx2>=2,<3"') == 3
    assert workflow.count("from fastapi.testclient import TestClient") == 3
    assert workflow.count("python -W error -c") == 3
    assert " uvicorn httpx scikit-learn" not in workflow
