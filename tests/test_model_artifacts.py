from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import torch

import psann
from psann.platform import (
    ARTIFACT_FORMAT_VERSION,
    ArtifactChecksumError,
    ArtifactExtensionError,
    ArtifactFormatError,
    ArtifactVersionError,
    DataSchema,
    LegacyCheckpointTrustError,
    LegacyCheckpointWarning,
    ModelSpec,
    TaskSpec,
    TrainingConfig,
    create_model,
    train,
)
from psann.platform.artifact_io import (
    all_payloads,
    inspect_bundle,
    json_bytes,
    parse_json,
    write_bundle,
)
from psann.platform.artifact_schema import (
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    MODEL_CONFIG_PATH,
    WEIGHTS_PATH,
)
from psann.platform.artifacts import _parsed_version
from psann.training_checkpoint import save_training_checkpoint


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        epochs=1,
        batch_size=4,
        deterministic=True,
    )


def _parameters(*, conv: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "hidden_layers": 1,
        "hidden_units": 4,
        "random_state": 7,
    }
    if conv:
        result["conv_channels"] = 4
    return result


@pytest.mark.parametrize(
    ("candidate", "required", "compatible"),
    [
        ("1.0.0rc1", "1.0.0", False),
        ("1.0.0", "1.0.0rc1", True),
        ("1.0.0.post1", "1.0.0", True),
        ("1.0.0.dev1", "1.0.0rc1", False),
        ("2.4.1+cu121", "2.4.1", True),
    ],
)
def test_artifact_version_comparison_uses_pep440(
    candidate: str,
    required: str,
    compatible: bool,
):
    assert (
        _parsed_version(candidate, field="candidate") >= _parsed_version(required, field="required")
    ) is compatible


@pytest.mark.parametrize("invalid", ["1.0.0garbage", "release-next", "1..2"])
def test_artifact_version_comparison_rejects_malformed_versions(invalid: str):
    with pytest.raises(ArtifactVersionError, match="valid PEP 440"):
        _parsed_version(invalid, field="candidate")


@pytest.fixture(scope="module")
def native_artifact(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("native_artifact")
    rng = np.random.default_rng(101)
    inputs = rng.normal(size=(12, 3)).astype(np.float32)
    targets = (inputs[:, 0] - 0.5 * inputs[:, 1]).astype(np.float32)
    spec = ModelSpec(
        input_schema=DataSchema(
            input_shape=(3,),
            output_names=("forecast",),
            preprocessing={"owner": "artifact-test"},
        ),
        activation="gelu",
        parameters={
            **_parameters(),
            "scaler": "standard",
            "target_scaler": "standard",
        },
    )
    run = train(create_model(spec), (inputs, targets), config=_training_config())
    path = run.export(
        directory / "model.psann",
        model_card="# Test model\n",
        metadata={"owner": "tests"},
        registry={"uri": "registry://team/model"},
    )
    return path, inputs, run.model.predict(inputs), run


def _rewrite_valid(
    source: Path,
    destination: Path,
    mutate: Callable[[dict[str, bytes], dict[str, Any]], None],
) -> Path:
    artifact = inspect_bundle(source)
    payloads = all_payloads(artifact)
    manifest = copy.deepcopy(dict(artifact.manifest))
    mutate(payloads, manifest)
    payloads[MANIFEST_PATH] = json_bytes(manifest)
    return write_bundle(destination, payloads)


def _rewrite_raw(
    source: Path,
    destination: Path,
    *,
    member: str,
    payload: bytes,
) -> Path:
    with zipfile.ZipFile(source, mode="r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members[member] = payload
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return destination


def test_manifest_inspection_does_not_deserialize_weights(native_artifact, monkeypatch):
    path, _, _, _ = native_artifact

    def fail_load(*args, **kwargs):
        raise AssertionError("inspection must not call torch.load")

    monkeypatch.setattr("psann.platform.artifact_models.torch.load", fail_load)
    info = psann.inspect_artifact(path)

    assert info.artifact_format_version == ARTIFACT_FORMAT_VERSION
    assert info.backbone == "psann_mlp"
    assert info.task == "regression"
    assert info.run_id
    assert info.manifest["metadata"]["owner"] == "tests"
    assert info.manifest["registry"]["uri"] == "registry://team/model"
    assert psann.is_model_artifact(path)


def test_native_loader_uses_restricted_torch_load(native_artifact, monkeypatch):
    path, inputs, expected, _ = native_artifact
    real_load = torch.load
    calls: list[object] = []

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr("psann.platform.artifact_models.torch.load", recording_load)
    loaded = psann.load_model(path, device="cpu")

    assert calls == [True]
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=0.0, atol=0.0)
    assert loaded.model_.training is False
    assert loaded.artifact_info_.artifact_id == loaded.artifact_id_


def test_checksum_index_covers_manifest_weights_and_model_card(native_artifact):
    path, _, _, _ = native_artifact
    with zipfile.ZipFile(path, mode="r") as archive:
        names = set(archive.namelist())
        records = {
            line.split("  ", 1)[1]
            for line in archive.read(CHECKSUMS_PATH).decode("utf-8").splitlines()
        }
    assert records == names - {CHECKSUMS_PATH}
    assert {MANIFEST_PATH, WEIGHTS_PATH, "model-card.md"} <= records


@pytest.mark.parametrize(
    ("backbone", "shape"),
    [
        ("psann_mlp", (3,)),
        ("respsann_mlp", (3,)),
        ("psann_conv1d", (1, 4)),
        ("psann_conv2d", (1, 3, 3)),
        ("psann_conv3d", (1, 2, 2, 2)),
        ("respsann_conv2d", (1, 3, 3)),
        ("wave_resnet", (3,)),
        ("sgr_psann", (3,)),
    ],
)
def test_every_registered_core_backbone_round_trips(
    tmp_path: Path,
    backbone: str,
    shape: tuple[int, ...],
):
    rng = np.random.default_rng(102)
    inputs = rng.normal(size=(8, *shape)).astype(np.float32)
    targets = rng.normal(size=8).astype(np.float32)
    spec = ModelSpec(
        backbone=backbone,
        input_schema=DataSchema(input_shape=shape),
        parameters=_parameters(conv="conv" in backbone),
    )
    model = create_model(spec)
    run = train(model, (inputs, targets), config=_training_config())
    loaded = psann.load_model(run.export(tmp_path / f"{backbone}.psann"))

    np.testing.assert_allclose(
        loaded.predict(inputs),
        model.predict(inputs),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    ("task", "targets"),
    [
        (TaskSpec(kind="binary", positive_label="yes"), np.asarray(["no", "yes"] * 6)),
        (TaskSpec(kind="multiclass"), np.asarray([0, 1, 2] * 4)),
        (
            TaskSpec(
                kind="multilabel",
                class_names=("risk", "review"),
                threshold=(0.4, 0.6),
            ),
            np.asarray([[0, 1], [1, 0], [1, 1]] * 4),
        ),
    ],
)
def test_classification_task_metadata_and_probabilities_round_trip(
    tmp_path: Path,
    task: TaskSpec,
    targets: np.ndarray,
):
    inputs = np.random.default_rng(103).normal(size=(12, 4)).astype(np.float32)
    model = create_model(
        ModelSpec(
            task=task,
            input_schema=DataSchema(input_shape=(4,)),
            activation="silu",
            parameters=_parameters(),
        )
    )
    run = train(model, (inputs, targets), config=_training_config())
    loaded = psann.load_model(run.export(tmp_path / f"{task.kind}.psann"))

    np.testing.assert_allclose(
        loaded.predict_proba(inputs),
        model.predict_proba(inputs),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(loaded.predict(inputs), model.predict(inputs))
    np.testing.assert_array_equal(loaded.classes_, model.classes_)


def test_named_schema_and_builtin_scaler_state_round_trip(tmp_path: Path):
    pandas = pytest.importorskip("pandas")
    inputs = pandas.DataFrame(
        np.random.default_rng(104).normal(size=(8, 2)),
        columns=["amount", "age"],
    )
    targets = pandas.DataFrame(
        np.random.default_rng(105).normal(size=(8, 2)),
        columns=["low", "high"],
    )
    spec = ModelSpec(
        input_schema=DataSchema(
            feature_names=("amount", "age"),
            output_names=("low", "high"),
            input_shape=(2,),
            feature_policy="reorder",
        ),
        parameters={
            **_parameters(),
            "scaler": "standard",
            "target_scaler": "minmax",
        },
    )
    model = create_model(spec)
    run = train(model, (inputs, targets), config=_training_config())
    loaded = psann.load_model(run.export(tmp_path / "schema.psann"))
    reordered = inputs[["age", "amount"]]

    np.testing.assert_allclose(
        loaded.predict(reordered),
        model.predict(reordered),
        rtol=0.0,
        atol=0.0,
    )
    assert loaded.feature_names_in_.tolist() == ["amount", "age"]
    assert loaded.output_names_.tolist() == ["low", "high"]
    assert loaded.preprocessing_contract_["input_scaler"]["kind"] == "standard"


def test_corrupt_weights_fail_before_deserialization(native_artifact, tmp_path: Path, monkeypatch):
    path, _, _, _ = native_artifact
    corrupted = _rewrite_raw(
        path,
        tmp_path / "corrupt.psann",
        member=WEIGHTS_PATH,
        payload=b"not the original weights",
    )

    def fail_load(*args, **kwargs):
        raise AssertionError("checksum failure must happen before torch.load")

    monkeypatch.setattr("psann.platform.artifact_models.torch.load", fail_load)
    with pytest.raises(ArtifactChecksumError, match="checksum mismatch"):
        psann.load_model(corrupted)


def test_truncated_and_incomplete_bundles_fail_closed(native_artifact, tmp_path: Path):
    path, _, _, _ = native_artifact
    truncated = tmp_path / "truncated.psann"
    truncated.write_bytes(path.read_bytes()[:64])
    with pytest.raises(ArtifactFormatError):
        psann.load_model(truncated)

    incomplete = tmp_path / "incomplete.psann"
    with zipfile.ZipFile(incomplete, mode="w") as archive:
        archive.writestr(MANIFEST_PATH, b"{}")
    with pytest.raises(ArtifactFormatError, match="incomplete"):
        psann.load_model(incomplete)


def test_future_artifact_and_runtime_versions_are_actionable(native_artifact, tmp_path: Path):
    path, _, _, _ = native_artifact

    def future_format(payloads, manifest):
        manifest["artifact_format_version"] = "99.0"

    future = _rewrite_valid(path, tmp_path / "future.psann", future_format)
    with pytest.raises(ArtifactVersionError, match="newer release"):
        psann.load_model(future)

    def future_runtime(payloads, manifest):
        manifest["requirements"]["psann_min"] = "99.0.0"

    incompatible = _rewrite_valid(
        path,
        tmp_path / "runtime.psann",
        future_runtime,
    )
    with pytest.raises(ArtifactVersionError, match="PSANN >= 99.0.0"):
        psann.load_model(incompatible)


def test_unknown_optional_metadata_is_forward_compatible(native_artifact, tmp_path: Path):
    path, inputs, expected, _ = native_artifact

    def add_metadata(payloads, manifest):
        manifest["future_optional_metadata"] = {"trace": "ignored"}

    rewritten = _rewrite_valid(path, tmp_path / "optional.psann", add_metadata)
    loaded = psann.load_model(rewritten)
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=0.0, atol=0.0)


def test_missing_registered_backbone_plugin_fails_actionably(native_artifact, tmp_path: Path):
    path, _, _, _ = native_artifact

    def require_plugin(payloads, manifest):
        model_config = dict(parse_json(payloads[MODEL_CONFIG_PATH], member=MODEL_CONFIG_PATH))
        model_config["backbone"] = "acme.forecaster"
        payloads[MODEL_CONFIG_PATH] = json_bytes(model_config)
        manifest["model"]["backbone"] = "acme.forecaster"
        manifest["model"]["plugin"] = {"identifier": "acme-psann", "version": "1.0"}
        manifest["required_extensions"] = [
            {
                "kind": "backbone_plugin",
                "identifier": "acme-psann",
                "version": "1.0",
            }
        ]

    plugin_artifact = _rewrite_valid(
        path,
        tmp_path / "plugin.psann",
        require_plugin,
    )
    with pytest.raises(ArtifactExtensionError, match="acme-psann"):
        psann.load_model(plugin_artifact)


def test_synthetic_0_9_manifest_schema_migrates_in_memory_and_can_be_rewritten(
    native_artifact,
    tmp_path: Path,
):
    path, inputs, expected, _ = native_artifact

    def synthetic_prior_schema(payloads, manifest):
        manifest["artifact_format_version"] = "0.9"

    synthetic = _rewrite_valid(
        path,
        tmp_path / "synthetic-format-0.9.psann",
        synthetic_prior_schema,
    )
    info = psann.inspect_artifact(synthetic)
    loaded = psann.load_model(synthetic)
    migrated = psann.migrate_artifact(synthetic, tmp_path / "migrated.psann")

    assert info.original_format_version == "0.9"
    assert info.migrations == ("0.9",)
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=0.0, atol=0.0)
    assert psann.inspect_artifact(migrated).original_format_version == "1.0"


@pytest.mark.parametrize(
    "producer_version",
    ("0.13.0", "0.14.0", "0.15.0", "0.16.0"),
)
def test_synthetic_producer_version_metadata_does_not_change_format_semantics(
    native_artifact,
    tmp_path: Path,
    producer_version: str,
):
    """Exercise manifest metadata only; these are not historical producer fixtures."""

    path, inputs, expected, _ = native_artifact

    def prior_producer(payloads, manifest):
        manifest["package"]["version"] = producer_version
        manifest["requirements"]["psann_min"] = producer_version

    synthetic = _rewrite_valid(
        path,
        tmp_path / f"synthetic-producer-metadata-{producer_version}.psann",
        prior_producer,
    )
    info = psann.inspect_artifact(synthetic)
    loaded = psann.load_model(synthetic)

    assert info.package_version == producer_version
    assert info.artifact_format_version == "1.0"
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=0.0, atol=0.0)


def test_training_checkpoint_is_never_accepted_by_deployment_loader(tmp_path: Path):
    checkpoint = save_training_checkpoint(
        tmp_path / "latest.psann-train",
        {"model_state": {"weight": torch.ones(1)}},
    )
    with pytest.raises(ArtifactFormatError, match="not a deployment artifact"):
        psann.load_model(checkpoint)


def test_arbitrary_modules_and_custom_scaler_objects_fail_closed(tmp_path: Path):
    inputs = np.ones((4, 2), dtype=np.float32)
    targets = np.ones(4, dtype=np.float32)
    adapter = psann.adapt_module(
        torch.nn.Linear(2, 1),
        epochs=1,
        batch_size=2,
    ).fit(inputs, targets)
    with pytest.raises(ArtifactExtensionError, match="Arbitrary"):
        psann.platform.export_model(
            adapter,
            tmp_path / "adapter.psann",
            model_spec=ModelSpec(input_schema=DataSchema(input_shape=(2,))),
        )

    class CustomScaler:
        def fit(self, values):
            return self

        def transform(self, values):
            return values

    estimator = psann.PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=2,
        scaler=CustomScaler(),
    ).fit(inputs, targets)
    with pytest.raises(ArtifactExtensionError, match="Custom input scaler"):
        psann.platform.export_model(
            estimator,
            tmp_path / "custom.psann",
            model_spec=ModelSpec(input_schema=DataSchema(input_shape=(2,))),
        )


def test_legacy_loading_and_migration_require_explicit_trust(tmp_path: Path):
    inputs = np.random.default_rng(106).normal(size=(8, 2)).astype(np.float32)
    targets = inputs[:, 0]
    estimator = psann.PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=3,
    ).fit(inputs, targets)
    legacy = tmp_path / "legacy.pt"
    with pytest.warns(LegacyCheckpointWarning):
        estimator.save(str(legacy))

    with pytest.raises(LegacyCheckpointTrustError, match="may execute arbitrary Python"):
        psann.load_model(legacy)
    with pytest.warns(LegacyCheckpointWarning):
        trusted = psann.load_model(legacy, trusted_legacy_checkpoint=True)
    np.testing.assert_allclose(trusted.predict(inputs), estimator.predict(inputs))

    with pytest.raises(LegacyCheckpointTrustError):
        psann.migrate_legacy_checkpoint(legacy, tmp_path / "refused.psann")
    with pytest.warns(LegacyCheckpointWarning):
        migrated = psann.migrate_legacy_checkpoint(
            legacy,
            tmp_path / "migrated.psann",
            trusted_legacy_checkpoint=True,
        )
    loaded = psann.load_model(migrated)
    np.testing.assert_allclose(loaded.predict(inputs), estimator.predict(inputs))


def test_public_0_12_7_legacy_checkpoint_loads_and_migrates_with_numerical_parity(
    tmp_path: Path,
):
    fixture_root = Path(__file__).parent / "fixtures" / "legacy"
    provenance = json.loads(
        (fixture_root / "psann-0.12.7-regressor.json").read_text(encoding="utf-8")
    )
    legacy = fixture_root / provenance["fixture"]["filename"]

    assert provenance["producer"]["version"] == "0.12.7"
    assert provenance["producer"]["wheel"]["sha256"] == (
        "43e6bc16a06a27b72e9073d1f80dbac70e07634df4dd01459ab949032997699b"
    )
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == provenance["fixture"]["sha256"]

    inputs = np.asarray(provenance["case"]["inputs"], dtype=np.float32)
    expected = np.asarray(provenance["case"]["expected_predictions"], dtype=np.float32)

    with pytest.raises(LegacyCheckpointTrustError, match="may execute arbitrary Python"):
        psann.load_model(legacy)
    with pytest.warns(LegacyCheckpointWarning):
        trusted = psann.load_model(legacy, trusted_legacy_checkpoint=True)
    np.testing.assert_allclose(trusted.predict(inputs), expected, rtol=1e-6, atol=1e-7)

    with pytest.raises(LegacyCheckpointTrustError):
        psann.migrate_legacy_checkpoint(legacy, tmp_path / "refused.psann")
    with pytest.warns(LegacyCheckpointWarning):
        migrated = psann.migrate_legacy_checkpoint(
            legacy,
            tmp_path / "migrated-from-public-0.12.7.psann",
            trusted_legacy_checkpoint=True,
        )

    info = psann.inspect_artifact(migrated)
    assert info.manifest["metadata"]["migrated_from_legacy_checkpoint"] is True
    loaded = psann.load_model(migrated)
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=1e-6, atol=1e-7)


def test_atomic_export_preserves_existing_target_on_replace_failure(
    native_artifact,
    tmp_path: Path,
    monkeypatch,
):
    _, _, _, run = native_artifact
    target = tmp_path / "existing.psann"
    target.write_bytes(b"existing artifact bytes")

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("psann.platform.artifact_io.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        run.export(target)

    assert target.read_bytes() == b"existing artifact bytes"
    assert list(tmp_path.glob(".existing.psann.*.tmp")) == []


def test_cross_process_cpu_load_parity(native_artifact):
    path, inputs, expected, _ = native_artifact
    repository = Path(__file__).resolve().parents[1]
    code = (
        "import json,numpy as np,psann;"
        f"x=np.asarray({inputs.tolist()!r},dtype=np.float32);"
        f"m=psann.load_model({str(path)!r},device='cpu');"
        "print(json.dumps(m.predict(x).tolist()))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    observed = np.asarray(json.loads(completed.stdout), dtype=np.float32)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
