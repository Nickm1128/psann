from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch

import psann.training_checkpoint as checkpoint_module
from psann import PSANNRegressor, TrainingCheckpointError
from psann.training_checkpoint import load_training_checkpoint, save_training_checkpoint


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(-1, 1)
    y = (0.75 * X + 0.2).astype(np.float32)
    return X, y


def _estimator() -> PSANNRegressor:
    return PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=4,
        batch_size=8,
        random_state=17,
        optimizer="adam",
    )


def test_interrupted_training_resumes_with_exact_cpu_continuity(tmp_path):
    X, y = _data()
    baseline = _estimator()
    baseline.fit(X, y, deterministic=True)

    checkpoint_dir = tmp_path / "checkpoints"

    def interrupt_after_second_epoch(event) -> None:
        if event.name == "checkpoint" and event.epoch == 2 and event.data.get("kind") == "latest":
            raise RuntimeError("simulated interruption")

    interrupted = _estimator()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        interrupted.fit(
            X,
            y,
            deterministic=True,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=1,
            callbacks=[interrupt_after_second_epoch],
        )

    latest = checkpoint_dir / "latest.psann-train"
    assert latest.exists()
    resumed = _estimator()
    resumed.fit(
        X,
        y,
        deterministic=True,
        resume_from=latest,
    )

    assert [entry["train_loss"] for entry in resumed.history_] == pytest.approx(
        [entry["train_loss"] for entry in baseline.history_],
        rel=0,
        abs=0,
    )
    for name, expected in baseline.model_.state_dict().items():
        assert torch.equal(resumed.model_.state_dict()[name], expected)


def test_checkpoint_contains_complete_resume_state_and_is_not_deployment_snapshot(
    tmp_path,
):
    X, y = _data()
    checkpoint_dir = tmp_path / "checkpoints"
    estimator = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=2,
        batch_size=8,
        random_state=11,
        early_stopping=True,
        patience=2,
    )
    estimator.fit(
        X,
        y,
        deterministic=True,
        scheduler="cosine",
        scheduler_params={"t_max": 2, "eta_min": 1e-5},
        checkpoint_dir=checkpoint_dir,
    )

    latest = checkpoint_dir / "latest.psann-train"
    state = load_training_checkpoint(latest)
    required = {
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "amp_scaler_state",
        "epoch",
        "global_step",
        "best_metric",
        "best_epoch",
        "patience_left",
        "history",
        "rng_state",
        "data_signature",
        "model_signature",
    }
    assert required.issubset(state)
    assert state["scheduler_state"] is not None
    assert state["rng_state"]["data_loader"] is not None

    with pytest.raises(ValueError, match="resume_from"):
        PSANNRegressor.load(str(latest))


def test_checkpoint_checksum_detects_corruption(tmp_path):
    X, y = _data()
    checkpoint_dir = tmp_path / "checkpoints"
    estimator = PSANNRegressor(epochs=1, hidden_layers=1, hidden_units=8)
    estimator.fit(X, y, checkpoint_dir=checkpoint_dir)
    source = checkpoint_dir / "latest.psann-train"
    corrupt = checkpoint_dir / "corrupt.psann-train"

    with zipfile.ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["state.pt"] = members["state.pt"] + b"corrupt"
    with zipfile.ZipFile(corrupt, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)

    with pytest.raises(TrainingCheckpointError, match="checksum mismatch"):
        load_training_checkpoint(corrupt)


def _checkpoint_members(path):
    with zipfile.ZipFile(path, "r") as archive:
        return [(info.filename, archive.read(info.filename)) for info in archive.infolist()]


def test_checkpoint_rejects_duplicate_and_unexpected_members(tmp_path):
    source = save_training_checkpoint(
        tmp_path / "source.psann-train",
        {"model_state": {"weight": torch.ones(1)}},
    )
    members = _checkpoint_members(source)

    duplicate = tmp_path / "duplicate.psann-train"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in members:
                archive.writestr(name, content)
            archive.writestr("state.pt", dict(members)["state.pt"])
    with pytest.raises(TrainingCheckpointError, match="duplicate members"):
        load_training_checkpoint(duplicate)
    assert checkpoint_module.is_training_checkpoint(duplicate) is False

    unexpected = tmp_path / "unexpected.psann-train"
    with zipfile.ZipFile(unexpected, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members:
            archive.writestr(name, content)
        archive.writestr("notes.txt", b"not part of the fixed format")
    with pytest.raises(TrainingCheckpointError, match="unexpected member"):
        load_training_checkpoint(unexpected)


def test_checkpoint_rejects_oversized_compressed_state_and_metadata(
    tmp_path,
    monkeypatch,
):
    source = save_training_checkpoint(
        tmp_path / "compressed.psann-train",
        {"large_zeros": torch.zeros(512 * 1024, dtype=torch.float32)},
    )
    with zipfile.ZipFile(source, "r") as archive:
        info = archive.getinfo("state.pt")
        assert info.file_size > info.compress_size * 100

    monkeypatch.setattr(checkpoint_module, "_MAX_STATE_BYTES", 1024 * 1024)
    with pytest.raises(TrainingCheckpointError, match="safe size limit"):
        load_training_checkpoint(source)

    monkeypatch.setattr(checkpoint_module, "_MAX_STATE_BYTES", 2 * 1024 * 1024 * 1024)
    monkeypatch.setattr(checkpoint_module, "_MAX_METADATA_BYTES", 32)
    with pytest.raises(TrainingCheckpointError, match="safe size limit"):
        load_training_checkpoint(source)


def test_checkpoint_rejects_total_size_and_malformed_checksum_metadata(
    tmp_path,
    monkeypatch,
):
    source = save_training_checkpoint(
        tmp_path / "source.psann-train",
        {"model_state": {"weight": torch.ones(1)}},
    )
    monkeypatch.setattr(checkpoint_module, "_MAX_TOTAL_BYTES", 1)
    with pytest.raises(TrainingCheckpointError, match="total-size"):
        load_training_checkpoint(source)

    monkeypatch.setattr(
        checkpoint_module,
        "_MAX_TOTAL_BYTES",
        checkpoint_module._MAX_STATE_BYTES + 2 * checkpoint_module._MAX_METADATA_BYTES,
    )
    members = dict(_checkpoint_members(source))
    manifest = json.loads(members["manifest.json"])
    manifest["state_sha256"] = "not-a-sha256"
    members["manifest.json"] = json.dumps(manifest).encode("utf-8")
    members["checksums.sha256"] = b"not-a-sha256  state.pt\n"
    malformed = tmp_path / "malformed.psann-train"
    with zipfile.ZipFile(malformed, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(TrainingCheckpointError, match="invalid state SHA-256"):
        load_training_checkpoint(malformed)


def test_valid_large_checkpoint_remains_loadable(tmp_path):
    expected = torch.arange(256 * 1024, dtype=torch.float32)
    checkpoint = save_training_checkpoint(
        tmp_path / "large-valid.psann-train",
        {"model_state": {"weight": expected}},
    )

    loaded = load_training_checkpoint(checkpoint)

    assert torch.equal(loaded["model_state"]["weight"], expected)


def test_checkpoint_retention_keeps_latest_best_and_bounded_periodic_files(tmp_path):
    X, y = _data()
    checkpoint_dir = tmp_path / "checkpoints"
    estimator = _estimator()
    estimator.fit(
        X,
        y,
        deterministic=True,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1,
        checkpoint_keep=2,
    )

    assert (checkpoint_dir / "latest.psann-train").exists()
    assert (checkpoint_dir / "best.psann-train").exists()
    periodic = sorted(checkpoint_dir.glob("epoch_*.psann-train"))
    assert [path.name for path in periodic] == [
        "epoch_000003.psann-train",
        "epoch_000004.psann-train",
    ]
    assert not list(checkpoint_dir.glob("*.tmp"))


def test_resume_rejects_changed_training_data_and_determinism(tmp_path):
    X, y = _data()
    checkpoint_dir = tmp_path / "checkpoints"
    estimator = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=8,
        random_state=3,
    )
    estimator.fit(X, y, deterministic=True, checkpoint_dir=checkpoint_dir)
    latest = checkpoint_dir / "latest.psann-train"

    changed = _estimator()
    X_changed = X.copy()
    X_changed[0, 0] += 0.01
    with pytest.raises(TrainingCheckpointError, match="data signature"):
        changed.fit(
            X_changed,
            y,
            deterministic=True,
            resume_from=latest,
        )

    mode_mismatch = _estimator()
    with pytest.raises(TrainingCheckpointError, match="deterministic mode"):
        mode_mismatch.fit(X, y, deterministic=False, resume_from=latest)
