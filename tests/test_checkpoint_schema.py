from __future__ import annotations

import numpy as np
import os
import subprocess
import sys
import torch

from psann.estimators import PSANNRegressor
from psann.architectures import (
    ArchitectureConfig,
    ConvolutionConfig,
    GeometryConfig,
    ResidualConfig,
)
from psann.lsm import LSMExpander


def test_schema_v1_round_trip_does_not_store_final_module(tmp_path):
    X = np.ones((8, 2), dtype=np.float32)
    estimator = PSANNRegressor(epochs=1, batch_size=4, random_state=0).fit(
        X, np.ones(8, dtype=np.float32)
    )
    path = tmp_path / "regressor.pt"
    estimator.save(str(path))
    payload = torch.load(path, weights_only=False)
    assert payload["schema"] == "psann.regressor"
    assert payload["schema_version"] == 1
    assert "model" not in payload
    loaded = PSANNRegressor.load(str(path))
    np.testing.assert_allclose(loaded.predict(X[:2]), estimator.predict(X[:2]))


def test_unversioned_phase2_payload_migrates_to_canonical_instance(tmp_path):
    from psann._sklearn.base import PSANNRegressor as Phase2Regressor

    X = np.ones((8, 2), dtype=np.float32)
    old = Phase2Regressor(epochs=1, batch_size=4, random_state=0).fit(
        X, np.ones(8, dtype=np.float32)
    )
    path = tmp_path / "phase2.pt"
    old.save(str(path))
    migrated = PSANNRegressor.load(str(path))
    assert type(migrated) is PSANNRegressor
    np.testing.assert_allclose(migrated.predict(X[:2]), old.predict(X[:2]))


def test_schema_v1_round_trip_reconstructs_lsm_preprocessor(tmp_path):
    X = np.arange(24, dtype=np.float32).reshape(8, 3)
    y = X.mean(axis=1)
    lsm = LSMExpander(output_dim=4, hidden_layers=1, hidden_units=4, epochs=1, batch_size=4)
    lsm.fit(X, epochs=1)
    estimator = PSANNRegressor(
        hidden_layers=1, hidden_units=4, epochs=1, batch_size=4, random_state=0, lsm=lsm
    ).fit(X, y)
    path = tmp_path / "lsm.pt"
    estimator.save(str(path))
    loaded = PSANNRegressor.load(str(path))
    np.testing.assert_allclose(loaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-6)


def test_schema_v1_round_trip_for_every_canonical_architecture(tmp_path):
    cases = (
        ("dense", ArchitectureConfig.dense(), np.ones((8, 2), dtype=np.float32)),
        (
            "residual",
            ArchitectureConfig.dense(residual=ResidualConfig()),
            np.ones((8, 2), dtype=np.float32),
        ),
        ("wave", ArchitectureConfig.for_wave(), np.ones((8, 2), dtype=np.float32)),
        ("sequence", ArchitectureConfig.for_sequence(), np.ones((8, 2), dtype=np.float32)),
        (
            "geo",
            ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(1, 2))),
            np.ones((8, 2), dtype=np.float32),
        ),
        (
            "conv",
            ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=4)),
            np.ones((8, 1, 2), dtype=np.float32),
        ),
    )
    for name, architecture, X in cases:
        estimator = PSANNRegressor(
            architecture=architecture,
            hidden_layers=1,
            hidden_units=4,
            epochs=1,
            batch_size=4,
            random_state=0,
        ).fit(X, X.reshape(len(X), -1).mean(axis=1))
        path = tmp_path / f"{name}.pt"
        estimator.save(str(path))
        np.testing.assert_allclose(
            PSANNRegressor.load(str(path)).predict(X[:2]), estimator.predict(X[:2]), rtol=1e-6
        )


def test_enriched_legacy_wave_migrates_and_survives_v1_resave(tmp_path):
    from psann._sklearn.wave import WaveResNetRegressor as LegacyWave

    X = np.ones((8, 2, 2), dtype=np.float32)
    y = X.mean(axis=(1, 2))
    legacy = LegacyWave(
        hidden_layers=2,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=0,
        first_layer_w0=31.0,
        use_spectral_gate=True,
        k_fft=2,
        progressive_depth_initial=1,
        w0_warmup_epochs=1,
    ).fit(X, y)
    old_path = tmp_path / "legacy-wave.pt"
    legacy.save(str(old_path))
    migrated = PSANNRegressor.load(str(old_path))
    assert migrated.architecture.spectral and migrated.architecture.spectral.k_fft == 2
    assert migrated.architecture.wave and migrated.architecture.wave.progressive_depth
    v1_path = tmp_path / "resaved.pt"
    migrated.save(str(v1_path))
    np.testing.assert_allclose(
        PSANNRegressor.load(str(v1_path)).predict(X[:2]), migrated.predict(X[:2]), rtol=1e-6
    )


def test_schema_v1_load_works_without_sklearn(tmp_path):
    X = np.ones((8, 2), dtype=np.float32)
    model = PSANNRegressor(hidden_layers=1, hidden_units=4, epochs=1, batch_size=4).fit(
        X, X.mean(axis=1)
    )
    path = tmp_path / "schema.pt"
    model.save(str(path))
    code = """import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'sklearn' or name.startswith('sklearn.'):
        raise ImportError('blocked')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from psann.estimators import PSANNRegressor
assert PSANNRegressor.load(sys.argv[1]).predict(__import__('numpy').ones((2, 2), dtype='float32')).shape == (2,)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, os.fspath(path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
