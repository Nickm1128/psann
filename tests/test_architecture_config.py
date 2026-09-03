from __future__ import annotations

import copy
import pickle
import warnings

import pytest
import numpy as np
import torch

from psann.architectures import (
    AttentionConfig,
    ArchitectureConfig,
    ConvolutionConfig,
    ResidualConfig,
    architecture_to_mapping,
    normalize_architecture,
)
from psann import PSANNRegressor
from psann.conv import ResidualPSANNConv2dNet


@pytest.mark.parametrize(
    ("value", "kind", "residual", "convolution"),
    [
        (" dense ", "dense", False, False),
        ("residual", "dense", True, False),
        ("convolutional", "convolutional", False, True),
        ("res_conv_psann", "convolutional", True, True),
        ("wave", "wave", True, False),
        ("sequence", "sequence", False, False),
        ("geo_sparse", "geometric-sparse", True, False),
    ],
)
def test_documented_preset_normalization(value, kind, residual, convolution):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        config = normalize_architecture(value)
    assert config.kind == kind
    assert (config.residual is not None) is residual
    assert (config.convolution is not None) is convolution
    if value.strip() in {"dense", "residual", "convolutional", "wave", "sequence"}:
        assert not record


def test_tagged_mapping_is_lossless_and_caller_mutation_isolated():
    raw = {"kind": "convolutional", "convolution": {"channels": 7, "kernel_size": 3}}
    config = normalize_architecture(raw)
    raw["convolution"]["channels"] = 99
    assert config.convolution == ConvolutionConfig(channels=7, kernel_size=3)
    assert normalize_architecture(architecture_to_mapping(config)) == config
    assert pickle.loads(pickle.dumps(config)) == config
    assert copy.deepcopy(config) == config


def test_invalid_policy_combinations_are_rejected_before_build():
    with pytest.raises(ValueError, match="residual and state"):
        ArchitectureConfig.dense(residual=ResidualConfig(), state={"rho": 0.8})
    with pytest.raises(ValueError, match="requires residual and wave"):
        ArchitectureConfig(kind="wave")
    with pytest.raises(ValueError, match="unsupported policy"):
        ArchitectureConfig.convolutional(state={"rho": 0.8})


def test_nested_policy_updates_are_transactional_and_validate_only_final_state():
    estimator = PSANNRegressor(architecture=ArchitectureConfig.for_wave())
    estimator.set_params(
        architecture__spectral=None,
        architecture__attention=AttentionConfig(num_heads=2),
    )
    assert estimator.architecture.spectral is None
    assert estimator.architecture.attention == AttentionConfig(num_heads=2)


def test_flat_architecture_routes_preserve_every_supported_policy():
    conv = PSANNRegressor(
        preserve_shape=True,
        norm="layer",
        attention={"kind": "mha", "num_heads": 1},
    ).architecture
    assert conv.kind == "convolutional"
    assert conv.residual and conv.residual.norm == "layer"
    assert conv.attention == AttentionConfig(num_heads=1)

    wave = PSANNRegressor(first_layer_w0=31.0, use_spectral_gate=True, k_fft=8).architecture
    assert wave.kind == "wave" and wave.spectral and wave.spectral.k_fft == 8

    sequence = PSANNRegressor(phase_init=0.25, pool="mean").architecture
    assert sequence.kind == "sequence"
    assert sequence.sequence and sequence.sequence.pool == "mean"


def test_flat_shaped_residual_attention_uses_registry_at_fit_boundary():
    X = np.ones((8, 1, 2, 2), dtype=np.float32)
    estimator = PSANNRegressor(
        preserve_shape=True,
        norm="layer",
        attention={"kind": "mha", "num_heads": 1},
        conv_channels=4,
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=0,
    ).fit(X, X.mean(axis=(1, 2, 3)))
    core = estimator.model_.core
    assert isinstance(core.conv_core, ResidualPSANNConv2dNet)
    assert core.attention is not None
    assert estimator._architecture_capabilities_.supports_attention
    assert estimator.predict(X[:2]).shape == (2,)


def test_flat_registry_checkpoint_reloads_the_registry_topology(tmp_path):
    X = np.ones((8, 1, 2, 2), dtype=np.float32)
    estimator = PSANNRegressor(
        preserve_shape=True,
        norm="layer",
        attention={"kind": "mha", "num_heads": 1},
        conv_channels=4,
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=0,
    ).fit(X, X.mean(axis=(1, 2, 3)))
    path = tmp_path / "flat-registry.pt"
    estimator.save(str(path))
    loaded = PSANNRegressor.load(str(path))
    assert isinstance(loaded.model_.core.conv_core, ResidualPSANNConv2dNet)
    assert loaded.model_.core.attention is not None
    assert loaded._architecture_capabilities_.supports_attention
    np.testing.assert_allclose(loaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-5)
    assert torch.load(path, weights_only=False)["fitted"]["legacy_flattened_preserve_shape"] is True
