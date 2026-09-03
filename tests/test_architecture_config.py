from __future__ import annotations

import copy
import pickle
import warnings

import pytest

from psann.architectures import (
    AttentionConfig,
    ArchitectureConfig,
    ConvolutionConfig,
    ResidualConfig,
    architecture_to_mapping,
    normalize_architecture,
)
from psann import PSANNRegressor


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
