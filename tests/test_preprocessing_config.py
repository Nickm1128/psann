"""Strict typed configuration coverage for the Phase 4 preprocessing API."""

from __future__ import annotations

import pytest

from psann.preprocessing import (
    LSMConfig,
    LSMPretrainingConfig,
    PreprocessorConfig,
    PreprocessorTrainingConfig,
    normalize_preprocessor,
    preprocessor_to_mapping,
)


def test_typed_mapping_round_trip_and_identity() -> None:
    typed = PreprocessorConfig(
        component=LSMConfig.dense(
            output_dim=8,
            hidden_layers=1,
            hidden_units=6,
            pretraining=LSMPretrainingConfig(epochs=1, lr=0.01, noisy=[0.1, 0.2]),
        ),
        training=PreprocessorTrainingConfig(trainable=True, lr=0.005),
    )
    mapping = preprocessor_to_mapping(typed)
    assert normalize_preprocessor(typed) is typed
    assert normalize_preprocessor(mapping) == typed
    mapping["lsm"]["output_dim"] = 99  # type: ignore[index]
    assert typed.component.output_dim == 8


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ({"kind": "lsm"}, "preprocessor.lsm"),
        ({"kind": "other", "lsm": {}}, "preprocessor.kind"),
        ({"kind": "lsm", "lsm": {"output_dim": True}}, "preprocessor.component.output_dim"),
        ({"kind": "lsm", "lsm": {"output_dim": "4"}}, "preprocessor.component.output_dim"),
        ({"kind": "lsm", "lsm": {"sparsity": float("nan")}}, "preprocessor.component.sparsity"),
        (
            {"kind": "lsm", "lsm": {"pretraining": {"epochs": True}}},
            "preprocessor.component.pretraining.epochs",
        ),
        ({"kind": "lsm", "lsm": {"unknown": 1}}, "preprocessor.lsm.unknown"),
        ({"kind": "lsm", "lsm": {"hidden_width": 3, "hidden_units": 4}}, "conflicting"),
    ],
)
def test_mapping_errors_are_path_specific(value: dict[str, object], path: str) -> None:
    with pytest.raises((TypeError, ValueError), match=path):
        normalize_preprocessor(value)


def test_conv_lsm_rejects_dense_only_pretraining_options() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        LSMConfig.convolutional(
            output_dim=4,
            pretraining=LSMPretrainingConfig(batch_size=2),
        )


@pytest.mark.parametrize(
    ("topology", "alias", "expected_field"),
    [
        ("dense", "hidden_width", "hidden_units"),
        ("conv2d", "out_channels", "output_dim"),
        ("conv2d", "hidden_channels", "hidden_units"),
    ],
    ids=["dense-hidden-width", "conv2d-out-channels", "conv2d-hidden-channels"],
)
def test_canonical_mapping_alias_matrix_normalizes_without_mutation(
    topology: str, alias: str, expected_field: str
) -> None:
    lsm: dict[str, object] = {
        "topology": topology,
        "output_dim": 4,
        "hidden_units": 5,
    }
    if topology == "conv2d":
        lsm["kernel_size"] = 1
    lsm.pop(expected_field)
    lsm[alias] = 6
    source = {"kind": "lsm", "lsm": lsm, "training": {"trainable": False, "lr": None}}
    normalized = normalize_preprocessor(source)
    assert normalized is not None
    assert getattr(normalized.component, expected_field) == 6
    assert alias in source["lsm"]  # type: ignore[operator]


@pytest.mark.parametrize(
    ("value", "path"),
    [
        ({"kind": "lsm", "lsm": [], "training": {}}, "preprocessor.lsm"),
        ({"kind": "lsm", "lsm": {}, "training": []}, "preprocessor.training"),
        ({"kind": "lsm", "lsm": {"output_dim": [4]}}, "output_dim"),
        ({"kind": "lsm", "lsm": {"output_dim": 4}, "training": {"trainable": 1}}, "trainable"),
        ({"kind": "lsm", "lsm": {"output_dim": 0}}, "output_dim"),
        ({"kind": "lsm", "lsm": {"sparsity": -0.1}}, "sparsity"),
        ({"kind": "lsm", "lsm": {"sparsity": 1.1}}, "sparsity"),
        ({"kind": "lsm", "lsm": {"random_state": True}}, "random_state"),
        ({"kind": "lsm", "lsm": {"pretraining": {"epochs": "1"}}}, "epochs"),
        ({"kind": "lsm", "lsm": {"pretraining": {"noisy": [0.1, "bad"]}}}, "noisy"),
        ({"kind": "lsm", "lsm": {"output_dim": 4}, "training": {"lr": "0.1"}}, "lr"),
        ({"kind": "lsm", "lsm": {"output_dim": 4, "out_channels": 5}}, "conflicting"),
    ],
)
def test_canonical_mapping_adversarial_type_and_domain_matrix(
    value: dict[str, object], path: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=path):
        normalize_preprocessor(value)
