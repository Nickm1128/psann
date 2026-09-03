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
