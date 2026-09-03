from __future__ import annotations

import torch

from psann.architectures import (
    ArchitectureBuildRequest,
    ArchitectureConfig,
    available_architectures,
    build_architecture,
)


def _request(config: ArchitectureConfig) -> ArchitectureBuildRequest:
    return ArchitectureBuildRequest(
        config,
        2,
        8,
        (2,),
        2,
        1,
        None,
        None,
        None,
        None,
        None,
        False,
        torch.device("cpu"),
        torch.float32,
        None,
        None,
        None,
    )


def test_builtins_are_registered_and_dense_builds_model_and_lifecycle():
    assert available_architectures() == (
        "convolutional",
        "dense",
        "geometric-sparse",
        "sequence",
        "wave",
    )
    result = build_architecture(_request(ArchitectureConfig.dense()))
    assert result.capabilities.kind == "dense"
    assert result.model(torch.zeros(2, 2)).shape == (2, 1)
    assert result.lifecycle.structure_metadata() == {}
