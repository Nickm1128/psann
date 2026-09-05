"""Canonical architecture construction for the advanced PyTorch experiments.

Use PSANNRegressor for regression workflows. These examples own their optimizer
because they compose temporal networks or train a classification objective.
"""

import math

import torch

from psann.architectures import ArchitectureBuildRequest, build_architecture


def build_backbone(architecture, input_shape, output_dim, *, depth=2, width=32):
    spatial = architecture.kind == "convolutional"
    input_shape = tuple(input_shape)
    request = ArchitectureBuildRequest(
        architecture=architecture,
        hidden_layers=depth,
        hidden_units=width,
        input_shape=input_shape,
        input_dim=math.prod(input_shape),
        output_dim=output_dim,
        spatial_shape=input_shape[1:] if spatial else None,
        spatial_ndim=len(input_shape) - 1 if spatial else None,
        in_channels=input_shape[0] if spatial else None,
        sequence_length=None,
        token_dim=None,
        per_element=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        preprocessor=None,
        preprocessor_output_dim=None,
        structure_metadata=None,
    )
    return build_architecture(request).model
