from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import nn


def count_params(model: nn.Module, *, trainable_only: bool = False) -> int:
    """Return total parameter count for a PyTorch module."""
    if trainable_only:
        return int(sum(int(p.numel()) for p in model.parameters() if p.requires_grad))
    return int(sum(int(p.numel()) for p in model.parameters()))


def sparse_linear_params(out_features: int, k: int, *, bias: bool = True) -> int:
    if out_features <= 0 or k <= 0:
        raise ValueError("out_features and k must be positive.")
    return int(out_features * k + (out_features if bias else 0))


def dense_linear_params(in_features: int, out_features: int, *, bias: bool = True) -> int:
    if in_features <= 0 or out_features <= 0:
        raise ValueError("in_features and out_features must be positive.")
    return int(in_features * out_features + (out_features if bias else 0))


def geo_sparse_block_params(features: int, k: int, *, bias: bool = True) -> int:
    # Two sparse linear layers in the residual branch, plus learnable alpha.
    return int(2 * sparse_linear_params(features, k, bias=bias) + 1)


def geo_sparse_net_params(
    *,
    shape: Tuple[int, int],
    depth: int,
    k: int,
    output_dim: int,
    bias: bool = True,
    include_head_bias: bool = True,
) -> int:
    if depth <= 0:
        raise ValueError("depth must be positive.")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive.")
    features = height * width
    total = depth * geo_sparse_block_params(features, k, bias=bias)
    total += dense_linear_params(features, output_dim, bias=include_head_bias)
    return int(total)


def dense_mlp_params(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    bias: bool = True,
) -> int:
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    total = dense_linear_params(input_dim, hidden_dim, bias=bias)
    if depth > 1:
        total += (depth - 1) * dense_linear_params(hidden_dim, hidden_dim, bias=bias)
    total += dense_linear_params(hidden_dim, output_dim, bias=bias)
    return int(total)


def match_dense_width(
    *,
    target_params: int,
    input_dim: int,
    output_dim: int,
    depth: int,
    bias: bool = True,
    width_candidates: Optional[Sequence[int]] = None,
    max_width: int = 8192,
) -> Tuple[int, int]:
    """Find a dense MLP width that best matches the target parameter count."""
    if target_params <= 0:
        raise ValueError("target_params must be positive.")
    if width_candidates is None:
        width_candidates = range(1, max_width + 1)
    best_width = None
    best_mismatch = None
    for width in width_candidates:
        params = dense_mlp_params(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=int(width),
            depth=int(depth),
            bias=bias,
        )
        mismatch = abs(params - target_params)
        if best_mismatch is None or mismatch < best_mismatch:
            best_mismatch = mismatch
            best_width = int(width)
    if best_width is None or best_mismatch is None:
        raise ValueError("No candidate widths to search.")
    return best_width, int(best_mismatch)
