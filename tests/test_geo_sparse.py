from __future__ import annotations

import torch

from psann.activations import PhaseSineParam, SineParam
from psann.layers.geo_sparse import (
    GeoSparseLinear,
    build_geo_connectivity,
    expand_in_indices_to_edges,
)
from psann.nn_geo_sparse import GeoSparseNet, GeoSparseResidualBlock
from psann.params import (
    count_params,
    dense_mlp_params,
    geo_sparse_net_params,
    match_dense_width,
)


def test_geo_connectivity_deterministic_with_seed() -> None:
    shape = (4, 4)
    k = 5
    idx1 = build_geo_connectivity(shape, k=k, pattern="random", radius=1, seed=123)
    idx2 = build_geo_connectivity(shape, k=k, pattern="random", radius=1, seed=123)
    assert torch.equal(idx1, idx2)


def test_geo_connectivity_indices_in_range() -> None:
    shape = (3, 5)
    k = 4
    indices = build_geo_connectivity(shape, k=k, pattern="local", radius=1, seed=0)
    assert indices.shape == (shape[0] * shape[1], k)
    assert int(indices.min().item()) >= 0
    assert int(indices.max().item()) < shape[0] * shape[1]


def test_geo_connectivity_edge_list_size() -> None:
    shape = (3, 3)
    k = 3
    indices = build_geo_connectivity(shape, k=k, pattern="hash", radius=1, seed=11)
    src, dst = expand_in_indices_to_edges(indices)
    n_out = shape[0] * shape[1]
    assert src.shape == (n_out * k,)
    assert dst.shape == (n_out * k,)
    dst_view = dst.view(n_out, k)
    expected = torch.arange(n_out).view(n_out, 1).expand(n_out, k)
    assert torch.equal(dst_view, expected)


def test_geo_sparse_linear_matches_dense_and_grads() -> None:
    torch.manual_seed(0)
    in_features = 5
    out_features = 3
    indices = torch.arange(in_features).repeat(out_features, 1)
    layer = GeoSparseLinear(in_features, out_features, indices, bias=True, compute_mode="gather")
    dense = torch.nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        dense.weight.copy_(layer.weight)
        dense.bias.copy_(layer.bias)

    x = torch.randn(4, in_features, requires_grad=True)
    x_dense = x.clone().detach().requires_grad_(True)
    out_sparse = layer(x)
    out_dense = dense(x_dense)
    assert torch.allclose(out_sparse, out_dense, atol=1e-6)

    out_sparse.sum().backward()
    out_dense.sum().backward()
    assert torch.allclose(x.grad, x_dense.grad, atol=1e-6)
    assert torch.allclose(layer.weight.grad, dense.weight.grad, atol=1e-6)
    assert torch.allclose(layer.bias.grad, dense.bias.grad, atol=1e-6)


def test_geo_sparse_linear_scatter_matches_gather() -> None:
    torch.manual_seed(1)
    in_features = 4
    out_features = 4
    indices = torch.arange(in_features).repeat(out_features, 1)
    gather = GeoSparseLinear(
        in_features, out_features, indices, bias=True, compute_mode="gather"
    )
    scatter = GeoSparseLinear(
        in_features, out_features, indices, bias=True, compute_mode="scatter"
    )
    with torch.no_grad():
        scatter.weight.copy_(gather.weight)
        scatter.bias.copy_(gather.bias)
    x = torch.randn(3, 2, in_features)
    out_gather = gather(x)
    out_scatter = scatter(x)
    assert out_gather.shape == (3, 2, out_features)
    assert torch.allclose(out_gather, out_scatter, atol=1e-6)


def test_geo_sparse_net_forward_shapes() -> None:
    shape = (4, 4)
    net = GeoSparseNet(
        input_dim=16,
        output_dim=3,
        shape=shape,
        depth=2,
        k=4,
        activation_type="relu",
        norm="none",
        seed=123,
    )
    x_flat = torch.randn(5, 16)
    out_flat = net(x_flat)
    assert out_flat.shape == (5, 3)

    x_grid = torch.randn(2, *shape)
    out_grid = net(x_grid)
    assert out_grid.shape == (2, 3)


def test_geo_sparse_activation_config_aliases() -> None:
    features = 3
    indices = torch.arange(features).repeat(features, 1)
    block = GeoSparseResidualBlock(
        features,
        indices,
        activation_type="psann",
        activation_config={"amp_init": 0.7, "freq_init": 1.1, "damp_init": 0.2, "trainable": False},
        norm="none",
    )
    act = block.act
    assert isinstance(act, SineParam)
    assert not act._A.requires_grad
    assert not act._f.requires_grad
    assert not act._d.requires_grad


def test_geo_sparse_phase_psann_config() -> None:
    features = 4
    indices = torch.arange(features).repeat(features, 1)
    block = GeoSparseResidualBlock(
        features,
        indices,
        activation_type="phase_psann",
        activation_config={"phase_init": 1.25, "phase_trainable": False},
        norm="none",
    )
    act = block.act
    assert isinstance(act, PhaseSineParam)
    assert torch.allclose(act._phi.detach(), torch.full((features,), 1.25))
    assert not act._phi.requires_grad


def test_geo_sparse_mixed_activation_builds_and_trains_sine_subset() -> None:
    from psann.activations import MixedActivation

    features = 8
    indices = torch.arange(features).repeat(features, 1)
    block = GeoSparseResidualBlock(
        features,
        indices,
        activation_type="mixed",
        activation_config={
            "activation_types": ["psann", "relu"],
            "activation_ratios": [0.5, 0.5],
            "trainable": True,
        },
        norm="none",
    )
    assert isinstance(block.act, MixedActivation)
    assert "psann" in block.act.acts
    x = torch.randn(2, features, requires_grad=True)
    y = block(x).sum()
    y.backward()
    sine = block.act.acts["psann"]
    assert sine._A.grad is not None


def test_param_helpers_and_matcher() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    assert count_params(model) == sum(p.numel() for p in model.parameters())

    dense = dense_mlp_params(input_dim=3, output_dim=2, hidden_dim=4, depth=2, bias=True)
    assert dense == (3 * 4 + 4) + (4 * 4 + 4) + (4 * 2 + 2)

    sparse = geo_sparse_net_params(shape=(2, 2), depth=2, k=3, output_dim=1, bias=True)
    assert sparse > 0

    target = dense_mlp_params(input_dim=3, output_dim=2, hidden_dim=6, depth=2, bias=True)
    width, mismatch = match_dense_width(
        target_params=target,
        input_dim=3,
        output_dim=2,
        depth=2,
        width_candidates=range(2, 9),
    )
    assert width == 6
    assert mismatch == 0
