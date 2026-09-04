"""Value-sensitive parity at the shared core component boundary."""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
import ast

import pytest
import torch
from torch import nn

from psann.architectures import ActivationConfig, GeometryConfig
from psann.architectures.components import (
    DropPath,
    GeometryConnectivity,
    RMSNorm,
    SpectralGate1D,
    build_activation,
    build_geometry_connectivity,
)
from psann.layers.geo_sparse import build_geo_connectivity, expand_in_indices_to_edges
from psann.layers.sine_residual import RMSNorm as ExistingRMSNorm
from psann.layers.spectral import SpectralGate1D as ExistingSpectralGate
from psann.nn import DropPath as ExistingDropPath
from psann.nn_geo_sparse import _build_activation


def assert_forward_and_backward(actual, expected, shape, *, seed=109):
    expected.load_state_dict(actual.state_dict())
    x = torch.linspace(-1.71, 2.39, torch.tensor(shape).prod().item()).reshape(shape)
    x.requires_grad_()
    y = x.detach().clone().requires_grad_()
    torch.manual_seed(seed)
    a = actual(x)
    torch.manual_seed(seed)
    b = expected(y)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    assert torch.count_nonzero(x.grad) > 1
    assert dict(actual.named_parameters()).keys() == dict(expected.named_parameters()).keys()
    for (_, p), (_, q) in zip(actual.named_parameters(), expected.named_parameters()):
        assert p.requires_grad == q.requires_grad
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kind", ["psann", "phase-psann", "mixed", "relu", "tanh", "relu-sigmoid-psann"]
)
@pytest.mark.parametrize("typed", [True, False])
def test_activation_policy_core_forward_and_parameter_gradient_parity(kind, typed):
    values = dict(
        kind=kind,
        amplitude_init=1.3,
        frequency_init=0.7,
        decay_init=0.04,
        learnable=("frequency",),
        bounds={"amplitude": (0.2, 1.1)},
    )
    if kind in {"phase-psann", "mixed"}:
        values.update(phase_init=0.37, phase_trainable=False)
    if kind == "mixed":
        values.update(
            activation_types=("psann", "phase-psann", "relu"),
            activation_ratios=(0.3, 0.3, 0.4),
            mix_seed=37,
            mix_layout="contiguous",
        )
    policy = ActivationConfig(**values)
    raw = {field.name: getattr(policy, field.name) for field in fields(policy)}
    if policy.activation_types:
        raw["activation_types"] = [key.replace("-", "_") for key in policy.activation_types]
    actual = build_activation(policy if typed else values, features=11)
    expected = _build_activation(kind.replace("-", "_"), 11, raw)
    assert_forward_and_backward(actual, expected, (2, 5, 11))
    assert policy == ActivationConfig(**values)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("layout", ["random", "contiguous"])
def test_mixed_axis_forward_and_gradient_matches_last_axis_reference(axis, layout):
    values = dict(
        kind="mixed",
        activation_types=("psann", "phase-psann", "relu"),
        activation_ratios=(0.3, 0.3, 0.4),
        mix_seed=37,
        mix_layout=layout,
        frequency_init=0.7,
        phase_init=0.37,
    )
    actual = build_activation(ActivationConfig(**values, feature_dim=axis), features=11)
    expected = build_activation(ActivationConfig(**values), features=11)
    last = torch.linspace(-1.7, 2.3, 110).reshape(2, 5, 11).requires_grad_()
    moved = last.detach().movedim(-1, axis).contiguous().requires_grad_()
    a = actual(moved)
    b = expected(last)
    torch.testing.assert_close(a.movedim(axis, -1), b, rtol=0, atol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(moved.grad.movedim(axis, -1), last.grad, rtol=0, atol=0)
    for p, q in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize("axis", [3, -4])
def test_mixed_axis_rejects_out_of_range_at_forward(axis):
    model = build_activation(
        ActivationConfig(kind="mixed", activation_types=("psann", "relu"), feature_dim=axis),
        features=11,
    )
    with pytest.raises(ValueError, match="feature_dim is out of range"):
        model(torch.zeros(2, 5, 11))


@pytest.mark.parametrize("config", ["gelu", {"kind": "gelu"}])
def test_fixed_gelu_exact_value_and_gradient_parity(config):
    assert_forward_and_backward(build_activation(config, features=11), nn.GELU(), (2, 5, 11))


@pytest.mark.parametrize(
    "config,features,match",
    [
        ("GELU", 11, "activation literal"),
        ({"kind": "gelu", "amplitude_init": 2}, 11, "activation.amplitude_init"),
        ({"kind": "psann", "unknown": 2}, 11, "activation.unknown"),
        (ActivationConfig(), True, "features"),
        (ActivationConfig(), "11", "features"),
        (ActivationConfig(), 0, "features"),
    ],
)
def test_activation_component_rejects_invalid_inputs(config, features, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build_activation(config, features=features)


def test_rmsnorm_last_dimension_forward_and_backward_parity():
    actual = RMSNorm(11, eps=0.003)
    with torch.no_grad():
        actual.scale.copy_(torch.linspace(0.4, 1.6, 11))
    assert_forward_and_backward(actual, ExistingRMSNorm(11, eps=0.003), (3, 5, 11))


@pytest.mark.parametrize("training", [True, False])
def test_drop_path_per_sample_mask_forward_and_backward_parity(training):
    actual = DropPath(0.35).train(training)
    expected = ExistingDropPath(0.35).train(training)
    assert_forward_and_backward(actual, expected, (13, 5, 11))
    torch.manual_seed(109)
    out = actual(torch.ones(13, 5, 11))
    if training:
        assert (out == 0).any() and (out > 1).any()
        assert torch.equal(out, out[:, :1, :1].expand_as(out))
    else:
        assert torch.equal(out, torch.ones_like(out))


@pytest.mark.parametrize("gate", ["rfft", "fourier_features"])
@pytest.mark.parametrize("groups", ["depthwise", "full"])
def test_spectral_gate_forward_and_parameter_gradient_parity(gate, groups):
    kwargs = dict(k_fft=5, gate_type=gate, gate_groups=groups, gate_init=0.3, gate_strength=0.7)
    actual = SpectralGate1D(12, **kwargs)
    expected = ExistingSpectralGate(12, **kwargs)
    assert_forward_and_backward(actual, expected, (2, 9, 12))


@pytest.mark.parametrize("pattern", ["local", "random", "hash"])
@pytest.mark.parametrize("wrap", ["clamp", "wrap"])
@pytest.mark.parametrize("offsets", [None, ((0, 0), (-1, 2), (1, -2))])
def test_geometry_exact_indices_edges_and_gather_scatter_backward(pattern, wrap, offsets):
    config = GeometryConfig(
        shape=(3, 5), k=7, pattern=pattern, radius=2, offsets=offsets, wrap_mode=wrap, seed=41
    )
    connectivity = build_geometry_connectivity(config, features=15)
    indices, src, dst = connectivity.as_tensors()
    legacy = build_geo_connectivity(
        (3, 5), k=7, pattern=pattern, radius=2, offsets=offsets, wrap_mode=wrap, seed=41
    )
    expected_src, expected_dst = expand_in_indices_to_edges(legacy)
    assert torch.equal(indices, legacy)
    assert torch.equal(src, expected_src) and torch.equal(dst, expected_dst)
    x = torch.linspace(-1.3, 2.1, 2 * 4 * 15).reshape(8, 15).requires_grad_()
    other = x.detach().clone().requires_grad_()
    weight = torch.linspace(0.1, 0.9, 15 * 7).reshape(15, 7)
    gathered = (x[:, indices] * weight).sum(-1)
    scattered = torch.zeros(8, 15).index_add(1, dst, other[:, src] * weight.flatten())
    torch.testing.assert_close(gathered, scattered, rtol=1e-6, atol=1e-6)
    gathered.square().sum().backward()
    scattered.square().sum().backward()
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(x.grad) > 15
    indices.fill_(0)
    src.fill_(0)
    dst.fill_(0)
    assert torch.equal(connectivity.as_tensors()[0], legacy)
    with pytest.raises(FrozenInstanceError):
        connectivity.shape = (5, 3)


@pytest.mark.parametrize("features,shape", [(11, (1, 11)), (15, (3, 5)), (36, (6, 6))])
def test_geometry_inferred_shape_and_explicit_shape_are_equal(features, shape):
    inferred = build_geometry_connectivity(GeometryConfig(k=3, seed=29), features=features)
    explicit = build_geometry_connectivity(GeometryConfig(shape=shape, k=3, seed=29))
    assert inferred == explicit
    assert torch.equal(inferred.as_tensors()[0], explicit.as_tensors()[0])


@pytest.mark.parametrize(
    "config,features,match",
    [
        (GeometryConfig(), None, "geometry.shape or features"),
        (GeometryConfig(), True, "features"),
        (GeometryConfig(), "15", "features"),
        (GeometryConfig(), 0, "features"),
        (GeometryConfig(shape=(3, 5)), 16, "geometry.shape"),
        ({"shape": (3, 5)}, 15, "geometry"),
    ],
)
def test_geometry_component_rejects_invalid_requests(config, features, match):
    with pytest.raises((TypeError, ValueError), match=match):
        build_geometry_connectivity(config, features=features)


@pytest.mark.parametrize(
    "rows,match",
    [
        (((0,),), "shape"),
        (((0,), (1, 2)), "equal lengths"),
        (((True,), (1,)), "integers"),
        (((-1,), (1,)), "out-of-range"),
        (((0,), (2,)), "out-of-range"),
    ],
)
def test_immutable_connectivity_rejects_malformed_indices(rows, match):
    with pytest.raises((TypeError, ValueError), match=match):
        GeometryConnectivity((1, 2), rows)


def test_core_source_contains_no_lm_import_or_optional_probe():
    core = Path(__file__).resolve().parents[1] / "src/psann"
    for file in core.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("psannlm"), str(file)
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("psannlm") for alias in node.names), str(file)
            elif isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name in {"find_spec", "import_module", "__import__"}:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            assert not arg.value.startswith("psannlm"), str(file)
