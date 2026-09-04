"""Resolved shared initializers preserve numerical behavior and validate inputs."""

from dataclasses import replace

import pytest
import torch

from psann.activations import SineParam
from psann.architectures import ActivationConfig, ArchitectureConfig, normalize_architecture
from psann.architectures.components import build_activation


@pytest.mark.parametrize("field", ["amplitude", "frequency", "decay"])
@pytest.mark.parametrize("vector", [False, True])
def test_resolved_initializer_exact_state_logits_and_parameter_gradients(field, vector):
    initial = torch.linspace(0.13, 1.7, 11) if vector else 0.37
    values = {field: initial}
    expected_values = dict(amplitude_init=1.0, frequency_init=1.0, decay_init=0.1)
    expected_values[field + "_init"] = initial
    expected = SineParam(11, **expected_values)
    actual = build_activation(ActivationConfig(), features=11, initial_values=values)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)
    x = torch.linspace(-1.73, 2.31, 154).reshape(2, 7, 11).requires_grad_()
    y = x.detach().clone().requires_grad_()
    a, b = actual(x), expected(y)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    for p, q in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    if vector:
        before = actual(x.detach()).detach().clone()
        initial.fill_(9)
        torch.testing.assert_close(actual(x.detach()), before, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["amplitude", "frequency", "decay"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        "0.3",
        float("inf"),
        float("nan"),
        torch.ones(1, 11),
        torch.ones(10),
        torch.tensor(0.3),
        torch.ones(11, dtype=torch.bool),
        torch.ones(11, dtype=torch.complex64),
        torch.full((11,), float("inf")),
    ],
)
def test_initial_values_rejected_neighbors(field, value):
    with pytest.raises((TypeError, ValueError), match="initial_values." + field):
        build_activation(ActivationConfig(), features=11, initial_values={field: value})


def test_initial_values_mapping_and_unknown_key_rejections():
    with pytest.raises(TypeError, match="initial_values"):
        build_activation(ActivationConfig(), features=11, initial_values=[0.3])
    with pytest.raises(ValueError, match="initial_values.extra"):
        build_activation(ActivationConfig(), features=11, initial_values={"extra": 0.3})
    with pytest.raises(ValueError, match="initial_values"):
        build_activation(
            ActivationConfig(kind="gelu"), features=11, initial_values={"amplitude": 0.3}
        )


@pytest.mark.parametrize("kind", ["dense", "convolutional", "wave", "sequence"])
def test_typed_gelu_does_not_broaden_unsupported_core_capabilities(kind):
    defaults = normalize_architecture(kind)
    with pytest.raises(ValueError, match="activation.kind='gelu'"):
        normalize_architecture(replace(defaults, activation=ActivationConfig(kind="gelu")))


def test_core_geometric_gelu_and_mixed_gelu_forward_backward():
    from psann.architectures import GeometryConfig, ResidualConfig

    # This component test also executes the accepted core GeoSparse numerical builder.
    from psann.nn_geo_sparse import GeoSparseNet

    for kind, extra in (
        ("gelu", {}),
        ("mixed", {"activation_types": ("psann", "gelu"), "activation_ratios": (0.4, 0.6)}),
    ):
        activation = ActivationConfig(kind=kind, **extra)
        policy = ArchitectureConfig(
            kind="geometric-sparse",
            activation=activation,
            residual=ResidualConfig(alpha_init=0.7),
            geometry=GeometryConfig(shape=(3, 4)),
        )
        assert normalize_architecture(policy) == policy
        model = GeoSparseNet(
            12,
            3,
            shape=(3, 4),
            activation_type=kind,
            activation_config=vars(activation),
            residual_alpha_init=0.7,
        )
        x = torch.linspace(-1.7, 2.3, 60).reshape(5, 12).requires_grad_()
        y = model(x)
        assert y.shape == (5, 3)
        y.square().sum().backward()
        assert torch.count_nonzero(x.grad) > 10
