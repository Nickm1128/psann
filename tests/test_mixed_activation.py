from __future__ import annotations

import torch

from psann.activations import MixedActivation, SineParam


def test_mixed_activation_defaults_to_equal_ratios() -> None:
    act = MixedActivation(10, activation_types=["relu", "tanh"])
    x = torch.randn(2, 10)
    y = act(x)
    assert y.shape == x.shape


def test_mixed_activation_ratio_sum_tolerance_and_renormalize() -> None:
    # Sum is 0.9999; within tol=1e-3 => accepted + renormalized internally.
    act = MixedActivation(
        12,
        activation_types=["relu", "tanh", "gelu"],
        activation_ratios=[0.3333, 0.3333, 0.3333],
    )
    x = torch.randn(1, 12)
    y = act(x)
    assert y.shape == x.shape


def test_mixed_activation_raises_when_ratios_sum_far_from_one() -> None:
    try:
        _ = MixedActivation(
            8,
            activation_types=["relu", "tanh"],
            activation_ratios=[0.2, 0.2],
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError for badly-normalized activation_ratios")


def test_mixed_activation_trains_sine_params_when_present() -> None:
    act = MixedActivation(
        10,
        activation_types=["psann", "relu"],
        activation_ratios=[0.5, 0.5],
    )
    assert "psann" in act.acts
    assert isinstance(act.acts["psann"], SineParam)

    x = torch.randn(4, 10, requires_grad=True)
    y = act(x).sum()
    y.backward()
    # Ensure sine parameters received gradients.
    sine = act.acts["psann"]
    assert sine._A.grad is not None
    assert sine._f.grad is not None
    assert sine._d.grad is not None

