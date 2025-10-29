import warnings

import pytest
import torch

from psann.state import StateController


def test_state_controller_warns_on_saturation():
    ctrl = StateController(size=1, rho=0.0, beta=10.0, max_abs=1.0, init=0.1, detach=True)
    activations = torch.ones(4, 1) * 5.0
    with pytest.warns(RuntimeWarning, match="98% of max_abs"):
        ctrl.apply(activations, feature_dim=1, update=True)
    ctrl.commit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctrl.apply(activations, feature_dim=1, update=True)
    assert not any("98% of max_abs" in str(item.message) for item in caught)


def test_state_controller_warns_on_collapse():
    ctrl = StateController(size=1, rho=0.0, beta=0.0, max_abs=2.0, init=1.0, detach=True)
    activations = torch.zeros(4, 1)
    with pytest.warns(RuntimeWarning, match="collapsed below 1e-3"):
        ctrl.apply(activations, feature_dim=1, update=True)


def test_state_controller_commit_and_reset_roundtrip():
    ctrl = StateController(size=2, rho=0.5, beta=1.0, max_abs=10.0, init=1.0, detach=True)
    activations = torch.tensor([[2.0, 0.5], [2.0, 1.5], [2.0, 1.0]], dtype=torch.float32)

    state_before = ctrl.state.clone()
    scaled = ctrl.apply(activations, feature_dim=1, update=True)
    pending = ctrl._pending_state  # type: ignore[attr-defined]
    raw_update = ctrl.rho * state_before + (1.0 - ctrl.rho) * torch.tensor([2.0, 1.0])
    expected = ctrl.max_abs * torch.tanh(raw_update / ctrl.max_abs)
    assert torch.allclose(scaled, activations)  # initial state is ones
    assert pending is not None
    assert torch.allclose(pending, expected)

    ctrl.commit()
    assert torch.allclose(ctrl.state, expected)

    ctrl.apply(activations, feature_dim=1, update=False)
    assert getattr(ctrl, "_pending_state", None) is None

    ctrl.reset(value=3.0)
    assert torch.allclose(ctrl.state, torch.full_like(ctrl.state, 3.0))


def test_state_controller_detach_toggle_controls_gradient_tracking():
    activations = torch.ones(2, 3, requires_grad=True)

    ctrl_detached = StateController(size=3, rho=0.0, beta=1.0, max_abs=10.0, init=1.0, detach=True)
    ctrl_detached.apply(activations, feature_dim=1, update=True)
    pending_detached = ctrl_detached._pending_state  # type: ignore[attr-defined]
    assert pending_detached is not None and not pending_detached.requires_grad

    ctrl_attached = StateController(size=3, rho=0.0, beta=1.0, max_abs=10.0, init=1.0, detach=False)
    ctrl_attached.apply(activations, feature_dim=1, update=True)
    pending_attached = ctrl_attached._pending_state  # type: ignore[attr-defined]
    assert pending_attached is not None and pending_attached.requires_grad


def test_state_controller_supports_negative_feature_dim_and_warning_reset():
    ctrl = StateController(size=2, rho=0.0, beta=5.0, max_abs=1.0, init=1.0, detach=True)
    tensor = torch.full((2, 3, 2), 5.0)

    with pytest.warns(RuntimeWarning, match="98% of max_abs"):
        ctrl.apply(tensor, feature_dim=-1, update=True)
    ctrl.commit()

    ctrl.reset_like_init(init=0.1)
    with pytest.warns(RuntimeWarning, match="98% of max_abs"):
        ctrl.apply(tensor, feature_dim=-1, update=True)
