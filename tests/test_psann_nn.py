from __future__ import annotations

import torch
from torch import nn

import pytest

from psann.nn import PSANNNet, PSANNBlock, ResidualPSANNBlock


def _identity_initialise(linear: nn.Linear) -> None:
    """Set a square Linear layer to the identity map."""
    eye = torch.eye(
        linear.out_features,
        linear.in_features,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    linear.weight.data.copy_(eye)
    if linear.bias is not None:
        linear.bias.data.zero_()


def test_residual_block_droppath_toggles_with_training_mode() -> None:
    torch.manual_seed(0)
    block = ResidualPSANNBlock(
        dim=4,
        activation_type="relu",
        norm="none",
        drop_path=0.5,
        residual_alpha_init=1.0,
    )
    _identity_initialise(block.fc1)
    _identity_initialise(block.fc2)

    x = torch.ones(6, 4)

    block.train()
    torch.manual_seed(0)
    out_train = block(x)

    block.eval()
    out_eval = block(x)

    assert torch.allclose(out_eval, torch.full_like(out_eval, 2.0))
    train_unique = torch.unique(out_train)
    assert set(train_unique.tolist()) == {1.0, 3.0}


@pytest.mark.parametrize("rho,beta", [(0.0, 0.5), (0.5, 1.0)])
def test_psann_net_state_commit_reset_and_disable_updates(rho: float, beta: float) -> None:
    state_cfg = {"rho": rho, "beta": beta, "init": 1.0, "max_abs": 10.0, "detach": True}
    net = PSANNNet(
        input_dim=3,
        output_dim=2,
        hidden_layers=1,
        hidden_units=4,
        hidden_width=None,
        state_cfg=state_cfg,
        activation_type="relu",
    )
    assert isinstance(net.body[0], PSANNBlock)
    ctrl = net.body[0].state_ctrl
    assert ctrl is not None

    net.train()
    x = torch.randn(5, 3)

    init_state = ctrl.state.clone()
    out = net(x)
    assert out.shape == (5, 2)
    assert torch.allclose(ctrl.state, init_state)
    pending = getattr(ctrl, "_pending_state", None)
    assert pending is not None

    net.commit_state_updates()
    updated_state = ctrl.state.clone()
    assert not torch.allclose(updated_state, init_state)
    assert getattr(ctrl, "_pending_state", None) is None

    net.reset_state()
    assert torch.allclose(ctrl.state, torch.full_like(ctrl.state, 1.0))

    net.set_state_updates(False)
    net.train()
    net(x)
    assert getattr(ctrl, "_pending_state", None) is None

    net.set_state_updates(True)
    net(x)
    assert getattr(ctrl, "_pending_state", None) is not None
