from __future__ import annotations

import torch
import pytest

from psann.conv import (
    PSANNConv1dNet,
    PSANNConv2dNet,
    PSANNConv3dNet,
    ResidualPSANNConv2dNet,
)


def test_psann_conv1d_segmentation_and_pooled_shapes() -> None:
    x = torch.randn(4, 3, 17)

    seg_net = PSANNConv1dNet(
        in_channels=3,
        out_dim=5,
        hidden_layers=2,
        conv_channels=8,
        hidden_channels=None,
        kernel_size=3,
        segmentation_head=True,
    )
    seg_out = seg_net(x)
    assert seg_out.shape == (4, 5, 17)

    pooled_net = PSANNConv1dNet(
        in_channels=3,
        out_dim=5,
        hidden_layers=2,
        conv_channels=8,
        hidden_channels=None,
        kernel_size=3,
        segmentation_head=False,
    )
    pooled_out = pooled_net(x)
    assert pooled_out.shape == (4, 5)

    x_short = torch.randn(4, 3, 9)
    pooled_out_short = pooled_net(x_short)
    assert pooled_out_short.shape == (4, 5)


def test_psann_conv2d_and_3d_segmentation_shapes() -> None:
    x2d = torch.randn(2, 4, 11, 9)
    net2d = PSANNConv2dNet(
        in_channels=4,
        out_dim=6,
        hidden_layers=3,
        conv_channels=12,
        hidden_channels=None,
        kernel_size=3,
        segmentation_head=True,
    )
    out2d = net2d(x2d)
    assert out2d.shape == (2, 6, 11, 9)

    pooled2d = PSANNConv2dNet(
        in_channels=4,
        out_dim=6,
        hidden_layers=3,
        conv_channels=12,
        hidden_channels=None,
        kernel_size=3,
        segmentation_head=False,
    )
    pooled2d_out = pooled2d(x2d)
    assert pooled2d_out.shape == (2, 6)

    x3d = torch.randn(1, 2, 5, 7, 9)
    net3d = PSANNConv3dNet(
        in_channels=2,
        out_dim=3,
        hidden_layers=1,
        conv_channels=4,
        hidden_channels=None,
        kernel_size=3,
        segmentation_head=True,
    )
    out3d = net3d(x3d)
    assert out3d.shape == (1, 3, 5, 7, 9)

    pooled3d = PSANNConv3dNet(
        in_channels=2,
        out_dim=3,
        hidden_layers=1,
        conv_channels=4,
        hidden_channels=None,
        kernel_size=3,
        segmentation_head=False,
    )
    pooled3d_out = pooled3d(x3d)
    assert pooled3d_out.shape == (1, 3)


def test_residual_conv2d_segmentation_vs_pooled_outputs() -> None:
    x = torch.randn(3, 2, 10, 8)

    seg_net = ResidualPSANNConv2dNet(
        in_channels=2,
        out_dim=7,
        hidden_layers=2,
        conv_channels=16,
        kernel_size=3,
        segmentation_head=True,
    )
    seg_out = seg_net(x)
    assert seg_out.shape == (3, 7, 10, 8)

    pooled_net = ResidualPSANNConv2dNet(
        in_channels=2,
        out_dim=7,
        hidden_layers=2,
        conv_channels=16,
        kernel_size=3,
        segmentation_head=False,
    )
    pooled_out = pooled_net(x)
    assert pooled_out.shape == (3, 7)


@pytest.mark.parametrize(
    ("net_cls", "kwargs"),
    [
        (PSANNConv1dNet, {"in_channels": 2, "out_dim": 4, "hidden_layers": 1}),
        (PSANNConv2dNet, {"in_channels": 2, "out_dim": 4, "hidden_layers": 1}),
        (PSANNConv3dNet, {"in_channels": 2, "out_dim": 4, "hidden_layers": 1}),
        (ResidualPSANNConv2dNet, {"in_channels": 2, "out_dim": 4, "hidden_layers": 1}),
    ],
)
def test_conv_alias_mismatch_raises(net_cls, kwargs) -> None:
    with pytest.raises(ValueError, match="must agree"):
        net_cls(conv_channels=8, hidden_channels=16, **kwargs)


@pytest.mark.parametrize(
    ("net_cls", "input_shape", "kwargs"),
    [
        (
            PSANNConv1dNet,
            (5, 3, 19),
            {
                "out_dim": 6,
                "hidden_layers": 3,
                "conv_channels": 12,
                "hidden_channels": 12,
                "kernel_size": 3,
            },
        ),
        (
            PSANNConv2dNet,
            (3, 4, 11, 9),
            {
                "out_dim": 5,
                "hidden_layers": 2,
                "conv_channels": 16,
                "hidden_channels": 16,
                "kernel_size": 3,
            },
        ),
        (
            PSANNConv3dNet,
            (2, 2, 7, 5, 3),
            {
                "out_dim": 4,
                "hidden_layers": 2,
                "conv_channels": 10,
                "hidden_channels": 10,
                "kernel_size": 3,
            },
        ),
        (
            ResidualPSANNConv2dNet,
            (2, 3, 13, 7),
            {"out_dim": 5, "hidden_layers": 3, "conv_channels": 24, "kernel_size": 3},
        ),
    ],
)
@pytest.mark.parametrize("segmentation_head", [False, True])
def test_conv_forward_production_configs(net_cls, input_shape, kwargs, segmentation_head) -> None:
    torch.manual_seed(6)
    batch, channels, *spatial = input_shape
    params = dict(kwargs)
    params.update({"segmentation_head": segmentation_head, "in_channels": channels})
    model = net_cls(**params)
    x = torch.randn(*input_shape)
    output = model(x)
    if segmentation_head:
        expected_shape = (batch, params["out_dim"], *spatial)
    else:
        expected_shape = (batch, params["out_dim"])
    assert output.shape == expected_shape
    assert torch.isfinite(output).all()
