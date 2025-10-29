from __future__ import annotations

import torch
from torch import nn
import pytest

from psann.models import WaveResNet, build_wave_resnet


def test_wave_resnet_forward_with_context() -> None:
    torch.manual_seed(0)
    model = WaveResNet(
        input_dim=3,
        hidden_dim=32,
        depth=6,
        output_dim=2,
        context_dim=4,
        norm="rms",
        dropout=0.1,
    )
    x = torch.randn(16, 3)
    c = torch.randn(16, 4)
    out = model(x, c)
    assert out.shape == (16, 2)


def test_wave_resnet_backward_is_finite() -> None:
    torch.manual_seed(1)
    model = WaveResNet(input_dim=5, hidden_dim=48, depth=12, output_dim=4)
    x = torch.randn(8, 5)
    target = torch.randn(8, 4)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "Expected gradients to be populated."
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_build_wave_resnet_factory() -> None:
    torch.manual_seed(2)
    model = build_wave_resnet(
        input_dim=2,
        hidden_dim=16,
        depth=4,
        output_dim=1,
        context_dim=None,
    )
    x = torch.randn(10, 2)
    out = model(x)
    assert out.shape == (10, 1)


def test_wave_resnet_blocks_disable_context_modules_without_context_dim() -> None:
    model = WaveResNet(
        input_dim=3,
        hidden_dim=12,
        depth=3,
        output_dim=1,
        context_dim=None,
        use_film=True,
        use_phase_shift=True,
    )
    for block in model.blocks:
        assert block.film is None
        assert block.phase_shift is None
        assert block.use_film is False
        assert block.use_phase_shift is False


def test_wave_resnet_blocks_enable_context_modules_with_context_dim() -> None:
    context_dim = 2
    model = WaveResNet(
        input_dim=3,
        hidden_dim=10,
        depth=2,
        output_dim=1,
        context_dim=context_dim,
        use_film=True,
        use_phase_shift=True,
    )
    for block in model.blocks:
        assert block.use_film is True
        assert block.use_phase_shift is True
        assert block.film is not None
        assert block.phase_shift is not None
        # phase shift should project from context_dim to hidden_dim
        weight = block.phase_shift.weight
        assert weight.shape[1] == context_dim

    # disabling either knob removes the respective module even with context available
    film_off = WaveResNet(
        input_dim=3,
        hidden_dim=8,
        depth=1,
        output_dim=1,
        context_dim=context_dim,
        use_film=False,
        use_phase_shift=True,
    )
    assert film_off.blocks[0].film is None
    assert film_off.blocks[0].use_film is False

    phase_off = WaveResNet(
        input_dim=3,
        hidden_dim=8,
        depth=1,
        output_dim=1,
        context_dim=context_dim,
        use_film=True,
        use_phase_shift=False,
    )
    assert phase_off.blocks[0].phase_shift is None
    assert phase_off.blocks[0].use_phase_shift is False


@pytest.mark.parametrize("dropout_rate", [0.0, 0.25])
def test_wave_resnet_dropout_configuration(dropout_rate: float) -> None:
    model = WaveResNet(
        input_dim=4,
        hidden_dim=16,
        depth=2,
        output_dim=1,
        dropout=dropout_rate,
    )
    for block in model.blocks:
        if dropout_rate == 0.0:
            assert isinstance(block.dropout, nn.Identity)
        else:
            assert isinstance(block.dropout, nn.Dropout)
            assert block.dropout.p == pytest.approx(dropout_rate)


@pytest.mark.parametrize(
    "config",
    [
        {"context_dim": None, "norm": "none", "dropout": 0.0},
        {"context_dim": 3, "norm": "rms", "dropout": 0.1},
        {"context_dim": 2, "norm": "weight", "dropout": 0.2},
    ],
)
def test_wave_resnet_forward_production_configs(config: dict[str, object]) -> None:
    torch.manual_seed(4)
    model = WaveResNet(
        input_dim=6,
        hidden_dim=48,
        depth=5,
        output_dim=3,
        use_film=True,
        use_phase_shift=True,
        norm=config["norm"],  # type: ignore[arg-type]
        dropout=config["dropout"],  # type: ignore[arg-type]
        context_dim=config["context_dim"],  # type: ignore[arg-type]
    )
    batch = 8
    x = torch.randn(batch, 6)
    context_dim = config["context_dim"]
    if context_dim is None:
        output = model(x)
    else:
        c = torch.randn(batch, int(context_dim))  # type: ignore[arg-type]
        output = model(x, c)
    assert output.shape == (batch, 3)
    assert torch.isfinite(output).all()
