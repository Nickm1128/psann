from __future__ import annotations

import torch

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
