import torch

from psannlm.lm.models.sine import build_sine, SineConfig


def test_sine_build_forward_backward_shapes():
    torch.manual_seed(0)
    out_features = 16
    act = build_sine(
        out_features, SineConfig(amp_init=1.0, freq_init=1.0, damp_init=0.05, trainable=True)
    )
    x = torch.randn(8, out_features, dtype=torch.float32)
    y = act(x)
    assert y.shape == x.shape
    loss = (y**2).mean()
    loss.backward()
    # gradients should exist for parameters
    params = [p for p in act.parameters() if p.requires_grad]
    assert params and all(p.grad is not None for p in params)
