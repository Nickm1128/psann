import pytest

torch = pytest.importorskip("torch")

from psann.activations import ReLUSigmoidPSANN, SigmoidParam, SineParam


def test_sineparam_forward_shape():
    act = SineParam(out_features=8, amplitude_init=1.0, frequency_init=1.0, decay_init=0.1)
    x = torch.randn(4, 8)
    y = act(x)
    assert y.shape == (4, 8)


def test_sineparam_vector_init_support():
    freq = torch.linspace(0.25, 2.0, steps=8)
    act = SineParam(out_features=8, amplitude_init=1.0, frequency_init=freq, decay_init=0.1)
    x = torch.randn(4, 8)
    y = act(x)
    assert y.shape == (4, 8)


def test_build_sine_freq_init_std_is_per_feature_and_reproducible():
    from psannlm.lm.models.sine import SineConfig, build_sine

    torch.manual_seed(0)
    cfg = SineConfig(freq_init=1.0, freq_init_std=0.5, trainable=False)
    act1 = build_sine(16, cfg)
    f1 = torch.nn.functional.softplus(act1._f).detach().clone()  # type: ignore[attr-defined]
    assert float(f1.std().item()) > 0.0

    torch.manual_seed(0)
    act2 = build_sine(16, cfg)
    f2 = torch.nn.functional.softplus(act2._f).detach()  # type: ignore[attr-defined]
    assert torch.allclose(f1, f2)


def test_relu_sigmoid_psann_is_zero_for_negative_and_clipped_to_one():
    act = ReLUSigmoidPSANN(out_features=6, slope_init=1.0, clip_max=1.0)
    x = torch.linspace(-2.0, 2.0, steps=18, dtype=torch.float32).reshape(3, 6)
    y = act(x)
    assert y.shape == x.shape
    assert torch.all(y[x < 0] == 0.0)
    assert float(y.max().item()) <= 1.0 + 1e-6


def test_relu_sigmoid_psann_slope_receives_grad_when_trainable():
    act = ReLUSigmoidPSANN(out_features=5, slope_trainable=True)
    x = torch.randn(4, 5, requires_grad=True)
    y = act(x).sum()
    y.backward()
    assert act._slope.grad is not None


def test_sigmoid_param_forward_shape_range_and_grad():
    act = SigmoidParam(out_features=6, slope_init=1.0, slope_trainable=True)
    x = torch.linspace(-3.0, 3.0, steps=18, dtype=torch.float32).reshape(3, 6)
    y = act(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)

    x_grad = x.clone().detach().requires_grad_(True)
    loss = act(x_grad).sum()
    loss.backward()
    assert act._slope.grad is not None


def test_sigmoid_param_vector_init_support():
    slope = torch.linspace(0.25, 2.0, steps=8)
    act = SigmoidParam(out_features=8, slope_init=slope)
    x = torch.randn(4, 8)
    y = act(x)
    assert y.shape == (4, 8)


def test_sigmoid_param_feature_dim_broadcasting():
    act = SigmoidParam(out_features=3, slope_init=0.8, feature_dim=1)
    x = torch.randn(2, 3, 5)
    y = act(x)
    assert y.shape == x.shape
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
