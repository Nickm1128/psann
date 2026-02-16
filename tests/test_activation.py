import pytest

torch = pytest.importorskip("torch")

from psann.activations import ReLUSigmoidPSANN, SineParam


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
