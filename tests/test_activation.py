import pytest

torch = pytest.importorskip("torch")

from psann.activations import SineParam


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
