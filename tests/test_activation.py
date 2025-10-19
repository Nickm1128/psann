import pytest

torch = pytest.importorskip("torch")

from psann.activations import SineParam


def test_sineparam_forward_shape():
    act = SineParam(out_features=8, amplitude_init=1.0, frequency_init=1.0, decay_init=0.1)
    x = torch.randn(4, 8)
    y = act(x)
    assert y.shape == (4, 8)
