import torch

from psann.lm.models.registry import get_base


def test_param_count_and_forward():
    vocab = 128
    model = get_base("respsann")(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4, d_mlp=128)
    total = sum(p.numel() for p in model.parameters())
    assert total > 1000
    x = torch.randint(0, vocab, (1, 3))
    y = model(x)
    assert y.shape == (1, 3, vocab)
