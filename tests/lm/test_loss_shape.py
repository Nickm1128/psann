import torch
from torch import nn

from psann.lm.models.registry import get_base


def test_loss_scalar_shape():
    vocab = 128
    model = get_base("respsann")(vocab_size=vocab, d_model=64, n_layers=2, n_heads=4, d_mlp=128, dropout=0.0, rope=True)
    B, T = 2, 6
    x = torch.randint(0, vocab, (B, T), dtype=torch.long)
    logits = model(x)
    crit = nn.CrossEntropyLoss()
    loss = crit(logits.view(B*T, vocab), x.view(B*T))
    assert loss.dim() == 0
