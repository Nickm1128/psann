import torch
from torch import nn

from psann.lm.models.registry import get_base


def _tiny_model(base: str, vocab_size: int = 64):
    factory = get_base(base)
    return factory(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, d_mlp=128, dropout=0.0, rope=True)


def _loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    B, T, V = logits.shape
    crit = nn.CrossEntropyLoss()
    return crit(logits.view(B * T, V), targets.view(B * T))


def test_gradient_checkpointing_forward_backward_respsann():
    model = _tiny_model("respsann", 64)
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)  # type: ignore[attr-defined]
    x = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, 8, 64)
    loss = _loss(logits, x)
    loss.backward()
    assert any((p.grad is not None) for p in model.parameters() if p.requires_grad)


def test_gradient_checkpointing_forward_backward_waveresnet():
    model = _tiny_model("waveresnet", 64)
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)  # type: ignore[attr-defined]
    x = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(x)
    assert logits.shape == (2, 8, 64)
    loss = _loss(logits, x)
    loss.backward()
    assert any((p.grad is not None) for p in model.parameters() if p.requires_grad)

