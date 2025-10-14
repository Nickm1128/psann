from __future__ import annotations

import torch

from psann.models import WaveResNet
from psann.utils import jacobian_spectrum, mutual_info_proxy, ntk_eigens, participation_ratio


def _build_model() -> WaveResNet:
    return WaveResNet(
        input_dim=2,
        hidden_dim=24,
        depth=3,
        output_dim=2,
        context_dim=1,
        norm="none",
    )


def test_jacobian_spectrum_positive_eigs() -> None:
    torch.manual_seed(0)
    model = _build_model()
    x = torch.randn(4, 2)
    c = torch.randn(4, 1)
    spec = jacobian_spectrum(model, x, c, topk=4)
    assert {"top_eigs", "trace", "condition_number"}.issubset(spec.keys())
    assert torch.all(spec["top_eigs"] >= 0)


def test_ntk_eigens_matches_shapes() -> None:
    torch.manual_seed(1)
    model = _build_model()
    x = torch.randn(3, 2)
    c = torch.randn(3, 1)
    values = ntk_eigens(model, x, c, topk=3)
    assert "top_eigs" in values
    assert values["top_eigs"].ndim == 1
    assert torch.all(values["top_eigs"] >= 0)


def test_participation_ratio_and_mi_proxy() -> None:
    torch.manual_seed(2)
    feats = torch.randn(128, 16)
    contexts = torch.randn(128, 4)
    pr = participation_ratio(feats)
    mi = mutual_info_proxy(feats, contexts)
    assert pr > 0
    assert mi >= 0
