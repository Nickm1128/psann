from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from psann.models import WaveEncoder
from psann.utils import encode_and_probe, fit_linear_probe, make_context_rotating_moons


def test_linear_probe_surpasses_baseline() -> None:
    torch.manual_seed(0)
    features, labels, contexts = make_context_rotating_moons(256, noise=0.02, seed=0)

    baseline = fit_linear_probe(
        torch.cat([features.float(), contexts.float()], dim=-1), labels.long()
    )

    encoder = WaveEncoder(
        input_dim=2,
        hidden_dim=32,
        depth=6,
        output_dim=16,
        context_dim=1,
        return_features=True,
    )
    dataset = TensorDataset(features.float(), labels.long(), contexts.float())
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    probe = encode_and_probe(
        encoder,
        loader,
        freeze_encoder=True,
        device="cpu",
    )

    assert probe["accuracy"] >= baseline["accuracy"]
    assert probe["feature_mean"].shape[0] == encoder.backbone.hidden_dim
    assert probe["feature_std"].shape == probe["feature_mean"].shape
    assert probe["effective_rank"] > 0
