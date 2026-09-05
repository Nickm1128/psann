"""Maintained public consumer contracts execute their configuration boundaries."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from psann import PSANNRegressor
from psannlm.architectures import to_mapping

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path", sorted((ROOT / "configs/hisso").glob("*.yaml")), ids=lambda p: p.name
)
def test_hisso_public_configuration_constructs_and_fits(path, tmp_path):
    config = yaml.safe_load(path.read_text())
    params = dict(config["estimator"]["params"])
    params.update(device="cpu", epochs=1, hidden_units=8, batch_size=8)
    model = PSANNRegressor(**params)
    x = torch.randn(16, 4).numpy()
    y = x[:, :1] * 0.5
    model.fit(x, y)
    assert model.predict(x).shape == y.shape
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    restored = PSANNRegressor.load(checkpoint)
    torch.testing.assert_close(
        torch.from_numpy(restored.predict(x)), torch.from_numpy(model.predict(x)), rtol=0, atol=0
    )


CONTRACTS = json.loads((ROOT / "tests/fixtures/benchmark_model_contracts.json").read_text())


@pytest.mark.parametrize("path", CONTRACTS)
def test_benchmark_sweeps_preserve_every_normalized_model_parameter(path):
    from scripts._bench_lm_bases.models import benchmark_model_config
    from scripts._bench_lm_bases.sweep import _expand_sweep_configs

    config = yaml.safe_load((ROOT / path).read_text())
    resolved = []
    for variant in _expand_sweep_configs(config):
        models = {
            name: to_mapping(benchmark_model_config(variant["cfg"], name))
            for name in config["models"]
        }
        resolved.append(
            hashlib.sha256(
                json.dumps(models, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
    assert len(resolved) == CONTRACTS[path]["sweep_count"]
    assert sorted(resolved) == CONTRACTS[path]["resolved_config_hashes"]


def test_advanced_convolution_builder_preserves_logits_and_gradients():
    from examples.torch_backbone import build_backbone
    from psann.architectures import ArchitectureConfig, ConvolutionConfig
    from psann.conv import PSANNConv2dNet

    torch.manual_seed(17)
    original = PSANNConv2dNet(1, 3, hidden_layers=2, hidden_channels=32, kernel_size=3)
    torch.manual_seed(17)
    canonical = build_backbone(
        ArchitectureConfig.convolutional(convolution=ConvolutionConfig(kernel_size=3)), (1, 8, 8), 3
    )
    x = torch.randn(4, 1, 8, 8)
    expected = original(x)
    actual = canonical(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for old, new in zip(original.parameters(), canonical.parameters()):
        torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)
