"""Release metadata and source consumer contracts shared with wheel audits."""

import importlib
from pathlib import Path
import re
import tomllib
import warnings

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_migration_document_constructor_pairs_fit_identically_and_resave_twice(tmp_path):
    namespace = {}
    document = (ROOT / "docs/migration.md").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for code in re.findall(r"```python\n(.*?)```", document, re.S):
            exec(compile(code, "docs/migration.md", "exec"), namespace)
    for old, new in [*namespace["pairs"], (namespace["old"], namespace["new"])]:
        assert old.architecture == new.architecture
        assert old.preprocessor == new.preprocessor
        spatial = new.architecture.convolution is not None
        width = int(np.prod(new.architecture.geometry.shape)) if new.architecture.geometry else 4
        x = (
            np.random.default_rng(7)
            .normal(size=(16, 1, 6, 6) if spatial else (16, width))
            .astype(np.float32)
        )
        y = np.arange(16, dtype=np.float32).reshape(-1, 1) / 16
        for model in (old, new):
            model.set_params(
                epochs=2,
                hidden_layers=2,
                hidden_units=8,
                batch_size=8,
                device="cpu",
                random_state=317,
                early_stopping=False,
            )
            model.fit(x, y)
        np.testing.assert_allclose(old.predict(x), new.predict(x), rtol=0, atol=0)
        source = tmp_path / "source.pt"
        new.save(source)
        migrated = namespace["migrate_core_checkpoint"](
            source, [tmp_path / "generation-1.pt", tmp_path / "generation-2.pt"], x
        )
        assert migrated.architecture == new.architecture
        assert migrated.preprocessor == new.preprocessor
        np.testing.assert_allclose(migrated.predict(x), new.predict(x), rtol=0, atol=0)


def test_release_metadata_dependency_and_version_agreement():
    import psann

    core = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    lm = tomllib.loads((ROOT / "psannlm/pyproject.toml").read_text())["project"]
    assert core["version"] == lm["version"] == psann.__version__ == "2.0.1"
    assert core["requires-python"] == lm["requires-python"] == ">=3.9"
    assert {"numpy>=1.23", "torch>=2.1", "PyYAML>=6.0"} <= set(core["dependencies"])
    assert {
        "psann>=2.0.1",
        "numpy>=1.23",
        "torch>=2.1",
        "PyYAML>=6.0",
        "sentencepiece>=0.1.99",
        "tokenizers>=0.13",
        "datasets>=2.14",
        "huggingface-hub>=0.16.4",
    } <= set(lm["dependencies"])
    assert not any(item.startswith("psannlm") for item in core["dependencies"])


def test_every_documented_public_import_is_exported():
    document = (ROOT / "docs/public_api.md").read_text()
    for module, names in re.findall(r"^\| `(psann[^`]*)` \| (.+) \|$", document, re.M):
        surface = importlib.import_module(module)
        for name in re.findall(r"`([^`]+)`", names):
            assert getattr(surface, name) is not None, (module, name)
            assert name in surface.__all__, (module, name)


@pytest.mark.parametrize(
    "module", ["psann", "psann.sklearn", "psann.estimators", "psannlm", "psannlm.lm"]
)
def test_wildcard_exports_resolve_and_exclude_legacy_names(module):
    namespace = {}
    exec(f"from {module} import *", namespace)
    assert set(importlib.import_module(module).__all__) <= namespace.keys()
    legacy = {
        "ResPSANNRegressor",
        "ResConvPSANNRegressor",
        "SGRPSANNRegressor",
        "WaveResNetRegressor",
        "GeoSparseRegressor",
        "psannLM",
        "psannLMDataPrep",
    }
    assert not legacy.intersection(namespace)
    if module == "psann":
        assert not {"ActivationConfig", "AttentionConfig", "StateConfig"}.intersection(namespace)
        assert (
            namespace["ArchitectureConfig"]
            is importlib.import_module("psann.architectures").ArchitectureConfig
        )


@pytest.mark.parametrize("kind,legacy", [("wave", "waveresnet"), ("residual", "respsann")])
def test_parameter_count_cli_preserves_nondefault_graph_logits_and_gradients(
    kind, legacy, monkeypatch, capsys
):
    from scripts import count_psannlm_params as consumer

    built = []
    original = consumer.build_lm_model

    def capture(config):
        result = original(config)
        built.append(result.model)
        return result

    monkeypatch.setattr(consumer, "build_lm_model", capture)
    args = [
        "--vocab-size",
        "37",
        "--d-model",
        "24",
        "--n-layers",
        "2",
        "--n-heads",
        "3",
        "--d-mlp",
        "36",
        "--pos-enc",
        "alibi",
        "--wave-interleave",
        "--wave-kernel-size",
        "5",
        "--wave-dilation-growth",
        "2",
        "--wave-dropout",
        "0.0",
    ]
    for flag, value in [("--base", legacy), ("--architecture", kind)]:
        torch.manual_seed(733)
        assert consumer.main([flag, value, *args]) == 0
    assert built[0].lm_config == built[1].lm_config
    ids = torch.arange(14).reshape(2, 7)
    for model in built:
        model.eval()
    torch.testing.assert_close(built[0](ids), built[1](ids), rtol=0, atol=0)
    for model in built:
        model(ids).square().mean().backward()
    for (name, left), (_, right) in zip(built[0].named_parameters(), built[1].named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0, msg=name)
    output = capsys.readouterr().out.splitlines()
    assert output[:2] == output[2:]


@pytest.mark.parametrize("kind,legacy", [("wave", "waveresnet"), ("residual", "respsann")])
def test_kv_benchmark_canonical_and_compatibility_routes_generate_identically(
    kind, legacy, monkeypatch
):
    from scripts import benchmark_kv_cache as consumer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    models = []
    original = consumer.PSANNLM

    def capture(*args, **kwargs):
        model = original(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(consumer, "PSANNLM", capture)
    outputs = []
    for base in (legacy, kind):
        torch.manual_seed(313)
        outputs.append(
            consumer.run_benchmark(
                consumer.BenchmarkConfig(
                    batch_size=2,
                    prompt_length=24,
                    max_new_tokens=3,
                    base=base,
                    d_model=24,
                    n_layers=2,
                    n_heads=3,
                    tokenizer="simple",
                    positional_encoding="alibi",
                    device_mode="cpu",
                )
            )
        )
    assert models[0].config == models[1].config
    assert (
        outputs[0]["fast_path"]["sample_output_preview"]
        == outputs[1]["fast_path"]["sample_output_preview"]
    )
    ids = torch.arange(14).reshape(2, 7)
    torch.testing.assert_close(models[0]._model(ids), models[1]._model(ids), rtol=0, atol=0)
    assert outputs[0]["batch_tokens"] == outputs[1]["batch_tokens"] == 6
