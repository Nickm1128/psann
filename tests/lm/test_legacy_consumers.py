"""Characterize real consumers before changing LM normalization."""

import ast
from psannlm.architectures import to_mapping
from pathlib import Path

import pytest
import torch
import yaml

from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig

ROOT = Path(__file__).resolve().parents[2]
TEXTS = ["hello world networks learn words", "sine waves model sequences of text"] * 3


def test_streaming_training_cli_checkpoint_and_export(tmp_path, monkeypatch):
    from psannlm.train import main
    from psannlm._train import cli

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    text = tmp_path / "text.txt"
    text.write_text("\n".join(TEXTS), encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(str(text), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    export = tmp_path / "export"
    captured = {}
    factory = cli.build_lm_model

    def observed_factory(config):
        result = factory(config)
        captured["before"] = {k: v.clone() for k, v in result.model.state_dict().items()}
        captured["config"] = to_mapping(result.model.lm_config)
        return result

    monkeypatch.setattr(cli, "build_lm_model", observed_factory)
    assert (
        main(
            [
                "--data-manifest",
                str(manifest),
                "--tokenizer-backend",
                "simple",
                "--base",
                "respsann",
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
                "--sine-freq-init",
                "0.7",
                "--sine-amp-init-std",
                "0.2",
                "--max-length",
                "8",
                "--batch-tokens",
                "16",
                "--num-workers",
                "0",
                "--max-steps",
                "2",
                "--warmup-steps",
                "0",
                "--amp",
                "fp32",
                "--ddp",
                "off",
                "--checkpoint-dir",
                str(checkpoint),
                "--export-dir",
                str(export),
            ]
        )
        == 0
    )
    saved = torch.load(checkpoint / "final.pt", weights_only=True)
    assert saved["state"]["step"] == 2
    assert captured["config"]["architecture"]["activation"]["frequency_init"] == 0.7
    assert captured["config"]["architecture"]["activation_initialization"]["amplitude_std"] == 0.2
    assert not torch.equal(saved["model"]["lm_head.weight"], captured["before"]["lm_head.weight"])
    assert saved["model"].keys() == captured["before"].keys()
    assert (export / "model.pt").read_bytes() == (checkpoint / "final.pt").read_bytes()
    import json

    meta = json.loads((export / "psann_artifacts.json").read_text())
    assert meta["model"] == saved["config"]
    assert meta["artifact_kind"] == "psannlm.trainer"


def test_yaml_training_cli_model_checkpoint(tmp_path, monkeypatch):
    from psannlm.lm.train.cli import main
    from psannlm.lm.api import PSANNLM

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    text = tmp_path / "texts.txt"
    text.write_text("\n".join(TEXTS), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "base": "waveresnet",
                    "d_model": 24,
                    "n_layers": 2,
                    "n_heads": 3,
                    "d_mlp": 36,
                    "positional_encoding": "sinusoidal",
                    "sine_params": {"freq_init": 0.7, "amp_init_std": 0.2},
                },
                "data": {"sources": [{"path": str(text)}], "tokenizer": "simple", "max_length": 8},
                "train": {
                    "epochs": 1,
                    "batch_tokens": 16,
                    "lr": 0.003,
                    "amp": "fp32",
                    "ddp": "off",
                    "checkpoint_dir": str(tmp_path / "output"),
                },
            }
        ),
        encoding="utf-8",
    )
    seen = {}
    original = PSANNLM._ensure_model

    def build(self, vocabulary):
        model = original(self, vocabulary)
        if "before" not in seen:
            seen["before"] = model.lm_head.weight.detach().clone()
        return model

    monkeypatch.setattr(PSANNLM, "_ensure_model", build)
    assert main(["--config", str(config)]) == 0
    payload = torch.load(tmp_path / "output/final_model.pt", weights_only=True)
    assert payload["config"]["positional_encoding"] == "sinusoidal"
    assert payload["config"]["architecture"]["activation"]["frequency_init"] == 0.7
    assert payload["config"]["architecture"]["activation_initialization"]["amplitude_std"] == 0.2
    assert not torch.equal(payload["state_dict"]["lm_head.weight"], seen["before"])
    restored = PSANNLM.load(str(tmp_path / "output/final_model.pt"))
    torch.testing.assert_close(
        restored._model.lm_head.weight, payload["state_dict"]["lm_head.weight"]
    )


def test_unified_resume_requires_checkpoint():
    from psannlm.cli import main

    with pytest.raises(SystemExit, match="resume requires --resume-ckpt"):
        main(["resume", "--data-manifest", "not-opened.txt"])


@pytest.mark.parametrize("backend", ["simple", "sentencepiece", "tokenizers"])
def test_tokenizer_ids_text_and_artifact_behavior(backend, tmp_path):
    model_path = (
        ROOT / "examples/lm/tokenizer/sample_texts.model" if backend == "sentencepiece" else None
    )
    cfg = TokenizerConfig(
        backend=backend,
        model_path=str(model_path) if model_path else None,
        vocab_size=128,
        min_frequency=1,
    )
    tokenizer = Tokenizer(cfg)
    tokenizer.fit(TEXTS)
    ids = tokenizer.encode("hello world", add_specials=True)
    assert ids[0] == tokenizer.bos_id and ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "hello world"
    if backend in {"simple", "sentencepiece"}:
        with pytest.raises(NotImplementedError, match="save.*not implemented"):
            tokenizer.save(str(tmp_path / "simple.json"))
        if backend == "sentencepiece":
            restored = Tokenizer(cfg)
            restored.fit([])
            assert restored.encode("hello world", add_specials=True) == ids
    else:
        suffix = ".model" if backend == "sentencepiece" else ".json"
        dest = tmp_path / ("tokenizer" + suffix)
        tokenizer.save(str(dest))
        restored = Tokenizer(TokenizerConfig(backend=backend, model_path=str(dest)))
        restored.fit([])
        assert restored.encode("hello world", add_specials=True) == ids
        assert restored.decode(ids) == tokenizer.decode(ids)


def test_current_core_import_inventory():
    modules = set()
    for path in (ROOT / "psannlm").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("psann.")
            ):
                modules.add(node.module)
    assert modules == {"psann.architectures", "psann.architectures.components", "psann.utils"}
    for path in (ROOT / "src/psann").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("psannlm")
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("psannlm") for alias in node.names)
