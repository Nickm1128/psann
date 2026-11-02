"""Standalone Hub exports preserve native inference without a PSANN install."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from psann.architectures import ActivationConfig, SpectralConfig
from psannlm import LMArchitectureConfig, LMConfig, LMTemporalConfig, PSANNLM, PSANNLMDataPrep
from psannlm.architectures import to_mapping
from psannlm.huggingface import export_huggingface
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.persistence import load_lm_checkpoint

TEXTS = ["hello world abc def é🙂 waves learn!", "other text for tiny language models"]


def make_checkpoint(tmp_path, architecture="residual", positional_encoding="rope", tokenizer=None):
    if tokenizer is None:
        tokenizer = PSANNLMDataPrep(TEXTS, tokenizer="simple").tokenizer
    config = LMConfig(
        architecture=architecture,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_mlp=24,
        vocab_size=tokenizer.vocab_size,
        positional_encoding=positional_encoding,
    )
    model = PSANNLM(config=config, device="cpu")
    model._ensure_model(tokenizer.vocab_size)
    model.attach_tokenizer(tokenizer)
    checkpoint = tmp_path / "native.pt"
    model.save(str(checkpoint))
    model._model.eval()
    return model, checkpoint


def load_export(path):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, local_files_only=True
    )
    return model, tokenizer


ARCHITECTURES = [
    "transformer",
    "residual",
    "wave",
    "geometric-sparse",
    LMArchitectureConfig.residual(spectral=SpectralConfig(k_fft=8)),
    *[
        LMArchitectureConfig.wave(temporal=LMTemporalConfig(mode=mode))
        for mode in ("interleave", "replace", "attention-only")
    ],
    LMArchitectureConfig.geometric_sparse(
        activation=ActivationConfig(
            kind="mixed", activation_types=("psann", "gelu"), activation_ratios=(0.5, 0.5)
        )
    ),
]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("positional_encoding", ["rope", "alibi", "sinusoidal"])
def test_logits_weights_greedy_and_resave(tmp_path, architecture, positional_encoding):
    native, checkpoint = make_checkpoint(tmp_path, architecture, positional_encoding)
    destination = export_huggingface(checkpoint, tmp_path / "hf", model_max_length=48)
    restored, tokenizer = load_export(destination)
    config = AutoConfig.from_pretrained(destination, trust_remote_code=True)
    assert config.lm_config == to_mapping(native._model.lm_config)
    assert config.use_cache is False
    for key, value in native._model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
    inputs = tokenizer("hello", return_tensors="pt")
    ids = inputs["input_ids"]
    torch.testing.assert_close(native._model(ids), restored(**inputs).logits, rtol=0, atol=0)
    expected = ids.clone()
    for _ in range(3):
        next_token = native._model(expected)[:, -1].argmax(dim=-1, keepdim=True)
        expected = torch.cat((expected, next_token), dim=1)
        if int(next_token.item()) == tokenizer.eos_token_id:
            break
    for _ in range(2):
        actual = restored.generate(**inputs, max_new_tokens=3, do_sample=False)
        assert torch.equal(actual, expected)
    again = tmp_path / "resaved"
    restored.save_pretrained(again)
    tokenizer.save_pretrained(again)
    second, second_tok = load_export(again)
    torch.testing.assert_close(
        second(**second_tok("hello", return_tensors="pt")).logits,
        restored(**inputs).logits,
        rtol=0,
        atol=0,
    )
    manifest = json.loads((destination / "runtime_manifest.json").read_text())
    for filename, entry in manifest["runtime_sources"].items():
        assert (
            hashlib.sha256((destination / filename).read_bytes()).hexdigest()
            == entry["bundled_sha256"]
        )


@pytest.mark.parametrize(
    "backend,passthrough",
    [("simple", False), ("sentencepiece", False), ("tokenizers", False), ("tokenizers", True)],
)
def test_tokenizer_exact_ids_text_specials_and_serialization(tmp_path, backend, passthrough):
    if backend != "simple":
        pytest.importorskip(backend)
    native = Tokenizer(
        TokenizerConfig(
            backend=backend, vocab_size=100, min_frequency=1, hf_passthrough_ids=passthrough
        )
    )
    native.fit(TEXTS)
    _, checkpoint = make_checkpoint(tmp_path, tokenizer=native)
    path = export_huggingface(checkpoint, tmp_path / "hf", model_max_length=128)
    _, restored = load_export(path)
    for text in ["hello", "é🙂", "", "unknown Ω", "[PAD] hello [EOS]", "<psann_id_1>"]:
        for specials in [False, True]:
            expected = native.encode(text, add_specials=specials)
            assert restored.encode(text, add_special_tokens=specials) == expected
            for skip in [False, True]:
                assert restored.decode(expected, skip_special_tokens=skip) == native.decode(
                    expected, skip_specials=skip
                )
    assert len(restored) == native.vocab_size
    assert sorted(restored.get_vocab().values()) == list(range(native.vocab_size))
    assert restored.padding_side == "left"
    encoded = restored(["hello world " * 4, ""], padding=True, return_tensors="pt")
    assert encoded["attention_mask"][1, 0] == 0
    again = tmp_path / "tokenizer_again"
    restored.save_pretrained(again)
    assert AutoTokenizer.from_pretrained(again, trust_remote_code=True).encode(
        "é🙂"
    ) == native.encode("é🙂")


def test_passthrough_special_ids_are_not_assumed_to_be_zero_to_three(tmp_path):
    from tokenizers import Tokenizer as Backend, models

    backend = Backend(
        models.WordLevel(
            {"hello": 0, "[UNK]": 1, "[EOS]": 2, "[PAD]": 3, "[BOS]": 4}, unk_token="[UNK]"
        )
    )
    native = Tokenizer.from_state(
        {
            "backend": "tokenizers",
            "model": backend.to_str(),
            "passthrough_ids": True,
            "special_ids": {"pad": 3, "bos": 4, "eos": 2, "unk": 1},
        }
    )
    _, checkpoint = make_checkpoint(tmp_path, tokenizer=native)
    path = export_huggingface(checkpoint, tmp_path / "hf", model_max_length=32)
    model, tokenizer = load_export(path)
    assert tokenizer.encode("hello") == native.encode("hello") == [4, 0, 2]
    assert tokenizer.pad_token_id == model.config.pad_token_id == 3
    assert tokenizer.bos_token_id == model.config.bos_token_id == 4


@pytest.mark.parametrize(
    "architecture",
    [
        "residual",
        LMArchitectureConfig.wave(temporal=LMTemporalConfig(mode="interleave")),
        LMArchitectureConfig.residual(spectral=SpectralConfig(k_fft=8)),
    ],
)
def test_padding_loss_sampling_and_validation(tmp_path, architecture):
    _, checkpoint = make_checkpoint(tmp_path, architecture)
    path = export_huggingface(checkpoint, tmp_path / "hf", model_max_length=32)
    model, tokenizer = load_export(path)
    for side in ("left", "right"):
        tokenizer.padding_side = side
        batch = tokenizer(["hello world", "hi"], padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[1, -2] = -100
        result = model(**batch, labels=labels)
        total, count = 0, 0
        for index, text in enumerate(["hello world", "hi"]):
            mask = batch["attention_mask"][index].bool()
            solo = model(**tokenizer(text, return_tensors="pt")).logits[0]
            torch.testing.assert_close(result.logits[index, mask], solo, rtol=0, atol=0)
            target = labels[index, mask][1:]
            total += torch.nn.functional.cross_entropy(
                solo[:-1], target, ignore_index=-100, reduction="sum"
            )
            count += int((target != -100).sum())
        torch.testing.assert_close(result.loss, total / count)
        result.loss.backward()
        assert model.model.embed.weight.grad is not None
    tokenizer.padding_side = "left"
    batch = tokenizer(["hello", "hi"], padding=True, return_tensors="pt")
    outputs = model.generate(**batch, max_new_tokens=2, do_sample=False)
    for index, text in enumerate(["hello", "hi"]):
        solo = tokenizer(text, return_tensors="pt")
        single_output = model.generate(**solo, max_new_tokens=2, do_sample=False)
        continuation = single_output[0, solo["input_ids"].shape[1] :]
        assert torch.equal(
            outputs[index, batch["input_ids"].shape[1] :][: len(continuation)], continuation
        )
    torch.manual_seed(21)
    sample1 = model.generate(**batch, max_new_tokens=2, do_sample=True, top_k=5)
    torch.manual_seed(21)
    assert torch.equal(sample1, model.generate(**batch, max_new_tokens=2, do_sample=True, top_k=5))
    with pytest.raises(ValueError, match="use_cache"):
        model.generate(**batch, max_new_tokens=1, use_cache=True)
    with pytest.raises(ValueError, match="empty row"):
        model(input_ids=torch.ones((1, 3), dtype=torch.long), attention_mask=torch.zeros((1, 3)))
    with pytest.raises(ValueError, match="contiguous"):
        model(
            input_ids=torch.ones((1, 3), dtype=torch.long), attention_mask=torch.tensor([[1, 0, 1]])
        )
    with pytest.raises(ValueError, match="model_max_length"):
        model(input_ids=torch.ones((1, 33), dtype=torch.long))
    ignored = torch.full_like(batch["input_ids"], -100)
    assert model(**batch, labels=ignored).loss.item() == 0
    single = torch.ones((1, 1), dtype=torch.long)
    assert model(input_ids=single, labels=single).loss.item() == 0
    assert len(model(**batch, return_dict=False)) == 1
    assert len(model(**batch, labels=ignored, return_dict=False)) == 2
    with torch.no_grad():
        model.get_output_embeddings().weight.zero_()
    model.generation_config.eos_token_id = 0  # All-zero logits deterministically choose ID 0.
    stopped = model.generate(**batch, max_new_tokens=5, do_sample=False)
    assert stopped.shape[1] == batch["input_ids"].shape[1] + 1


def test_isolated_offline_load_resave_and_sharding_without_psann(tmp_path):
    _, checkpoint = make_checkpoint(tmp_path)
    path = export_huggingface(
        checkpoint, tmp_path / "hf", model_max_length=32, max_shard_size="1KB"
    )
    assert (path / "model.safetensors.index.json").is_file()
    code = """
import importlib.abc, sys
from pathlib import Path
class BlockPSANN(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'psann', 'psannlm'}:
            raise RuntimeError('PSANN import attempted: ' + fullname)
sys.meta_path.insert(0, BlockPSANN())
from transformers import AutoTokenizer, AutoModelForCausalLM
root = Path(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True, local_files_only=True)
inputs = tokenizer('hello', return_tensors='pt')
expected = model.generate(**inputs, max_new_tokens=2)
model.save_pretrained(root.parent / 'again')
tokenizer.save_pretrained(root.parent / 'again')
tokenizer2 = AutoTokenizer.from_pretrained(root.parent / 'again', trust_remote_code=True)
model2 = AutoModelForCausalLM.from_pretrained(root.parent / 'again', trust_remote_code=True)
assert expected.equal(model2.generate(**tokenizer2('hello', return_tensors='pt'), max_new_tokens=2))
assert not any(n.split('.')[0] in {'psann', 'psannlm'} for n in sys.modules)
"""
    env = dict(
        os.environ,
        PYTHONPATH="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_MODULES_CACHE=str(tmp_path / "modules"),
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_raw_weights_require_config_and_tokenizer_and_keep_native_unchanged(tmp_path):
    native, checkpoint = make_checkpoint(tmp_path)
    original = checkpoint.read_bytes()
    weights = tmp_path / "raw.pt"
    torch.save(native._model.state_dict(), weights)
    path = tmp_path / "hf"
    with pytest.raises(ValueError, match="config"):
        export_huggingface(weights, path, model_max_length=32)
    with pytest.raises(ValueError, match="no tokenizer"):
        export_huggingface(
            weights, path, model_max_length=32, legacy_config=native._model.lm_config
        )
    export_huggingface(
        weights,
        path,
        model_max_length=32,
        legacy_config=native._model.lm_config,
        tokenizer=native._tokenizer,
    )
    assert checkpoint.read_bytes() == original
    assert load_lm_checkpoint(checkpoint).config == native._model.lm_config
    with pytest.raises(ValueError, match="empty directory"):
        export_huggingface(checkpoint, path, model_max_length=32)
    bad = tmp_path / "bad.pt"
    torch.save({"schema": "wrong", "schema_version": 1}, bad)
    with pytest.raises(ValueError, match="schema"):
        export_huggingface(bad, tmp_path / "bad", model_max_length=32)
    assert not (tmp_path / "bad").exists()


def test_missing_optional_dependency_and_conflicting_tokenizer(tmp_path, monkeypatch):
    _, checkpoint = make_checkpoint(tmp_path)
    different = PSANNLMDataPrep(["different"], tokenizer="simple").tokenizer
    with pytest.raises(ValueError, match="conflicts"):
        export_huggingface(checkpoint, tmp_path / "hf", model_max_length=32, tokenizer=different)
    original = builtins.__import__

    def no_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_transformers)
    with pytest.raises(ImportError, match="psannlm\\[huggingface\\]"):
        export_huggingface(checkpoint, tmp_path / "hf", model_max_length=32)


def test_trainer_checkpoint_external_tokenizer_and_cli_legacy_config(tmp_path):
    from psannlm import TrainConfig
    from psannlm.cli import main
    from psannlm.huggingface.tokenization_psann import PSANNTokenizer
    import yaml

    data = PSANNLMDataPrep(TEXTS, tokenizer="simple", max_length=4)
    config = LMConfig(
        "residual", d_model=16, n_layers=1, n_heads=2, d_mlp=24, vocab_size=data.vocab_size
    )
    native = PSANNLM(config=config, device="cpu")
    native.fit(
        data,
        train=TrainConfig(
            steps_per_epoch=1,
            batch_tokens=8,
            warmup_steps=0,
            amp="fp32",
            ddp="off",
            dataloader_num_workers=0,
            checkpoint_dir=str(tmp_path / "trainer"),
        ),
    )
    path = export_huggingface(
        tmp_path / "trainer/final.pt",
        tmp_path / "hf",
        model_max_length=32,
        tokenizer=data.tokenizer,
    )
    model, tok = load_export(path)
    ids = tok("hello", return_tensors="pt")["input_ids"]
    torch.testing.assert_close(native._model.eval()(ids), model(ids).logits, rtol=0, atol=0)
    weights = tmp_path / "weights.pt"
    torch.save(native._model.state_dict(), weights)
    original_config = tmp_path / "model.yaml"
    original_config.write_text(yaml.safe_dump(to_mapping(native._model.lm_config)))
    PSANNTokenizer(tokenizer_state=data.tokenizer.to_state()).save_pretrained(tmp_path / "tokens")
    assert (
        main(
            [
                "export-hf",
                "--ckpt",
                str(weights),
                "--output-dir",
                str(tmp_path / "cli"),
                "--max-length",
                "32",
                "--model-config",
                str(original_config),
                "--tokenizer-state",
                str(tmp_path / "tokens/psann_tokenizer.json"),
            ]
        )
        == 0
    )


def test_custom_builders_and_invalid_context_are_rejected_without_output(tmp_path, monkeypatch):
    from psannlm.architectures import registry

    _, checkpoint = make_checkpoint(tmp_path)
    for invalid in (0, -1, True, 2.5):
        with pytest.raises((TypeError, ValueError), match="model_max_length"):
            export_huggingface(checkpoint, tmp_path / "hf", model_max_length=invalid)
        assert not (tmp_path / "hf").exists()
    monkeypatch.setitem(
        registry._BUILDERS, "residual", lambda _: pytest.fail("Custom builder executed")
    )
    with pytest.raises(ValueError, match="built-in"):
        export_huggingface(checkpoint, tmp_path / "hf", model_max_length=32)


def test_optional_hf_module_import_does_not_import_transformers():
    code = """
import importlib.abc, sys
class BlockHF(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'transformers', 'safetensors'}:
            raise RuntimeError('Optional import attempted: ' + fullname)
sys.meta_path.insert(0, BlockHF())
import psannlm
from psannlm.huggingface import export_huggingface
assert callable(export_huggingface)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_exports_and_requires_context_length(tmp_path):
    from psannlm.cli import main

    _, checkpoint = make_checkpoint(tmp_path)
    with pytest.raises(SystemExit):
        main(["export-hf", "--ckpt", str(checkpoint), "--output-dir", str(tmp_path / "hf")])
    assert (
        main(
            [
                "export-hf",
                "--ckpt",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "hf"),
                "--max-length",
                "32",
            ]
        )
        == 0
    )
    assert (tmp_path / "hf/config.json").is_file()
