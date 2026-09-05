"""Canonical entrypoints execute real bounded training and reconstruction."""

from copy import deepcopy
import json
from pathlib import Path
import re
import runpy
import sys

import pytest
import torch
import yaml

from psannlm import PSANNLM
from psannlm.cli import main
from psannlm import LMConfig
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig

ROOT = Path(__file__).resolve().parents[2]
TEXT = "abc def ghij klmn opq rst uvw xyz " * 8


def fitted_tokenizer(tmp_path):
    tok = Tokenizer(
        TokenizerConfig(
            backend="tokenizers", vocab_size=128, min_frequency=1, hf_passthrough_ids=True
        )
    )
    tok.fit([TEXT, "User: explain waves Assistant: networks learn useful sequences"] * 4)
    tok.save(
        str(tmp_path / "tokenizer.json"),
        special_tokens_map_path=str(tmp_path / "special_tokens_map.json"),
    )
    return tok


@pytest.mark.parametrize("kind", ["transformer", "residual", "wave", "geometric-sparse"])
def test_sft_uses_saved_nondefault_config_and_updates_response_loss_parameters(
    kind, tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from psannlm.sft import main as sft

    tok = fitted_tokenizer(tmp_path)
    config = LMConfig(
        architecture=kind,
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=tok.vocab_size,
        positional_encoding="alibi",
    )
    lm = PSANNLM(config=config, device="cpu")
    lm._ensure_model(tok.vocab_size)
    lm.attach_tokenizer(tok)
    source = tmp_path / "source.pt"
    lm.save(source)
    before = lm._model.lm_head.weight.detach().clone()
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "\n".join(
            json.dumps({"prompt": "explain waves", "response": "networks learn useful sequences"})
            for _ in range(8)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "sft"
    assert (
        sft(
            [
                "--init-ckpt",
                str(source),
                "--tokenizer-dir",
                str(tmp_path),
                "--sft-source",
                "pairs",
                "--dataset",
                "json",
                "--data-files",
                str(pairs),
                "--checkpoint-dir",
                str(out),
                "--seq-len",
                "7",
                "--batch-tokens",
                "14",
                "--grad-accum-steps",
                "2",
                "--max-steps",
                "3",
                "--lr",
                "0.004",
                "--warmup-steps",
                "0",
                "--amp",
                "fp32",
                "--num-workers",
                "0",
            ]
        )
        == 0
    )
    payload = torch.load(out / "final.pt", weights_only=True)
    assert payload["config"] == torch.load(source, weights_only=True)["config"]
    assert payload["state"]["step"] == 3
    assert not torch.equal(payload["model"]["lm_head.weight"], before)
    assert all(v["step"].item() == 3 for v in payload["optim"]["state"].values())


def test_benchmark_main_all_five_factories_train_eval_and_save_canonical_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    import importlib

    benchmark = importlib.import_module("scripts._bench_lm_bases.main")
    fitted_tokenizer(tmp_path)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"text": TEXT}) for _ in range(8)), encoding="utf-8")
    cfg = {
        "bench": {
            "run_name": "tiny",
            "out_dir": str(tmp_path / "bench"),
            "bases": ["transformer", "respsann", "sgrpsann", "waveresnet", "geosparse"],
            "seeds": [137],
        },
        "data": {
            "dataset": "json",
            "data_files": str(corpus),
            "train_split": "train",
            "val_split": "train",
            "max_length": 7,
            "shuffle": False,
        },
        "tokenizer": {
            "backend": "tokenizers",
            "model_path": str(tmp_path / "tokenizer.json"),
            "special_tokens_map_path": str(tmp_path / "special_tokens_map.json"),
            "save_dir": str(tmp_path / "tok"),
        },
        "train": {
            "d_model": 24,
            "n_layers": 2,
            "n_heads": 3,
            "d_mlp": 36,
            "max_steps": 3,
            "batch_tokens": 14,
            "lr": 0.004,
            "warmup_steps": 0,
            "amp": "fp32",
            "ddp": "off",
            "positional_encoding": "alibi",
        },
        "eval": {"max_batches": 2, "max_tokens": 28, "batch_tokens": 14},
    }
    file = tmp_path / "bench.yaml"
    file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["benchmark", "--config", str(file)])
    seen = []
    original = benchmark.build_lm_model

    def build(config):
        result = original(config)
        seen.append((result.model, result.model.lm_head.weight.detach().clone()))
        return result

    monkeypatch.setattr(benchmark, "build_lm_model", build)
    assert benchmark.main() == 0
    assert len(seen) == 5
    for model, before in seen:
        assert not torch.equal(before, model.lm_head.weight)
    artifacts = list((tmp_path / "bench").rglob("final_model.pt"))
    assert len(artifacts) == 5
    for file in artifacts:
        model = PSANNLM.load(file, map_location="cpu")
        payload = torch.load(file.parent / "checkpoints/final.pt", weights_only=True)
        assert payload["config"] == torch.load(file, weights_only=True)["config"]
        assert payload["state"]["step"] == 3
        torch.testing.assert_close(
            model._model.lm_head.weight, payload["model"]["lm_head.weight"], rtol=0, atol=0
        )
        metrics = json.loads((file.parent / "metrics.json").read_text())
        assert metrics["status"] == "ok" and metrics["val_tokens"] == 28


@pytest.mark.parametrize("kind", ["transformer", "residual", "wave", "geometric-sparse"])
@pytest.mark.parametrize("config_equals", [False, True], ids=["config-spaced", "config-equals"])
@pytest.mark.parametrize("resume_equals", [False, True], ids=["resume-spaced", "resume-equals"])
def test_unified_cli_real_yaml_train_resume_eval_generate(
    kind, config_equals, resume_equals, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    text = tmp_path / "texts.txt"
    text.write_text(TEXT, encoding="utf-8")
    output = tmp_path / "output"
    raw = {
        "model": {
            "kind": "lm",
            "architecture": kind,
            "d_model": 24,
            "n_layers": 2,
            "n_heads": 3,
            "d_mlp": 36,
            "positional_encoding": "alibi",
        },
        "data": {"sources": [str(text)], "tokenizer": "simple", "max_length": 7},
        "train": {
            "kind": "train",
            "lr": 0.003,
            "warmup_steps": 0,
            "amp": "fp32",
            "ddp": "off",
            "batch_tokens": 14,
            "steps_per_epoch": 3,
            "checkpoint_dir": str(output),
        },
    }
    original = deepcopy(raw)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config_args = [f"--config={config}"] if config_equals else ["--config", str(config)]
    assert main(["train", *config_args]) == 0
    first = torch.load(output / "final.pt", weights_only=True)
    assert first["state"]["step"] == 3
    assert first["config"]["architecture"]["kind"] == kind
    assert yaml.safe_load(config.read_text()) == original
    raw["train"]["steps_per_epoch"] = 6
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    resume_args = (
        [f"--resume-ckpt={output / 'final.pt'}"]
        if resume_equals
        else ["--resume-ckpt", str(output / "final.pt")]
    )
    assert main(["resume", *config_args, *resume_args]) == 0
    final = torch.load(output / "final.pt", weights_only=True)
    assert final["state"]["step"] == 6 and final["config"] == first["config"]
    assert not torch.equal(final["model"]["lm_head.weight"], first["model"]["lm_head.weight"])
    artifact = output / "final_model.pt"
    lm = PSANNLM.load(artifact, map_location="cpu")
    data_file = tmp_path / "eval.jsonl"
    data_file.write_text(json.dumps({"text": TEXT}) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert (
        main(
            [
                "eval",
                "--ckpt",
                str(artifact),
                "--dataset",
                "json",
                "--data-files",
                str(data_file),
                "--seq-len",
                "7",
                "--max-batches",
                "3",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    result = capsys.readouterr().out
    match = re.search(r"tokens=(\d+) loss=([\d.]+)", result)
    assert match and int(match[1]) == 21
    tokens = lm._tokenizer.encode(TEXT, add_specials=False)
    lm._model.eval()
    losses = []
    for start in (0, 7, 14):
        ids = torch.tensor([tokens[start : start + 8]])
        logits = lm._model(ids[:, :-1])
        losses.append(
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten()).item()
        )
    assert abs(float(match[2]) - sum(losses) / 3) < 0.000051
    from psannlm import cli

    sampled = []
    original_sample = cli.sample_next_token

    def sample(logits, **kwargs):
        result = original_sample(logits, **kwargs)
        sampled.append(result.item())
        return result

    monkeypatch.setattr(cli, "sample_next_token", sample)
    torch.manual_seed(653)
    assert (
        main(
            [
                "generate",
                "--ckpt",
                str(artifact),
                "--prompt",
                "abc",
                "--max-new-tokens",
                "9",
                "--no-stop-at-eos",
                "--temperature",
                "0",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    generated = capsys.readouterr().out
    ids = torch.tensor([lm._tokenizer.encode("abc", add_specials=False)])
    expected = []
    for _ in range(9):
        token = lm._model(ids)[:, -1].argmax(-1)
        expected.append(token.item())
        ids = torch.cat([ids, token[:, None]], 1)
    assert sampled == expected
    assert "[output]\n" + cli._pretty_detok(lm._tokenizer.decode(expected)) in generated


@pytest.mark.parametrize("example", ["minimal_train.py", "generate.py"])
def test_maintained_examples_execute_builder_optimizer_and_generation(
    example, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        (
            [example, "--epochs", "1", "--repeat", "1", "--out", str(tmp_path / "example")]
            if example == "minimal_train.py"
            else [example]
        ),
    )
    from psannlm.lm import api

    observed = []
    build = api.build_lm_model

    def capture(config):
        result = build(config)
        observed.append((result.model, result.model.lm_head.weight.detach().clone()))
        return result

    monkeypatch.setattr(api, "build_lm_model", capture)
    runpy.run_path(str(ROOT / "examples/lm" / example), run_name="__main__")
    assert len(observed) == 1
    model, before = observed[0]
    assert not torch.equal(before, model.lm_head.weight)
    payload = torch.load(tmp_path / "runs/lm/exp/final.pt", map_location="cpu", weights_only=True)
    assert payload["schema"] == "psannlm.trainer" and payload["state"]["step"] > 0
    torch.testing.assert_close(
        payload["model"]["lm_head.weight"], model.lm_head.weight.cpu(), rtol=0, atol=0
    )
    if example == "minimal_train.py":
        output = json.loads((tmp_path / "example/generations.json").read_text())
        assert set(output["generations"]) == set(output["prompts"])


@pytest.mark.parametrize("entrypoint", ["canonical", "legacy-stream", "legacy-yaml"])
@pytest.mark.parametrize("equals", [False, True], ids=["spaced", "equals"])
def test_training_entrypoints_delegate_identical_nondefault_config_updates_and_export(
    entrypoint, equals, tmp_path, capsys
):
    from psannlm.architectures import LMArchitectureConfig, to_mapping
    from psann.architectures import SpectralConfig
    from psannlm.train import main as legacy_stream
    from psannlm.lm.train.cli import main as legacy_yaml

    tok = fitted_tokenizer(tmp_path)
    text = tmp_path / "texts.txt"
    text.write_text(TEXT * 4, encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(str(text), encoding="utf-8")
    config = LMConfig(
        LMArchitectureConfig.residual(spectral=SpectralConfig(k_fft=3, strength=0.37)),
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=tok.vocab_size,
        positional_encoding="alibi",
    )
    config_file = tmp_path / "model.json"
    config_file.write_text(json.dumps(to_mapping(config)), encoding="utf-8")
    common = [
        "--data-manifest",
        str(manifest),
        "--tokenizer-backend",
        "tokenizers",
        "--tokenizer-model-path",
        str(tmp_path / "tokenizer.json"),
        "--tokenizer-special-map-path",
        str(tmp_path / "special_tokens_map.json"),
        "--model-config",
        str(config_file),
        "--max-length",
        "7",
        "--batch-tokens",
        "14",
        "--max-steps",
        "3",
        "--warmup-steps",
        "0",
        "--lr",
        "0.003",
        "--amp",
        "fp32",
        "--ddp",
        "off",
        "--device",
        "cpu",
        "--num-workers",
        "0",
    ]
    capsys.readouterr()
    torch.manual_seed(733)
    assert main(["train", *common, "--checkpoint-dir", str(tmp_path / "reference")]) == 0
    capsys.readouterr()
    torch.manual_seed(733)
    function = {
        "canonical": lambda argv: main(["train", *argv]),
        "legacy-stream": legacy_stream,
        "legacy-yaml": legacy_yaml,
    }[entrypoint]
    entry = function
    function = lambda argv: entry(
        [f"{name}={value}" for name, value in zip(argv[::2], argv[1::2])] if equals else argv
    )
    assert (
        function(
            [
                *common,
                "--checkpoint-dir",
                str(tmp_path / "actual"),
                "--export-dir",
                str(tmp_path / "export"),
            ]
        )
        == 0
    )
    output = capsys.readouterr()
    assert output.err.count("DeprecationWarning") == (entrypoint != "canonical")
    reference = torch.load(tmp_path / "reference/final.pt", weights_only=True)
    actual = torch.load(tmp_path / "actual/final.pt", weights_only=True)
    assert actual["config"] == reference["config"] == to_mapping(config)
    assert actual["state"]["step"] == 3 and tok.vocab_size > 20
    for key in reference["model"]:
        torch.testing.assert_close(actual["model"][key], reference["model"][key], rtol=0, atol=0)
    metadata = json.loads((tmp_path / "export/psann_artifacts.json").read_text())
    assert metadata["model"] == actual["config"] and metadata["schema_version"] == 1
    assert (tmp_path / "export/model.pt").read_bytes() == (
        tmp_path / "actual/final.pt"
    ).read_bytes()
    resumed = list(common)
    resumed[resumed.index("--max-steps") + 1] = "6"
    assert (
        main(
            [
                "resume",
                *resumed,
                "--checkpoint-dir",
                str(tmp_path / "actual"),
                "--resume-ckpt",
                str(tmp_path / "actual/final.pt"),
            ]
        )
        == 0
    )
    final = torch.load(tmp_path / "actual/final.pt", weights_only=True)
    assert final["state"]["step"] == 6 and final["config"] == actual["config"]
    assert not torch.equal(final["model"]["lm_head.weight"], actual["model"]["lm_head.weight"])


@pytest.mark.parametrize("mode", ["disabled", "interleave", "replace", "attention-only"])
def test_cli_legacy_raw_wave_uses_explicit_config_for_exact_greedy_tokens(
    mode, tmp_path, monkeypatch, capsys
):
    from psannlm.architectures import LMArchitectureConfig, build_lm_model, to_mapping
    from psannlm import cli

    tok = fitted_tokenizer(tmp_path)
    config = LMConfig(
        LMArchitectureConfig.wave(temporal={"mode": mode}),
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=tok.vocab_size,
        positional_encoding="alibi",
    )
    model = build_lm_model(config).model.eval()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    config_file = tmp_path / "model.json"
    config_file.write_text(json.dumps(to_mapping(config)), encoding="utf-8")
    expected = []
    ids = torch.tensor([tok.encode("abc", add_specials=False)])
    for _ in range(7):
        token = model(ids)[:, -1].argmax(-1)
        expected.append(token.item())
        ids = torch.cat([ids, token[:, None]], 1)
    actual = []
    sample = cli.sample_next_token

    def capture(logits, **kwargs):
        token = sample(logits, **kwargs)
        actual.append(token.item())
        return token

    monkeypatch.setattr(cli, "sample_next_token", capture)
    assert (
        main(
            [
                "generate",
                "--ckpt",
                str(path),
                "--model-config",
                str(config_file),
                "--tokenizer-dir",
                str(tmp_path),
                "--prompt",
                "abc",
                "--max-new-tokens",
                "7",
                "--temperature",
                "0",
                "--no-stop-at-eos",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert actual == expected
    assert "[output]" in capsys.readouterr().out


def test_chat_pipeline_sizing_pretrain_sft_and_saved_model_execute_canonical_builder(
    tmp_path, monkeypatch
):
    pytest.importorskip("transformers")
    from scripts import train_psannlm_chat as chat
    from psannlm.lm import api
    from tokenizers import Tokenizer as HFTokenizer
    from transformers import PreTrainedTokenizerFast

    fitted_tokenizer(tmp_path)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=HFTokenizer.from_file(str(tmp_path / "tokenizer.json")),
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    fast.save_pretrained(tmp_path / "hf")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    landing = chat.land_config(
        "tiny",
        30000,
        vocab_size=fast.vocab_size,
        base="waveresnet",
        width_choices=[24],
        layer_choices=[2],
        max_heads=3,
    )
    assert landing.d_model == 24 and landing.n_layers == 2
    monkeypatch.setattr(chat, "land_config", lambda *a, **k: landing)
    monkeypatch.setattr(
        chat,
        "build_wikitext_stream",
        lambda: chat.TextStream(lambda: [TEXT * 8], name="local-text"),
    )
    monkeypatch.setattr(
        chat,
        "build_oasst_pair_stream",
        lambda count: chat.TextStream(lambda: [TEXT * 8], name="local-pairs"),
    )
    seen = []
    build = api.build_lm_model

    def capture(config):
        result = build(config)
        seen.append((result.model, result.model.lm_head.weight.detach().clone()))
        return result

    monkeypatch.setattr(api, "build_lm_model", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chat",
            "--tokenizer",
            str(tmp_path / "hf"),
            "--save-dir",
            str(tmp_path / "out"),
            "--pretrain-steps",
            "3",
            "--sft-steps",
            "3",
            "--tokens-per-step",
            "14",
            "--seq-len",
            "7",
            "--pretrain-grad-accum",
            "1",
            "--sft-grad-accum",
            "1",
            "--warmup-steps",
            "0",
            "--amp",
            "fp32",
        ],
    )
    chat.main()
    (artifact,) = (tmp_path / "out").rglob("psannlm_chat_final.pt")
    saved = torch.load(artifact, weights_only=True)
    before_sft = torch.load(artifact.parent / "checkpoint_pretrain.pt", weights_only=True)
    assert len(seen) == 1
    from psannlm.architectures import build_lm_model, LMArchitectureConfig

    counted = build_lm_model(
        LMConfig(
            LMArchitectureConfig.wave(temporal={"mode": "interleave"}),
            d_model=24,
            n_layers=2,
            n_heads=3,
            d_mlp=96,
            vocab_size=fast.vocab_size,
        )
    ).model
    assert sum(p.numel() for p in counted.parameters()) == landing.landed_params
    # Historical sizing requests interleaving; the high-level training default disables it.
    assert seen[0][0].lm_config.architecture.temporal.mode == "disabled"
    assert sum(p.numel() for p in seen[0][0].parameters()) < landing.landed_params
    assert not torch.equal(seen[0][1], before_sft["state_dict"]["lm_head.weight"])
    assert not torch.equal(
        saved["state_dict"]["lm_head.weight"], before_sft["state_dict"]["lm_head.weight"]
    )
    assert (
        saved["config"] == before_sft["config"]
        and saved["config"]["architecture"]["kind"] == "wave"
    )
    summary = json.loads((artifact.parent / "summary.json").read_text())
    assert summary["pretrain"]["tokens_total"] == summary["sft"]["tokens_total"] == 42


@pytest.mark.parametrize(
    "field,value", [("warmup_steps", 0), ("grad_accum_steps", 2), ("steps_per_epoch", 3)]
)
def test_yaml_legacy_inactive_training_fields_preserve_updates_and_canonical_neighbors_execute(
    field, value, tmp_path, monkeypatch
):
    from psannlm.configuration import run_yaml
    from psannlm.architectures import to_mapping

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.chdir(tmp_path)
    text = tmp_path / "texts.txt"
    text.write_text(TEXT, encoding="utf-8")
    raw = {
        "model": {"base": "respsann", "d_model": 24, "n_layers": 2, "n_heads": 3, "d_mlp": 36},
        "data": {"sources": [str(text)], "tokenizer": "simple", "max_length": 7},
        "train": {
            "epochs": 1,
            "batch_tokens": 14,
            "lr": 0.003,
            "amp": "fp32",
            "ddp": "off",
            "checkpoint_dir": str(tmp_path / "out"),
        },
    }
    results = []
    for extra in (False, True):
        cfg = deepcopy(raw)
        if extra:
            cfg["train"][field] = value
        path = tmp_path / "legacy.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        torch.manual_seed(137)
        with pytest.warns(DeprecationWarning) as warnings:
            model = run_yaml(path)
        assert len(warnings) == 1 and warnings[0].filename == __file__
        if extra:
            assert "train." + field in str(warnings[0].message)
        results.append(model)
    assert results[0]._trainer.cfg == results[1]._trainer.cfg
    for key, state in results[0]._model.state_dict().items():
        torch.testing.assert_close(state, results[1]._model.state_dict()[key], rtol=0, atol=0)
    canonical = deepcopy(raw)
    canonical["model"] = to_mapping(results[0].config)
    canonical["train"][field] = value
    path = tmp_path / "canonical.yaml"
    path.write_text(yaml.safe_dump(canonical), encoding="utf-8")
    torch.manual_seed(137)
    executed = run_yaml(path)
    assert getattr(executed._trainer.cfg, field) == value
    assert not torch.equal(executed._model.lm_head.weight, results[0]._model.lm_head.weight)
    final = torch.load(tmp_path / "out/final.pt", weights_only=True)
    assert final["state"]["step"] == executed._trainer.state.step
    assert all(
        item["step"].item() == final["state"]["step"] for item in final["optim"]["state"].values()
    )
