"""Execute shell consumer syntax, model configuration, and numerical parity."""

from dataclasses import replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest
import torch

from psannlm import LMConfig
from psannlm.architectures import build_lm_model, normalize_lm_config
from psannlm.architectures.compat import legacy_lm_config
from psannlm.lm.models.sine import SineConfig

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "docs/consumer_manifest.json").read_text())
SHELLS = sorted(ROOT.glob("scripts/*.sh"))


def test_manifest_classifies_every_shell_helper():
    assert {row["path"] for row in MANIFEST.get("shell", [])} == {
        path.relative_to(ROOT).as_posix() for path in SHELLS
    }
    for row in MANIFEST["shell"]:
        assert row["classification"] in {"canonical", "compatibility/migration"}
        if row["classification"] == "compatibility/migration":
            assert "Compatibility sampling diagnostic" in (ROOT / row["path"]).read_text()


@pytest.mark.parametrize("path", SHELLS, ids=lambda p: p.name)
def test_shell_syntax_references_and_canonical_commands(path):
    bash = shutil.which("bash")
    if bash is None:
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
        assert (
            candidate.is_file()
        ), "Install Bash (Git for Windows includes it) for shell validation."
        bash = str(candidate)
    result = subprocess.run([bash, "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source = path.read_text()
    assert not re.search(r"-m psannlm\.(?:train|cli|lm\.train\.cli|sft)\b", source)
    assert not re.search(r"--(?:base|sine-[\w-]+)\s", source)
    assert not re.search(r"\.(?:\[lm(?:,viz)?\])|install -e ./psannlm", source)
    for referenced in re.findall(
        r"(?:scripts|examples|configs)/[\w/.-]+\.(?:py|sh|yaml|txt)\b", source
    ):
        assert (ROOT / referenced).is_file(), (path.name, referenced)
    for code in re.findall(r"<<'([A-Z_]+)'\n(.*?)\n\1", source, re.S):
        compile(code[1], path.name + ":embedded-python", "exec")


def assert_optimized_models_match(left, right):
    assert left == right
    models = []
    for config in (left, right):
        # Retain every architecture policy; bound only shared dimensions and depth.
        torch.manual_seed(57)
        model = build_lm_model(
            replace(config, d_model=16, d_mlp=24, n_layers=1, n_heads=2, vocab_size=29)
        ).model
        before = model.lm_head.weight.detach().clone()
        ids = torch.arange(12).reshape(2, 6)
        logits = model(ids)
        torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), ids.roll(-1, 1).flatten()
        ).backward()
        gradients = {
            name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
        }
        torch.optim.AdamW(model.parameters(), lr=0.002).step()
        assert not torch.equal(before, model.lm_head.weight)
        models.append((model, logits.detach(), gradients))
    torch.testing.assert_close(models[0][1], models[1][1], rtol=0, atol=0)
    assert models[0][2].keys() == models[1][2].keys()
    for name, gradient in models[0][2].items():
        torch.testing.assert_close(gradient, models[1][2][name], rtol=0, atol=0)
    for name, tensor in models[0][0].state_dict().items():
        torch.testing.assert_close(tensor, models[1][0].state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["runpod_smoke_train.sh", "runpod_train_1b.sh"])
def test_shell_wave_preset_preserves_trained_legacy_cli_model(name):
    source = (ROOT / "scripts" / name).read_text()
    kind = re.search(r"--architecture (\S+)", source)
    assert kind is not None
    flags = {
        flag.replace("-", "_"): int(value)
        for flag, value in re.findall(r"--(d-model|d-mlp|n-layers|n-heads) (\d+)", source)
    }
    assert_optimized_models_match(
        LMConfig(architecture=kind[1], **flags),
        legacy_lm_config("waveresnet", dict(**flags, sine=SineConfig()), warn=False),
    )


@pytest.mark.parametrize("values", [(1.0, 0.001, 2.25, 0.25), (0.7, 0.02, 1.8, 0.1)])
def test_shell_300m_configuration_preserves_initialization_and_trained_model(
    values, tmp_path, monkeypatch
):
    source = (ROOT / "scripts/runpod_train_300m.sh").read_text()
    match = re.search(r"<<'MODEL_CONFIG'\n(.*?)\nMODEL_CONFIG", source, re.S)
    assert match is not None
    for key, value in zip(
        ("SINE_AMP_INIT", "SINE_DAMP_INIT", "SINE_FREQ_INIT", "SINE_FREQ_INIT_STD"), values
    ):
        monkeypatch.setenv(key, str(value))
    monkeypatch.setenv("ATTN_FLAGS", "--attn-impl=sdpa")
    output = tmp_path / "model.json"
    monkeypatch.setattr(sys, "argv", ["shell-model-config", str(output)])
    exec(compile(match[1], "runpod_train_300m.sh:model", "exec"), {"__name__": "__main__"})
    config = normalize_lm_config(json.loads(output.read_text()))
    previous = legacy_lm_config(
        "waveresnet",
        dict(
            d_model=1024,
            n_layers=16,
            n_heads=16,
            d_mlp=4096,
            attn_impl="sdpa",
            sine=SineConfig(
                amp_init=values[0],
                damp_init=values[1],
                freq_init=values[2],
                freq_init_std=values[3],
            ),
        ),
        warn=False,
    )
    assert_optimized_models_match(config, previous)
