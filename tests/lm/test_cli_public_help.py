"""CLI module compatibility and canonical help are executable public contracts."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("module", ["psannlm.train", "psannlm.lm.train.cli", "psannlm.cli"])
def test_legacy_cli_module_warns_once_and_delegates_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONUTF8="1"),
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("DeprecationWarning") == 1
    assert "python -m psannlm" in result.stderr
    assert "--base" not in result.stdout
    assert "--sine-" not in result.stdout


@pytest.mark.parametrize("command", [[], ["train"], ["resume"], ["eval"], ["generate"]])
def test_canonical_cli_help_teaches_only_canonical_architecture_options(command):
    result = subprocess.run(
        [sys.executable, "-m", "psannlm", *command, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONUTF8="1"),
    )
    assert result.returncode == 0, result.stderr
    assert "DeprecationWarning" not in result.stderr
    assert "--base" not in result.stdout
    assert "--sine-" not in result.stdout
