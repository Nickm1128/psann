from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import psannlm
import pytest
from psannlm._compat import (
    PSANN_CORE_REQUIREMENT,
    PSANNCoreCompatibilityError,
    ensure_core_compatibility,
)

import psann

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("version", ["1.1.0rc1", "1.1.0", "1.1.9"])
def test_supported_core_versions_are_accepted(version: str):
    ensure_core_compatibility(version)


@pytest.mark.parametrize("version", ["1.0.9", "1.2.0rc1", "2.0.0", "not-a-version"])
def test_unsupported_core_versions_fail_with_actionable_error(version: str):
    with pytest.raises(PSANNCoreCompatibilityError, match=r"required >=1\.1\.0rc1,<1\.2"):
        ensure_core_compatibility(version)


def test_lm_version_is_its_own_distribution_version(monkeypatch: pytest.MonkeyPatch):
    lm_version = psannlm.__version__
    monkeypatch.setattr(psann, "__version__", "1.1.0")

    reloaded = importlib.reload(psannlm)

    assert reloaded.__version__ == lm_version
    assert reloaded.__version__ != psann.__version__


def test_lm_dependency_metadata_matches_runtime_compatibility_band():
    metadata = tomllib.loads((ROOT / "psannlm" / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]

    assert f"psann{PSANN_CORE_REQUIREMENT}" in dependencies


def test_import_rejects_a_mismatched_installed_core(tmp_path: Path):
    fake_core = tmp_path / "psann"
    fake_core.mkdir()
    (fake_core / "__init__.py").write_text('__version__ = "1.0.9"\n', encoding="utf-8")
    script = "import sys; " f"sys.path.insert(0, {str(ROOT)!r}); " "import psannlm"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "PSANN-LM is incompatible with the installed psann core" in result.stderr
    assert "found 1.0.9, required >=1.1.0rc1,<1.2" in result.stderr
