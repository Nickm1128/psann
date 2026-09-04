"""Inspect real independently built distribution artifacts."""

from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest


@pytest.mark.slow
@pytest.mark.parametrize("distribution", ["psann", "psannlm"])
def test_built_distribution_has_exact_package_boundary(distribution, tmp_path):
    root = Path(__file__).resolve().parents[1]
    project = root if distribution == "psann" else root / "psannlm"
    result = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--wheel", "--outdir", str(tmp_path)],
        cwd=project,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = tmp_path.glob("*.whl")
    (sdist,) = tmp_path.glob("*.tar.gz")
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        metadata = archive.read(
            next(n for n in wheel_names if n.endswith(".dist-info/METADATA"))
        ).decode()
    with tarfile.open(sdist) as archive:
        sdist_names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    source = root / ("src/psann" if distribution == "psann" else "psannlm")
    modules = {str(path.relative_to(source)).replace("\\", "/") for path in source.rglob("*.py")}
    assert modules
    assert {f"{distribution}/{name}" for name in modules} <= wheel_names
    prefix = "src/psann/" if distribution == "psann" else ""
    assert {prefix + name for name in modules} <= sdist_names
    assert all(name.startswith((distribution + "/", distribution + "-")) for name in wheel_names)
    assert not any(".psann-dev" in name for name in wheel_names | sdist_names)
    if distribution == "psann":
        assert not any(name.startswith("psannlm/") for name in wheel_names | sdist_names)
        assert "psann/architectures/components.py" in wheel_names
        assert "Requires-Dist: psannlm" not in metadata
    else:
        assert "Requires-Dist: psann>=" in metadata
        assert not any(
            name.startswith(("psann/", "src/psann/")) for name in wheel_names | sdist_names
        )
