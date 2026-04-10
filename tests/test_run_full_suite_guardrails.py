from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_script_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{REPO_ROOT / 'src'}{os.pathsep}{existing}" if existing else f"{REPO_ROOT / 'src'}"
    )
    return env


def _skip_everything_args() -> list[str]:
    return [
        "--skip-light-probes",
        "--skip-ablations",
        "--skip-geo-bench",
        "--skip-geo-sweep",
        "--skip-geo-micro",
    ]


def test_run_full_suite_dry_run_smoke(tmp_path: Path):
    out_root = tmp_path / "suite"
    cmd = [
        sys.executable,
        "scripts/run_full_suite.py",
        "--dry-run",
        "--out-root",
        str(out_root),
        *_skip_everything_args(),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, env=_repo_script_env(), check=True)
    assert (out_root / "suite_manifest.json").exists()


def test_run_full_suite_rejects_git_commit_for_generated_output_dirs():
    out_root = REPO_ROOT / "reports" / "full_suite" / "pytest_guardrail"
    if out_root.exists():
        shutil.rmtree(out_root)
    cmd = [
        sys.executable,
        "scripts/run_full_suite.py",
        "--dry-run",
        "--git-commit",
        "--out-root",
        str(out_root),
        *_skip_everything_args(),
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_repo_script_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        assert result.returncode != 0
        combined = f"{result.stdout}\n{result.stderr}"
        assert "--git-commit cannot be used with generated-output directories" in combined
    finally:
        if out_root.exists():
            shutil.rmtree(out_root)
