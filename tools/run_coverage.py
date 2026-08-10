#!/usr/bin/env python
"""Run the fast suite once and publish separate coverage reports by code area."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "coverage"
DATA_FILE = OUTPUT_DIR / ".coverage"
SOURCE_AREAS = ("src/psann", "psannlm", "scripts")
REPORTS = (
    ("core", "src/psann/*", 70),
    ("psannlm", "psannlm/*", 35),
    ("scripts", "scripts/*", 0),
    ("release-helper", "scripts/release.py", 60),
)


def _run(args: Sequence[str], *, env: dict[str, str]) -> None:
    command = [sys.executable, "-m", "coverage", *args]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.check_call(command, cwd=REPO_ROOT, env=env)


def main() -> int:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    OUTPUT_DIR.mkdir(parents=True)

    env = os.environ.copy()
    env["COVERAGE_FILE"] = str(DATA_FILE)
    _run(
        (
            "run",
            "--branch",
            f"--source={','.join(SOURCE_AREAS)}",
            "-m",
            "pytest",
            "-m",
            "not slow and not gpu",
            "-q",
        ),
        env=env,
    )

    for name, include, threshold in REPORTS:
        print(f"\nCoverage area: {name}", flush=True)
        _run(
            ("report", f"--include={include}", f"--fail-under={threshold}"),
            env=env,
        )
        _run(
            (
                "xml",
                f"--include={include}",
                "-o",
                str(OUTPUT_DIR / f"{name}.xml"),
                "--fail-under=0",
            ),
            env=env,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
