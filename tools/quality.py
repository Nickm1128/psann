#!/usr/bin/env python
"""Run the repository's canonical formatting and static-quality commands.

Examples:
  python tools/quality.py lint
  python tools/quality.py format
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
QUALITY_PATHS = (
    "src",
    "tests",
    "scripts",
    "examples",
    "psannlm",
    "tools",
    "_bench_psann_lm",
    "bench_psann_lm.py",
    "psann_adapter.py",
)


def _run(args: Sequence[str]) -> int:
    command = [sys.executable, *args]
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.call(command, cwd=REPO_ROOT)


def _run_external(args: Sequence[str]) -> int:
    command = list(args)
    print(f"+ {' '.join(command)}", flush=True)
    return subprocess.call(command, cwd=REPO_ROOT)


def run_lint() -> int:
    """Run every blocking static check and report all failures."""

    commands = (
        ("-m", "ruff", "check", *QUALITY_PATHS),
        ("-m", "black", "--check", *QUALITY_PATHS),
        ("-m", "mypy", "--config-file", "mypy.ini"),
    )
    failed = False
    for command in commands:
        failed = bool(_run(command)) or failed
    for command in (
        ("git", "diff", "--check"),
        ("git", "diff", "--cached", "--check"),
    ):
        failed = bool(_run_external(command)) or failed
    return int(failed)


def run_format() -> int:
    """Apply import fixes, then finish with the canonical Black formatter."""

    if _run(("-m", "ruff", "check", "--fix", *QUALITY_PATHS)):
        return 1
    return _run(("-m", "black", *QUALITY_PATHS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lint", "format"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "format":
        return run_format()
    return run_lint()


if __name__ == "__main__":
    raise SystemExit(main())
