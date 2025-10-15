"""Release helper for psann.

Run from the project root to bump the package version, rebuild wheels/sdist,
and upload the new artifacts to PyPI with a single command.

Examples
--------
Patch release with an explicit token:

    python scripts/release.py --part patch --token pypi-XXXXXXXXXXXXXXXX

Provide a fully qualified version instead of bumping:

    python scripts/release.py --version 0.11.3

If TWINE_USERNAME / TWINE_PASSWORD are already configured, omit --token.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ROOT / "pyproject.toml"
INIT_PATH = ROOT / "src" / "psann" / "__init__.py"

VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)
INIT_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def read_current_version() -> str:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not locate project.version in {PYPROJECT_PATH}")
    return match.group(1)


def bump_semver(version: str, part: str) -> str:
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version.strip())
    if not m:
        raise ValueError(f"Unsupported semantic version format: {version!r}")
    major, minor, patch = map(int, m.groups())
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump part: {part}")
    return f"{major}.{minor}.{patch}"


def write_pyproject_version(new_version: str) -> None:
    text = PYPROJECT_PATH.read_text(encoding="utf-8")
    if not VERSION_RE.search(text):
        raise RuntimeError(f"Could not update version in {PYPROJECT_PATH}")
    updated = VERSION_RE.sub(f'version = "{new_version}"', text, count=1)
    PYPROJECT_PATH.write_text(updated, encoding="utf-8")


def write_init_version(new_version: str) -> None:
    text = INIT_PATH.read_text(encoding="utf-8")
    if not INIT_VERSION_RE.search(text):
        raise RuntimeError(f"Could not update __version__ in {INIT_PATH}")
    updated = INIT_VERSION_RE.sub(f'__version__ = "{new_version}"', text, count=1)
    INIT_PATH.write_text(updated, encoding="utf-8")


def clean_artifacts(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
            continue
        shutil.rmtree(path, ignore_errors=True)


def discover_egg_info(root: Path) -> Iterable[Path]:
    yield from root.glob("*.egg-info")


def run_cmd(args: list[str], *, env: Optional[dict[str, str]] = None) -> None:
    print(f"+ {' '.join(args)}")
    subprocess.check_call(args, cwd=ROOT, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump psann version and publish to PyPI.")
    parser.add_argument(
        "--version",
        help="Explicit version to release (overrides --part).",
    )
    parser.add_argument(
        "--part",
        choices=("major", "minor", "patch"),
        default="patch",
        help="Semantic version component to bump when --version is omitted. Default: patch.",
    )
    parser.add_argument(
        "--token",
        help="PyPI API token for twine (will set TWINE_USERNAME='__token__').",
    )
    parser.add_argument(
        "--username",
        default="__token__",
        help="Override Twine username (default: __token__).",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build artifacts but skip the twine upload step.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip python -m build (only adjust version metadata).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the changes without touching files or running commands.",
    )
    args = parser.parse_args(argv)

    current = read_current_version()
    if args.version:
        new_version = args.version.strip()
    else:
        new_version = bump_semver(current, args.part)

    print(f"Current version: {current}")
    print(f"Releasing version: {new_version}")

    if args.dry_run:
        print("Dry run: no files modified, no commands executed.")
        return 0

    write_pyproject_version(new_version)
    write_init_version(new_version)
    print(f"Updated version metadata to {new_version}.")

    artifacts = [ROOT / "dist", ROOT / "build", *discover_egg_info(ROOT)]
    clean_artifacts(artifacts)
    print("Removed previous build artifacts.")

    if not args.skip_build:
        run_cmd([sys.executable, "-m", "build"])
    else:
        print("Skipping build step (--skip-build).")

    if args.skip_upload:
        print("Skipping upload step (--skip-upload).")
        return 0

    env = os.environ.copy()
    token = args.token or env.get("TWINE_PASSWORD")
    if token:
        env.setdefault("TWINE_USERNAME", args.username or "__token__")
        env.setdefault("TWINE_PASSWORD", token)
    else:
        print("No token provided; relying on existing Twine configuration or keyring.")

    run_cmd([sys.executable, "-m", "twine", "upload", "--non-interactive", "dist/*"], env=env)
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
