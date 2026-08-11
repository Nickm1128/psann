"""Build and explicitly publish coordinated PSANN distributions.

Examples
--------
Validate and build the selected candidate without uploading:

    python scripts/release.py --version 1.1.0rc1 --skip-upload

Publish only after reviewing every preflight and artifact check:

    python scripts/release.py --version 1.1.0rc1 --confirm-upload 1.1.0rc1
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = ROOT / "src" / "psann" / "_version.py"
LM_VERSION_PATH = ROOT / "psannlm" / "_version.py"
LM_PYPROJECT_PATH = ROOT / "psannlm" / "pyproject.toml"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
PYPI_PROJECTS = ("psann", "psannlm")

VERSION_RE = re.compile(
    r'^__version__[^\S\r\n]*=[^\S\r\n]*"([^"]+)"[^\S\r\n]*$',
    re.MULTILINE,
)


class ReleasePreflightError(RuntimeError):
    """Raised when release preparation is incomplete or ambiguous."""


def read_version(path: Path = VERSION_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise ReleasePreflightError(f"Could not locate __version__ in {path}")
    return match.group(1)


def read_current_version() -> str:
    return read_version(VERSION_PATH)


def bump_semver(version: str, part: str) -> str:
    """Bump the public release tuple of a valid PEP 440 version."""

    parsed = parse_release_version(version)
    release = (*parsed.release, 0, 0)
    major, minor, patch = release[:3]
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


def parse_release_version(version: str) -> Version:
    """Return a canonical publishable PEP 440 version."""

    try:
        parsed = Version(version.strip())
    except InvalidVersion as exc:
        raise ValueError(f"Release version must be valid PEP 440: {version!r}") from exc
    if parsed.local is not None:
        raise ValueError("Release versions cannot contain a local version segment.")
    return parsed


def write_version(new_version: str) -> None:
    new_version = str(parse_release_version(new_version))
    for path in (VERSION_PATH, LM_VERSION_PATH):
        text = path.read_text(encoding="utf-8")
        if not VERSION_RE.search(text):
            raise ReleasePreflightError(f"Could not update __version__ in {path}")
        updated = VERSION_RE.sub(f'__version__ = "{new_version}"', text, count=1)
        path.write_text(updated, encoding="utf-8")


def _capture(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def ensure_clean_tree() -> None:
    result = _capture(["git", "status", "--porcelain", "--untracked-files=all"])
    if result.returncode != 0:
        raise ReleasePreflightError(f"Could not inspect the Git worktree: {result.stderr.strip()}")
    if result.stdout.strip():
        sample = "\n".join(result.stdout.splitlines()[:10])
        raise ReleasePreflightError(
            "Release preparation requires a clean Git worktree. Commit or remove all "
            f"tracked and untracked changes first:\n{sample}"
        )


def ensure_versions_synchronized(expected: str | None = None) -> None:
    core_version = read_version(VERSION_PATH)
    lm_version = read_version(LM_VERSION_PATH)
    if core_version != lm_version:
        raise ReleasePreflightError(
            f"Package versions are not synchronized: psann={core_version}, psannlm={lm_version}."
        )
    if expected is not None and core_version != expected:
        raise ReleasePreflightError(
            f"Package version {core_version} does not match requested release {expected}."
        )


def ensure_changelog_entry(version: str) -> None:
    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    heading = re.compile(rf"^##[ \t]+{re.escape(version)}(?:[ \t]+-|[ \t]*$)", re.MULTILINE)
    if not heading.search(changelog):
        raise ReleasePreflightError(
            f"CHANGELOG.md has no release heading for {version}; add one before building."
        )


def _lm_core_requirement() -> Requirement:
    metadata = tomllib.loads(LM_PYPROJECT_PATH.read_text(encoding="utf-8"))
    for value in metadata["project"]["dependencies"]:
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise ReleasePreflightError(f"Invalid PSANN-LM dependency: {value!r}") from exc
        if canonicalize_name(requirement.name) == "psann":
            return requirement
    raise ReleasePreflightError("PSANN-LM has no explicit psann core dependency.")


def ensure_lm_core_compatibility(version: str) -> None:
    requirement = _lm_core_requirement()
    if not requirement.specifier.contains(version, prereleases=True):
        raise ReleasePreflightError(
            f"PSANN-LM dependency {requirement} does not accept release version {version}."
        )


def ensure_tag_available(version: str, *, check_remote: bool) -> None:
    tag = f"v{version}"
    local = _capture(["git", "tag", "--list", tag])
    if local.returncode != 0:
        raise ReleasePreflightError(f"Could not inspect local Git tags: {local.stderr.strip()}")
    if local.stdout.strip():
        raise ReleasePreflightError(f"Git tag {tag} already exists locally and cannot be reused.")

    if not check_remote:
        return
    remote = _capture(["git", "ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}"])
    if remote.returncode == 0:
        raise ReleasePreflightError(f"Git tag {tag} already exists on origin and cannot be reused.")
    if remote.returncode != 2:
        raise ReleasePreflightError(
            f"Could not verify remote tag availability for {tag}: {remote.stderr.strip()}"
        )


def ensure_pypi_version_available(project: str, version: str) -> None:
    request = Request(
        f"https://pypi.org/pypi/{project}/{version}/json",
        headers={"User-Agent": "psann-release-preflight/1"},
    )
    try:
        response = urlopen(request, timeout=15)
    except HTTPError as exc:
        if exc.code == 404:
            return
        raise ReleasePreflightError(
            f"PyPI availability check failed for {project} {version}: HTTP {exc.code}."
        ) from exc
    except (OSError, URLError) as exc:
        raise ReleasePreflightError(
            f"PyPI availability check failed for {project} {version}: {exc}."
        ) from exc
    else:
        response.close()
    raise ReleasePreflightError(
        f"PyPI already contains {project} {version}; versions are immutable."
    )


def run_preflights(version: str, *, check_remote: bool = True) -> None:
    ensure_clean_tree()
    ensure_versions_synchronized()
    ensure_changelog_entry(version)
    ensure_lm_core_compatibility(version)
    ensure_tag_available(version, check_remote=check_remote)
    if check_remote:
        for project in PYPI_PROJECTS:
            ensure_pypi_version_available(project, version)


def _checked_artifact_path(path: Path) -> Path:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise ReleasePreflightError(
            f"Refusing to remove artifact outside the repository: {resolved}"
        )
    return resolved


def clean_artifacts(paths: Iterable[Path]) -> None:
    for raw_path in paths:
        path = _checked_artifact_path(raw_path)
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
            continue
        shutil.rmtree(path)


def discover_egg_info(root: Path) -> Iterable[Path]:
    yield from root.glob("*.egg-info")


def run_cmd(
    args: list[str],
    *,
    env: Optional[dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> None:
    print(f"+ {' '.join(args)} (cwd={cwd or ROOT})", flush=True)
    subprocess.check_call(args, cwd=str(cwd or ROOT), env=env)


def validate_release_artifacts(version: str) -> list[Path]:
    expected_version = Version(version)
    artifacts: list[Path] = []
    for project, directory in (("psann", ROOT / "dist"), ("psannlm", ROOT / "psannlm" / "dist")):
        wheels = sorted(directory.glob("*.whl"))
        sdists = sorted(directory.glob("*.tar.gz"))
        if len(wheels) != 1 or len(sdists) != 1:
            raise ReleasePreflightError(
                f"Expected one wheel and one sdist for {project} in {directory}; "
                f"found {len(wheels)} wheel(s) and {len(sdists)} sdist(s)."
            )
        wheel_name, wheel_version, _, _ = parse_wheel_filename(wheels[0].name)
        sdist_name, sdist_version = parse_sdist_filename(sdists[0].name)
        expected_name = canonicalize_name(project)
        if (
            canonicalize_name(wheel_name) != expected_name
            or canonicalize_name(sdist_name) != expected_name
        ):
            raise ReleasePreflightError(f"Unexpected project identity in {directory}.")
        if wheel_version != expected_version or sdist_version != expected_version:
            raise ReleasePreflightError(
                f"Built {project} artifacts do not match requested version {version}."
            )
        artifacts.extend((wheels[0].resolve(), sdists[0].resolve()))
    return artifacts


def validate_built_packages(version: str) -> list[Path]:
    artifacts = validate_release_artifacts(version)
    run_cmd([sys.executable, "-m", "twine", "check", *map(str, artifacts)])
    run_cmd([sys.executable, "tools/package_smoke.py"])
    return artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and publish coordinated PSANN releases.")
    parser.add_argument("--version", help="Exact PEP 440 version to release; overrides --part.")
    parser.add_argument(
        "--part",
        choices=("major", "minor", "patch"),
        default="patch",
        help="Release component to bump when --version is omitted. Default: patch.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Run build, Twine, and package-smoke validation without uploading.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Reuse existing artifacts; they must exactly match the requested version.",
    )
    parser.add_argument(
        "--skip-remote-checks",
        action="store_true",
        help="Skip origin/PyPI availability checks for offline --skip-upload preparation only.",
    )
    parser.add_argument(
        "--confirm-upload",
        metavar="VERSION",
        help="Required exact version acknowledgement for a PyPI upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the selected version without modifying files or running preflights.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    current = read_current_version()
    new_version = (
        str(parse_release_version(args.version))
        if args.version
        else bump_semver(current, args.part)
    )

    print(f"Current version: {current}")
    print(f"Releasing version: {new_version}")
    if args.dry_run:
        print("Dry run: no files modified, no commands or preflights executed.")
        return 0
    if args.skip_remote_checks and not args.skip_upload:
        parser.error("--skip-remote-checks is permitted only with --skip-upload")
    if not args.skip_upload and args.confirm_upload != new_version:
        parser.error(f"upload requires --confirm-upload {new_version}")

    run_preflights(new_version, check_remote=not args.skip_remote_checks)
    print("Release preflights passed.")
    write_version(new_version)
    ensure_versions_synchronized(new_version)
    print(f"Synchronized package versions at {new_version}.")

    if not args.skip_build:
        generated = [
            ROOT / "dist",
            ROOT / "build",
            ROOT / "psannlm" / "dist",
            ROOT / "psannlm" / "build",
            *discover_egg_info(ROOT),
            *discover_egg_info(ROOT / "psannlm"),
        ]
        clean_artifacts(generated)
        print("Removed previous build artifacts.")
        run_cmd([sys.executable, "-m", "build"])
        run_cmd([sys.executable, "-m", "build"], cwd=ROOT / "psannlm")
    else:
        print("Reusing existing build artifacts (--skip-build).")

    artifacts = validate_built_packages(new_version)
    print("Twine and installed-wheel package smoke checks passed.")
    if args.skip_upload:
        print("Skipping upload step (--skip-upload).")
        return 0

    env = os.environ.copy()
    print(
        f"Upload confirmed for {new_version}; using trusted publishing or configured Twine credentials."
    )
    run_cmd(
        [sys.executable, "-m", "twine", "upload", "--non-interactive", *map(str, artifacts)],
        env=env,
    )
    print("Upload complete for psann and psannlm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
