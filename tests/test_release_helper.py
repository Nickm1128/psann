from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.error import HTTPError

import pytest
from scripts import release


@pytest.mark.parametrize(
    ("current", "part", "expected"),
    [
        ("1.0.0rc1", "patch", "1.0.1"),
        ("1.0.0rc1", "minor", "1.1.0"),
        ("1.0.0rc1", "major", "2.0.0"),
        ("1.2.3.post1", "patch", "1.2.4"),
        ("1.2.3.dev4", "minor", "1.3.0"),
    ],
)
def test_bump_semver_accepts_pep440_versions(
    current: str,
    part: str,
    expected: str,
):
    assert release.bump_semver(current, part) == expected


@pytest.mark.parametrize("invalid", ["1.0.0garbage", "release-next", "1.0.0+private"])
def test_release_version_validation_fails_closed(invalid: str):
    with pytest.raises(ValueError):
        release.parse_release_version(invalid)


def test_release_dry_run_does_not_modify_version_sources():
    paths = (release.VERSION_PATH, release.LM_VERSION_PATH)
    before = {path: path.read_bytes() for path in paths}

    assert release.main(["--version", "1.0.0rc1", "--dry-run"]) == 0

    assert {path: path.read_bytes() for path in paths} == before


def test_release_helper_has_no_command_line_credential_option():
    source = Path(release.__file__).read_text(encoding="utf-8")

    assert "--token" not in source
    assert "pypi-" not in source.lower()


def test_clean_tree_preflight_rejects_tracked_and_untracked_changes(
    monkeypatch: pytest.MonkeyPatch,
):
    dirty = subprocess.CompletedProcess(
        args=["git"],
        returncode=0,
        stdout=" M src/psann/_version.py\n?? unexpected.txt\n",
        stderr="",
    )
    monkeypatch.setattr(release, "_capture", lambda *args, **kwargs: dirty)

    with pytest.raises(release.ReleasePreflightError, match="clean Git worktree"):
        release.ensure_clean_tree()


def test_version_preflight_rejects_unsynchronized_distribution_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    core = tmp_path / "core.py"
    lm = tmp_path / "lm.py"
    core.write_text('__version__ = "1.1.0rc1"\n', encoding="utf-8")
    lm.write_text('__version__ = "1.1.0"\n', encoding="utf-8")
    monkeypatch.setattr(release, "VERSION_PATH", core)
    monkeypatch.setattr(release, "LM_VERSION_PATH", lm)

    with pytest.raises(release.ReleasePreflightError, match="not synchronized"):
        release.ensure_versions_synchronized()


def test_changelog_preflight_requires_exact_release_heading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("# Changelog\n\n## 1.1.0rc1 - 2026-08-10\n", encoding="utf-8")
    monkeypatch.setattr(release, "CHANGELOG_PATH", changelog)

    release.ensure_changelog_entry("1.1.0rc1")
    with pytest.raises(release.ReleasePreflightError, match="no release heading"):
        release.ensure_changelog_entry("1.1.0")


def test_tag_preflight_rejects_existing_local_tag(monkeypatch: pytest.MonkeyPatch):
    existing = subprocess.CompletedProcess(
        args=["git"], returncode=0, stdout="v1.1.0rc1\n", stderr=""
    )
    monkeypatch.setattr(release, "_capture", lambda *args, **kwargs: existing)

    with pytest.raises(release.ReleasePreflightError, match="already exists locally"):
        release.ensure_tag_available("1.1.0rc1", check_remote=True)


def test_pypi_preflight_accepts_404_and_rejects_existing_version(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing(*args, **kwargs):
        raise HTTPError("https://pypi.org", 404, "not found", {}, None)

    monkeypatch.setattr(release, "urlopen", missing)
    release.ensure_pypi_version_available("psann", "1.1.0rc1")

    class ExistingResponse:
        def close(self):
            return None

    monkeypatch.setattr(release, "urlopen", lambda *args, **kwargs: ExistingResponse())
    with pytest.raises(release.ReleasePreflightError, match="already contains"):
        release.ensure_pypi_version_available("psann", "1.1.0rc1")


def test_artifact_preflight_requires_exact_project_and_version_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    core_dist = tmp_path / "dist"
    lm_dist = tmp_path / "psannlm" / "dist"
    core_dist.mkdir(parents=True)
    lm_dist.mkdir(parents=True)
    for path in (
        core_dist / "psann-1.1.0rc1-py3-none-any.whl",
        core_dist / "psann-1.1.0rc1.tar.gz",
        lm_dist / "psannlm-1.1.0rc1-py3-none-any.whl",
        lm_dist / "psannlm-1.1.0rc1.tar.gz",
    ):
        path.touch()
    monkeypatch.setattr(release, "ROOT", tmp_path)

    artifacts = release.validate_release_artifacts("1.1.0rc1")
    assert len(artifacts) == 4

    (lm_dist / "psannlm-1.1.0rc1.tar.gz").rename(lm_dist / "psannlm-1.1.0.tar.gz")
    with pytest.raises(release.ReleasePreflightError, match="do not match"):
        release.validate_release_artifacts("1.1.0rc1")


def test_release_upload_requires_exact_version_confirmation():
    with pytest.raises(SystemExit):
        release.main(["--version", "1.1.0rc1"])

    with pytest.raises(SystemExit):
        release.main(
            [
                "--version",
                "1.1.0rc1",
                "--confirm-upload",
                "1.1.0",
            ]
        )


def test_validated_skip_upload_flow_runs_every_local_gate(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        release,
        "run_preflights",
        lambda version, check_remote: calls.append(("preflight", (version, check_remote))),
    )
    monkeypatch.setattr(release, "write_version", lambda version: calls.append(("write", version)))
    monkeypatch.setattr(
        release,
        "ensure_versions_synchronized",
        lambda version: calls.append(("synchronized", version)),
    )
    monkeypatch.setattr(release, "clean_artifacts", lambda paths: calls.append(("clean", True)))
    monkeypatch.setattr(
        release,
        "run_cmd",
        lambda args, **kwargs: calls.append(("command", tuple(args))),
    )
    monkeypatch.setattr(
        release,
        "validate_built_packages",
        lambda version: calls.append(("validate", version)) or [Path("core.whl")],
    )

    assert release.main(["--version", "1.1.0rc1", "--skip-upload"]) == 0

    assert ("preflight", ("1.1.0rc1", True)) in calls
    assert ("write", "1.1.0rc1") in calls
    assert ("synchronized", "1.1.0rc1") in calls
    assert ("validate", "1.1.0rc1") in calls
    assert calls.index(("preflight", ("1.1.0rc1", True))) < calls.index(("write", "1.1.0rc1"))
    build_commands = [value for kind, value in calls if kind == "command"]
    assert build_commands.count((release.sys.executable, "-m", "build")) == 2


def test_remote_checks_cannot_be_skipped_for_upload():
    with pytest.raises(SystemExit):
        release.main(
            [
                "--version",
                "1.1.0rc1",
                "--skip-remote-checks",
                "--confirm-upload",
                "1.1.0rc1",
            ]
        )
