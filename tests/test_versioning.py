import json
import re
from pathlib import Path

import psannlm

import psann

RELEASE_CANDIDATE = "1.1.0rc1"
RELEASE_TAG = "v1.1.0rc1"


def test_runtime_packages_report_their_selected_distribution_versions():
    assert psann.__version__ == RELEASE_CANDIDATE
    assert psannlm.__version__ == RELEASE_CANDIDATE


def test_coordinated_builds_use_synchronized_dynamic_version_sources():
    repo_root = Path(__file__).resolve().parents[1]
    core_config = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    lm_config = (repo_root / "psannlm" / "pyproject.toml").read_text(encoding="utf-8")
    canonical_version = (repo_root / "src" / "psann" / "_version.py").read_text(encoding="utf-8")
    derived_version = (repo_root / "psannlm" / "_version.py").read_text(encoding="utf-8")

    assert 'dynamic = ["version"]' in core_config
    assert 'path = "src/psann/_version.py"' in core_config
    assert 'dynamic = ["version"]' in lm_config
    assert 'path = "_version.py"' in lm_config
    version_pattern = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
    assert version_pattern.search(derived_version).group(1) == version_pattern.search(
        canonical_version
    ).group(1)


def test_documented_release_identity_matches_package_sources():
    repo_root = Path(__file__).resolve().parents[1]
    public_api = json.loads(
        (repo_root / "docs" / "workplace_public_api.json").read_text(encoding="utf-8")
    )
    identity = (repo_root / "docs" / "release_identity.md").read_text(encoding="utf-8")
    changelog = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    security = (repo_root / "SECURITY.md").read_text(encoding="utf-8")

    assert public_api["candidate"] == RELEASE_CANDIDATE
    assert f"## {RELEASE_CANDIDATE} - " in changelog
    assert RELEASE_CANDIDATE in identity
    assert RELEASE_TAG in identity
    assert RELEASE_CANDIDATE in security
