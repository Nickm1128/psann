from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)\n]+)\)")


def _repository_files(pattern: str) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            pattern,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / item for item in result.stdout.splitlines() if item]


def _local_link_target(markdown: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    path_text = unquote(target.split("#", maxsplit=1)[0])
    if not path_text:
        return None
    return (
        (ROOT / path_text.lstrip("/")) if path_text.startswith("/") else markdown.parent / path_text
    )


def test_all_local_markdown_links_resolve():
    missing: list[str] = []
    for markdown in _repository_files("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = _local_link_target(markdown, raw_target)
            if target is not None and not target.resolve().exists():
                missing.append(f"{markdown.relative_to(ROOT)} -> {raw_target}")

    assert missing == [], "Broken local Markdown links:\n" + "\n".join(missing)


@pytest.mark.parametrize("workflow", sorted((ROOT / ".github" / "workflows").glob("*.yml")))
def test_workflow_yaml_parses(workflow: Path):
    parsed = yaml.safe_load(workflow.read_text(encoding="utf-8"))

    assert isinstance(parsed, dict), f"{workflow.relative_to(ROOT)} must contain a YAML mapping"


def test_release_evidence_names_include_version_and_commit_identity():
    certification = (ROOT / ".github" / "workflows" / "release-certification.yml").read_text(
        encoding="utf-8"
    )
    security = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")

    assert 'default: "1.1.0rc1"' in certification
    for artifact_prefix in (
        "release-source-gates",
        "release-candidate",
        "release-candidate-cuda",
    ):
        assert (
            f"{artifact_prefix}-${{{{ inputs.release_version }}}}-${{{{ github.sha }}}}"
            in certification
        )
    assert "release_identity: ${{ inputs.release_version }}" in certification
    assert certification.count("CERT_WORKDIR=$(mktemp -d)") == 2
    assert certification.count('cd "$CERT_WORKDIR"') == 2
    assert '"$REPO_ROOT/reports/certification/cpu"' in certification
    assert '"$REPO_ROOT/reports/certification/cuda"' in certification
    assert "inputs.release_identity || 'development'" in security


def test_every_support_matrix_row_links_to_executable_or_policy_evidence():
    matrix = (ROOT / "docs" / "workplace_support_matrix.md").read_text(encoding="utf-8")
    tables = [block for block in re.split(r"\n\s*\n", matrix) if block.startswith("|")]

    assert tables
    for table in tables:
        rows = table.splitlines()
        assert "evidence" in rows[0].lower(), f"Missing evidence column: {rows[0]}"
        for row in rows[2:]:
            if row.startswith("|"):
                evidence = row.rstrip("|").split("|")[-1]
                assert "](" in evidence, f"Support row lacks linked evidence: {row}"


def test_ci_blocks_on_installed_python_windows_dependency_and_shap_claims():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "installed-wheel-platform:" in workflow
    assert "os: [ubuntu-latest, windows-latest]" in workflow
    assert 'python-version: ["3.11", "3.12", "3.13"]' in workflow
    assert workflow.count('python-version: ["3.11", "3.12", "3.13"]') == 3
    assert "constraints/workplace-floor.txt" in workflow
    assert "constraints/workplace-py311.txt" in workflow
    assert "constraints/explain-floor-py311.txt" in workflow
    assert "torch.__version__.split('+')[0] == '2.4.1'" in workflow
    assert "importlib.metadata.version('shap') == '0.50.0'" in workflow
