import json
import subprocess
import sys
from pathlib import Path

AUDIT_PATH = Path(__file__).resolve().parents[1] / "tools" / "repo_hygiene_audit.py"


def _run_audit(repo_root: Path) -> dict:
    subprocess.run(["git", "init", "-q"], cwd=repo_root, check=True)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True)
    result = subprocess.run(
        [sys.executable, str(AUDIT_PATH), "--repo-root", str(repo_root), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    return json.loads(result.stdout)


def test_repo_hygiene_audit_has_no_tracked_output_violations():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "tools/repo_hygiene_audit.py", "--json"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["prohibited_tracked"] == []
    assert payload["provenance_references"] == []


def test_repo_hygiene_audit_detects_every_private_path_family_and_inventory(tmp_path):
    paths = [
        ".psann-dev/plan.md",
        "docs/internal/notes.md",
        "docs/archive/legacy.md",
        "docs/backlog/todo.md",
        "benchmarks/lm_plan.md",
        "docs/HISSO_logging_spec.md",
        "docs/attention_api_plan.md",
        "docs/extras_removal_inventory.md",
        "docs/geosparse_vs_relu_colab_notebook_todo.md",
        "docs/hisso_optimization_todo.md",
        "docs/lsm_robustness_todo.md",
        "docs/phase1_audit.md",
        "docs/project_cleanup_todo.md",
        "docs/repo_hygiene_audit.md",
        "docs/repo_hygiene_followups.md",
        "docs/repo_hygiene_waves.md",
        "docs/future_todo.md",
        "docs/new_architecture_plan.md",
        "docs/team-follow-up.md",
        "docs/team_follow_up.md",
        "benchmarks/next_sweep_plan.md",
    ]
    for path in paths:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("private material\n", encoding="utf-8")

    payload = _run_audit(tmp_path)
    assert {item["path"] for item in payload["prohibited_tracked"]} == set(paths)


def test_repo_hygiene_audit_detects_provenance_and_excludes_detector_terms(tmp_path):
    public = tmp_path / "README.md"
    public.write_text(
        "Codex\nChatGPT\nClaude\nCopilot\nGPT-5.4\nai-generated\nagent-authored\n",
        encoding="utf-8",
    )
    detector = tmp_path / "tools" / "repo_hygiene_audit.py"
    detector.parent.mkdir()
    detector.write_text(
        "Codex ChatGPT Claude Copilot GPT-5 ai-generated agent-authored\n", encoding="utf-8"
    )
    tests = tmp_path / "tests" / "test_repo_hygiene_audit.py"
    tests.parent.mkdir()
    tests.write_text(
        "Codex ChatGPT Claude Copilot GPT-5 ai-generated agent-authored\n", encoding="utf-8"
    )
    notebook = tmp_path / "notebooks" / "provenance.ipynb"
    notebook.parent.mkdir()
    notebook.write_text('{"cells": [{"source": ["Codex"]}]}\n', encoding="utf-8")

    payload = _run_audit(tmp_path)
    references = payload["provenance_references"]
    assert len(references) == 8
    assert {item["path"] for item in references} == {"README.md", "notebooks/provenance.ipynb"}


def test_repo_hygiene_audit_allows_technical_ai_lm_and_gpt2_terms(tmp_path):
    (tmp_path / "README.md").write_text(
        "Artificial Intelligence classifier; language model tooling; gpt2 tokenizer.\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    result = subprocess.run(
        [sys.executable, str(AUDIT_PATH), "--repo-root", str(tmp_path), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["prohibited_tracked"] == []
    assert payload["provenance_references"] == []
