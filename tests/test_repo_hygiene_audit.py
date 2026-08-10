import json
import subprocess
import sys
from pathlib import Path

from tools.repo_hygiene_audit import (
    _classify_top_level_file,
    _inspect_notebook,
)


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
    assert payload["top_level_violations"] == []
    assert payload["notebook_violations"] == []


def test_top_level_hygiene_uses_an_explicit_allowlist():
    assert _classify_top_level_file("README.md") is None
    assert _classify_top_level_file("docs/note.md") is None
    assert _classify_top_level_file("scratch.txt") is not None


def test_notebook_hygiene_reports_outputs_and_counts(tmp_path):
    notebook = tmp_path / "example.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 3,
                        "metadata": {},
                        "outputs": [{"output_type": "stream", "text": ["result\n"]}],
                        "source": ["print('result')"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    issue = _inspect_notebook(tmp_path, "example.ipynb")

    assert issue is not None
    assert "committed outputs" in issue.reason
    assert "execution counts" in issue.reason


def test_notebook_hygiene_accepts_clean_utf8_bom_notebook(tmp_path):
    notebook = tmp_path / "clean.ipynb"
    notebook.write_text(
        "\ufeff"
        + json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )

    assert _inspect_notebook(tmp_path, "clean.ipynb") is None
