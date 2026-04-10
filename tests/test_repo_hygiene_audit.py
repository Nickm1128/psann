import json
import subprocess
import sys
from pathlib import Path


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
