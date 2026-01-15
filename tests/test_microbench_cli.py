import json
import os
import subprocess
import sys
from pathlib import Path


def test_microbench_cli_smoke(tmp_path: Path):
    out = tmp_path / "microbench.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{Path.cwd() / 'src'}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "scripts/microbench_psann.py",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--batch-size",
        "8",
        "--input-dim",
        "8",
        "--output-dim",
        "2",
        "--hidden-layers",
        "1",
        "--hidden-units",
        "8",
        "--steps",
        "2",
        "--warmup-steps",
        "1",
        "--baselines",
        "dense",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, env=env)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "results" in payload
    assert "psann_psann" in payload["results"]
