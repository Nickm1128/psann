"""
Run the full pytest suite on a CUDA-enabled machine and save artifacts.

Outputs are written to: <out>/YYYYMMDD_HHMMSS/

Contents:
- system.json        : basic Python/Torch/CUDA/GPU info
- env.json           : detailed environment report (from gpu_env_report.py)
- stdout.log         : full pytest stdout/stderr
- junit.xml          : JUnit test report (CI-friendly)
- pytest_report.json : JSON report (if pytest-json-report plugin available)
- gpu_outputs/       : artifacts produced by GPU tests (via PSANN_OUTPUT_DIR)

Usage:
  python scripts/run_cuda_tests.py --out reports/tests
  # extra pytest args after --
  python scripts/run_cuda_tests.py --out reports/tests -- -k "lm or gpu" -vv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def system_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split(" ")[0],
        "platform": sys.platform,
    }
    try:
        import torch  # type: ignore

        info.update(
            {
                "torch": getattr(torch, "__version__", None),
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
                "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
            }
        )
        if torch.cuda.is_available():
            gpus: List[Dict[str, Any]] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 2),
                        "major": props.major,
                        "minor": props.minor,
                    }
                )
            info["gpus"] = gpus
    except Exception as e:  # pragma: no cover - optional
        info["torch_import_error"] = str(e)
    return info


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def has_pytest_jsonreport() -> bool:
    return importlib.util.find_spec("pytest_jsonreport") is not None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run all tests on CUDA and save artifacts")
    ap.add_argument("--out", type=str, default="reports/tests", help="Base directory for reports")
    # Everything after -- is forwarded to pytest
    args, extra = ap.parse_known_args()
    setattr(args, "pytest_extra", extra)
    return args


def main() -> int:
    args = parse_args()
    base = Path(args.out).resolve()
    tag = _now_tag()
    outdir = base / tag
    ensure_dir(outdir)

    # Set PSANN_OUTPUT_DIR for GPU tests to store artifacts under our report folder
    gpu_outputs = outdir / "gpu_outputs"
    ensure_dir(gpu_outputs)
    env = os.environ.copy()
    env["PSANN_OUTPUT_DIR"] = str(gpu_outputs)

    # Write system info upfront
    sysinfo = system_info()
    write_json(outdir / "system.json", sysinfo)

    # Detailed environment report (best-effort)
    try:
        subprocess.run(
            [sys.executable, "scripts/gpu_env_report.py", "--outdir", str(outdir)],
            check=False,
            env=env,
        )
    except Exception:
        pass

    # Build pytest command
    junit_path = outdir / "junit.xml"
    json_path = outdir / "pytest_report.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        "-rA",
        "--durations=25",
        f"--junit-xml={junit_path}",
    ]
    if has_pytest_jsonreport():
        cmd += ["--json-report", f"--json-report-file={json_path}"]
    # forward any extra user-provided args (after --)
    cmd += args.pytest_extra

    # Run tests and tee output to file
    log_fp = (outdir / "stdout.log").open("w", encoding="utf-8")
    print(f"[INFO] Writing test report to: {outdir}")
    print(f"[INFO] Running: {' '.join(cmd)}")
    rc = 0
    try:
        proc = subprocess.run(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
        rc = int(proc.returncode)
    finally:
        log_fp.close()

    # Minimal summary (parse JUnit if present)
    summary: Dict[str, Any] = {"timestamp_utc": tag, "exit_code": rc}
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(junit_path)
        root = tree.getroot()
        # Pytest may write either <testsuite> or <testsuites>
        suites = []
        if root.tag == "testsuite":
            suites = [root]
        elif root.tag == "testsuites":
            suites = list(root.findall("testsuite"))
        total = failures = errors = skipped = 0
        for s in suites:
            total += int(s.attrib.get("tests", 0))
            failures += int(s.attrib.get("failures", 0))
            errors += int(s.attrib.get("errors", 0))
            skipped += int(s.attrib.get("skipped", 0))
        summary.update(
            {
                "tests": total,
                "failures": failures,
                "errors": errors,
                "skipped": skipped,
            }
        )
    except Exception:
        pass

    write_json(outdir / "summary.json", summary)
    print(f"[DONE] Test run complete. Artifacts at: {outdir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
