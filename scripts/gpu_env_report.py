"""Write a compact GPU/CUDA environment report for bug reports and benchmarks.

Writes a JSON payload plus a short text summary under an output directory.

Usage:
  python scripts/gpu_env_report.py
  python scripts/gpu_env_report.py --outdir outputs/gpu_tests/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(cmd):
    try:
        out = subprocess.run(cmd, check=False, capture_output=True, text=True, env=os.environ)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def gather_env_info():
    info = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executables": {},
        "libraries": {},
        "gpu": {},
    }

    # Executable versions
    for exe, args in (
        (
            "nvidia-smi",
            [
                "--query-gpu=name,driver_version,temperature.gpu,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
        ),
        ("nvcc", ["--version"]),
    ):
        code, out, err = _run([exe, *args])
        info["executables"][exe] = {
            "returncode": code,
            "stdout": out,
            "stderr": err,
        }

    # PyTorch
    try:
        import torch

        cuda_devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem_total = None
                mem_free = None
                try:
                    mem_free, mem_total = torch.cuda.mem_get_info(i)  # bytes
                except Exception:
                    pass
                cuda_devices.append(
                    {
                        "index": i,
                        "name": name,
                        "capability": cap,
                        "memory_total_bytes": mem_total,
                        "memory_free_bytes": mem_free,
                    }
                )

        info["libraries"]["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": getattr(torch.backends.cudnn, "version", lambda: None)(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        info["gpu"]["torch_cuda_devices"] = cuda_devices
    except Exception as e:  # pragma: no cover - optional
        info["libraries"]["torch"] = {"import_error": str(e)}

    # Transformers (optional)
    try:
        import transformers  # type: ignore

        info["libraries"]["transformers"] = {"version": getattr(transformers, "__version__", None)}
    except Exception as e:  # pragma: no cover - optional
        info["libraries"]["transformers"] = {"import_error": str(e)}

    # BitsAndBytes (optional)
    try:
        import bitsandbytes as bnb  # type: ignore

        info["libraries"]["bitsandbytes"] = {"version": getattr(bnb, "__version__", None)}
    except Exception as e:  # pragma: no cover - optional
        info["libraries"]["bitsandbytes"] = {"import_error": str(e)}

    return info


def main():
    parser = argparse.ArgumentParser(description="Report GPU/ML environment details")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to store the report (default: outputs/gpu_tests/<timestamp>)",
    )
    args = parser.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) if args.outdir else Path("outputs") / "gpu_tests" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    info = gather_env_info()

    env_path = outdir / "env.json"
    with env_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)

    # Also write a small text summary for quick glance
    summary_lines = [
        f"Timestamp: {info['timestamp']}",
        f"Host: {info['hostname']}",
        f"Platform: {info['platform']}",
        f"Python: {info['python']}",
    ]
    torch_info = info["libraries"].get("torch", {})
    if torch_info:
        summary_lines.extend(
            [
                f"torch: {torch_info.get('version')}",
                f"CUDA available: {torch_info.get('cuda_available')}",
                f"CUDA version: {torch_info.get('cuda_version')}",
                f"cuDNN version: {torch_info.get('cudnn_version')}",
                f"CUDA device count: {torch_info.get('device_count')}",
            ]
        )

    nvidia_smi = info["executables"].get("nvidia-smi", {})
    if nvidia_smi and nvidia_smi.get("stdout"):
        summary_lines.append("\nRaw nvidia-smi query (csv):")
        summary_lines.append(nvidia_smi["stdout"])  # already csv,noheader

    (outdir / "SUMMARY.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Environment report written to: {env_path}")


if __name__ == "__main__":
    main()
