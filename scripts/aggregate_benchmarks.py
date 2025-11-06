"""Aggregate GPU validation outputs into benchmark artifacts.

Reads reports from scripts/run_gpu_validation.py and writes:
- throughput.csv: timestamp, base, batch_tokens, tokens_per_s
- memory.json: snapshot from GPU-04 (if present)

Usage:
  python scripts/aggregate_benchmarks.py --gpu-reports reports/gpu --out reports/benchmarks/<timestamp>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_summaries(base: Path) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    if not base.exists():
        return out
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        js = child / "summary.json"
        if not js.exists():
            continue
        try:
            payload = json.loads(js.read_text(encoding="utf-8"))
            stamp = payload.get("timestamp_utc") or child.name
            out.append((str(stamp), payload))
        except Exception:
            continue
    return out


def write_throughput_csv(outdir: Path, summaries: List[Tuple[str, Dict[str, Any]]]) -> None:
    rows: List[str] = ["timestamp,base,batch_tokens,tokens_per_s"]
    for stamp, payload in summaries:
        res = payload.get("results", {})
        g3 = res.get("GPU-03")
        if not isinstance(g3, dict):
            continue
        for base in ("respsann", "waveresnet"):
            entry = g3.get(base)
            if not isinstance(entry, dict):
                continue
            tokens = int(entry.get("tokens", 0))
            steps = int(entry.get("steps", 1)) or 1
            batch_tokens = tokens // steps
            tps = float(entry.get("tokens_per_s", 0.0))
            rows.append(f"{stamp},{base},{batch_tokens},{tps}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "throughput.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_memory_json(outdir: Path, summaries: List[Tuple[str, Dict[str, Any]]]) -> None:
    # Take the last available GPU-04 result as the snapshot
    snap: Dict[str, Any] | None = None
    for stamp, payload in summaries:
        res = payload.get("results", {})
        g4 = res.get("GPU-04")
        if isinstance(g4, dict) and g4.get("status") == "ok":
            snap = {
                "timestamp": stamp,
                "base": (g4.get("model") or {}).get("base"),
                "grad_checkpoint": bool(g4.get("grad_checkpoint")),
                "amp": "bf16",  # default in run_gpu_validation paths
                "max_memory_allocated_mb": g4.get("max_memory_allocated_mb"),
                "max_memory_reserved_mb": g4.get("max_memory_reserved_mb"),
                "elapsed_s": g4.get("elapsed_s"),
            }
    if snap is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "memory.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-reports", type=str, default="reports/gpu")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    base = Path(args.gpu_reports)
    outdir = Path(args.out)

    summaries = _load_summaries(base)
    if not summaries:
        raise SystemExit(f"No summary.json files found under {base}")

    write_throughput_csv(outdir, summaries)
    write_memory_json(outdir, summaries)


if __name__ == "__main__":
    main()

