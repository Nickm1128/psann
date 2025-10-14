#!/usr/bin/env python
"""Compare HISSO benchmark JSON payloads and assert metrics stay within tolerance."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


@dataclass(frozen=True)
class ResultKey:
    dataset: str
    device: str
    variant: str


def _load_results(path: Path) -> Tuple[Mapping[str, object], Dict[ResultKey, Mapping[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = {}
    for entry in data.get("results", []):
        key = ResultKey(
            dataset=str(entry.get("dataset") or data.get("metadata", {}).get("dataset", "")),
            device=str(entry.get("device")),
            variant=str(entry.get("variant")),
        )
        results[key] = entry
    return data, results


def _format_key(key: ResultKey) -> str:
    return f"{key.dataset}:{key.device}:{key.variant}"


def _approx_equal(
    baseline: float,
    candidate: float,
    *,
    rtol: float,
    atol: float,
) -> bool:
    if math.isnan(baseline) and math.isnan(candidate):
        return True
    if math.isnan(baseline) or math.isnan(candidate):
        return False
    return math.isclose(candidate, baseline, rel_tol=rtol, abs_tol=atol)


def compare_benchmarks(
    baseline: Path,
    candidate: Path,
    *,
    reward_rtol: float,
    reward_atol: float,
    wall_rtol: float,
    wall_atol: float,
    allow_missing: bool,
) -> List[str]:
    baseline_meta, baseline_results = _load_results(baseline)
    candidate_meta, candidate_results = _load_results(candidate)

    messages: List[str] = []

    if baseline_meta.get("dataset") != candidate_meta.get("dataset"):
        messages.append(
            f"Dataset mismatch: baseline={baseline_meta.get('dataset')} candidate={candidate_meta.get('dataset')}"
        )

    missing = [
        key for key in baseline_results
        if key not in candidate_results
    ]
    if missing and not allow_missing:
        names = ", ".join(_format_key(k) for k in missing)
        messages.append(f"Candidate missing results for: {names}")

    for key, base_entry in baseline_results.items():
        candidate_entry = candidate_results.get(key)
        if candidate_entry is None:
            if allow_missing:
                continue
            # handled earlier
            continue

        # Structural checks
        for field in ("series_length", "primary_dim", "episode_length"):
            b_val = base_entry.get(field)
            c_val = candidate_entry.get(field)
            if b_val != c_val:
                messages.append(
                    f"{_format_key(key)} field '{field}' mismatch: baseline={b_val} candidate={c_val}"
                )

        if list(base_entry.get("feature_shape") or []) != list(candidate_entry.get("feature_shape") or []):
            messages.append(
                f"{_format_key(key)} feature_shape mismatch: baseline={base_entry.get('feature_shape')} "
                f"candidate={candidate_entry.get('feature_shape')}"
            )

        # Metric checks
        base_reward = float(base_entry.get("final_reward_mean", math.nan))
        cand_reward = float(candidate_entry.get("final_reward_mean", math.nan))
        if not _approx_equal(base_reward, cand_reward, rtol=reward_rtol, atol=reward_atol):
            messages.append(
                f"{_format_key(key)} final_reward_mean drifted: baseline={base_reward:.6g} "
                f"candidate={cand_reward:.6g}"
            )

        base_wall = float(base_entry.get("mean_wall_time_s", math.nan))
        cand_wall = float(candidate_entry.get("mean_wall_time_s", math.nan))
        if not _approx_equal(base_wall, cand_wall, rtol=wall_rtol, atol=wall_atol):
            messages.append(
                f"{_format_key(key)} mean_wall_time_s drifted beyond tolerance: baseline={base_wall:.3f}s "
                f"candidate={cand_wall:.3f}s"
            )

    return messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Reference benchmark JSON.")
    parser.add_argument("candidate", type=Path, help="New benchmark JSON to validate.")
    parser.add_argument(
        "--reward-rtol",
        type=float,
        default=0.25,
        help="Relative tolerance for final_reward_mean comparisons (default 0.25).",
    )
    parser.add_argument(
        "--reward-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for final_reward_mean comparisons (default 1e-3).",
    )
    parser.add_argument(
        "--wall-rtol",
        type=float,
        default=0.5,
        help="Relative tolerance for mean_wall_time_s comparisons (default 0.5).",
    )
    parser.add_argument(
        "--wall-atol",
        type=float,
        default=1.0,
        help="Absolute tolerance for mean_wall_time_s comparisons (default 1.0 seconds).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip variants that exist in baseline but not in candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = compare_benchmarks(
        args.baseline,
        args.candidate,
        reward_rtol=float(args.reward_rtol),
        reward_atol=float(args.reward_atol),
        wall_rtol=float(args.wall_rtol),
        wall_atol=float(args.wall_atol),
        allow_missing=bool(args.allow_missing),
    )
    if failures:
        for msg in failures:
            print(f"[FAIL] {msg}")
        raise SystemExit(1)
    print("Benchmarks within tolerance.")


if __name__ == "__main__":
    main()
