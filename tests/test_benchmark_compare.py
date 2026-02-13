from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_hisso_benchmarks import compare_benchmarks


def _write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_payload() -> dict:
    return {
        "metadata": {
            "dataset": "portfolio",
            "devices": ["cpu"],
            "epochs": 4,
            "repeats": 1,
            "window": 64,
        },
        "results": [
            {
                "dataset": "portfolio",
                "device": "cpu",
                "variant": "dense",
                "series_length": 128,
                "primary_dim": 2,
                "episode_length": 64,
                "feature_shape": [2],
                "final_reward_mean": -1.2e-6,
                "mean_wall_time_s": 5.0,
            },
        ],
    }


def test_compare_benchmarks_allows_small_drift(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    baseline = _base_payload()
    candidate = _base_payload()
    candidate["results"][0]["final_reward_mean"] = -1.5e-6  # within tolerances
    candidate["results"][0]["mean_wall_time_s"] = 5.8  # within tolerances

    _write_payload(baseline_path, baseline)
    _write_payload(candidate_path, candidate)

    failures = compare_benchmarks(
        baseline_path,
        candidate_path,
        modes=None,
        reward_rtol=0.5,
        reward_atol=1e-3,
        wall_rtol=0.5,
        wall_atol=1.5,
        allow_missing=False,
    )
    assert failures == []


def test_compare_benchmarks_flags_large_drift(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    baseline = _base_payload()
    candidate = _base_payload()
    candidate["results"][0]["final_reward_mean"] = -0.05  # large drift

    _write_payload(baseline_path, baseline)
    _write_payload(candidate_path, candidate)

    failures = compare_benchmarks(
        baseline_path,
        candidate_path,
        modes=None,
        reward_rtol=0.1,
        reward_atol=1e-3,
        wall_rtol=0.5,
        wall_atol=1.0,
        allow_missing=False,
    )

    assert failures, "Expected drift detection"
    assert any("final_reward_mean" in msg for msg in failures)


def test_compare_benchmarks_detects_missing_variant(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    baseline = _base_payload()
    candidate = _base_payload()
    candidate["results"] = []  # missing dense variant

    _write_payload(baseline_path, baseline)
    _write_payload(candidate_path, candidate)

    failures = compare_benchmarks(
        baseline_path,
        candidate_path,
        modes=None,
        reward_rtol=0.5,
        reward_atol=1e-3,
        wall_rtol=0.5,
        wall_atol=1.0,
        allow_missing=False,
    )

    assert failures
    assert any("missing results" in msg.lower() for msg in failures)


def test_compare_benchmarks_mode_filter_targets_selected_schedule(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    baseline = _base_payload()
    fast_entry = dict(baseline["results"][0])
    fast_entry["mode"] = "fast"
    baseline["results"][0]["mode"] = "compat"
    baseline["results"].append(fast_entry)

    candidate = _base_payload()
    compat_entry = dict(candidate["results"][0])
    compat_entry["mode"] = "compat"
    fast_candidate = dict(candidate["results"][0])
    fast_candidate["mode"] = "fast"
    fast_candidate["mean_wall_time_s"] = 15.0  # obvious regression only in fast mode
    candidate["results"] = [compat_entry, fast_candidate]

    _write_payload(baseline_path, baseline)
    _write_payload(candidate_path, candidate)

    compat_failures = compare_benchmarks(
        baseline_path,
        candidate_path,
        modes="compat",
        reward_rtol=0.2,
        reward_atol=1e-3,
        wall_rtol=0.2,
        wall_atol=1.0,
        allow_missing=False,
    )
    assert compat_failures == []

    fast_failures = compare_benchmarks(
        baseline_path,
        candidate_path,
        modes="fast",
        reward_rtol=0.2,
        reward_atol=1e-3,
        wall_rtol=0.2,
        wall_atol=1.0,
        allow_missing=False,
    )
    assert fast_failures
    assert any("mean_wall_time_s" in msg for msg in fast_failures)
