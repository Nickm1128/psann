from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        reward_rtol=0.5,
        reward_atol=1e-3,
        wall_rtol=0.5,
        wall_atol=1.0,
        allow_missing=False,
    )

    assert failures
    assert any("missing results" in msg.lower() for msg in failures)

