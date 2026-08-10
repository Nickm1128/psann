from __future__ import annotations

import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("shap")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from psann.platform.certification import SCENARIOS, run_certification


def test_certification_registry_covers_the_six_workplace_scenarios():
    assert tuple(SCENARIOS) == (
        "tabular_regression",
        "binary_classification",
        "multiclass_classification",
        "convolutional",
        "sequence_context",
        "custom_registered_backbone",
    )


@pytest.mark.parametrize("scenario", tuple(SCENARIOS))
def test_workplace_scenario_passes_and_writes_privacy_safe_evidence(tmp_path, scenario):
    report = run_certification(
        tmp_path / scenario,
        scenarios=(scenario,),
        soak_iterations=1,
    )

    assert report["status"] == "passed"
    assert report["device"] == "cpu"
    assert report["privacy"] == {
        "contains_raw_inputs": False,
        "contains_targets": False,
        "contains_row_level_attributions": False,
    }
    assert report["scenarios"][0]["name"] == scenario
    assert report["scenarios"][0]["status"] == "passed"
    report_path = tmp_path / scenario / "workplace-certification-cpu.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "passed"
