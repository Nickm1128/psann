from __future__ import annotations

from tools.run_coverage import REPORTS


def test_scoped_coverage_policy_keeps_release_critical_floors_blocking():
    thresholds = {name: threshold for name, _, threshold in REPORTS}

    assert thresholds == {
        "core": 70,
        "psannlm": 35,
        "scripts": 0,
        "release-helper": 60,
    }


def test_release_helper_has_a_dedicated_report_instead_of_aggregate_script_policy():
    reports = {name: include for name, include, _ in REPORTS}

    assert reports["scripts"] == "scripts/*"
    assert reports["release-helper"] == "scripts/release.py"
