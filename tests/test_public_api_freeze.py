from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.check_public_api import check_legacy_manifest, check_manifest

ROOT = Path(__file__).resolve().parents[1]
CURRENT_MANIFEST = ROOT / "docs" / "workplace_public_api.json"
LEGACY_MANIFEST = ROOT / "docs" / "public_api_0_12_7.json"


def test_workplace_public_api_matches_published_freeze():
    assert check_manifest(CURRENT_MANIFEST) == []


def test_public_0_12_7_api_remains_compatible():
    assert check_legacy_manifest(LEGACY_MANIFEST) == []
    legacy = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "legacy" / "psann-0.12.7-regressor.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(legacy["exports"]) == 53
    assert len(legacy["principal_estimators"]) == 6
    assert legacy["provenance"]["wheel_sha256"] == fixture["producer"]["wheel"]["sha256"]


def test_current_manifest_classifies_every_export_in_both_public_modules():
    payload = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))

    assert set(payload["modules"]) == {"psann", "psann.platform"}
    assert payload["modules"]["psann"]["experimental_exports"] == ["GeoSparseRegressor"]
    assert len(payload["modules"]["psann"]["stable_exports"]) == 147
    assert len(payload["modules"]["psann.platform"]["stable_exports"]) == 115


def test_current_freeze_rejects_uninventoried_export_signature_and_method(tmp_path: Path):
    original = json.loads(CURRENT_MANIFEST.read_text(encoding="utf-8"))

    missing_export = copy.deepcopy(original)
    missing_export["modules"]["psann"]["stable_exports"].remove("PSANNRegressor")
    missing_export_path = tmp_path / "missing-export.json"
    missing_export_path.write_text(json.dumps(missing_export), encoding="utf-8")
    assert any(
        "exports missing from manifest inventory" in failure
        for failure in check_manifest(missing_export_path)
    )

    changed_signature = copy.deepcopy(original)
    changed_signature["modules"]["psann.platform"]["signatures"]["create_model"] = [
        "spec",
        "silently_breaking",
    ]
    changed_signature_path = tmp_path / "changed-signature.json"
    changed_signature_path.write_text(json.dumps(changed_signature), encoding="utf-8")
    assert any(
        "parameters changed" in failure for failure in check_manifest(changed_signature_path)
    )

    missing_method = copy.deepcopy(original)
    missing_method["required_members"]["psann.TrainingRun"].append("silently_removed")
    missing_method_path = tmp_path / "missing-method.json"
    missing_method_path.write_text(json.dumps(missing_method), encoding="utf-8")
    assert any(
        "missing callable member" in failure for failure in check_manifest(missing_method_path)
    )


def test_legacy_freeze_rejects_removed_estimator_parameter(tmp_path: Path):
    payload = json.loads(LEGACY_MANIFEST.read_text(encoding="utf-8"))
    payload["principal_estimators"]["PSANNRegressor"]["parameters"].append(
        "removed_legacy_parameter"
    )
    manifest = tmp_path / "legacy-api.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert any("parameters removed" in failure for failure in check_legacy_manifest(manifest))
