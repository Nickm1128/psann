"""Snapshot the reviewed current public API after an intentional contract change.

This is a maintainer tool, not a release gate. Its explicit acknowledgement prevents
routine formatting or testing from silently accepting an API change.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

MODULE_EXPERIMENTAL = {"psann": {"GeoSparseRegressor"}, "psann.platform": set()}
REQUIRED_MEMBERS = {
    "psann.PSANNClassifier": [
        "decision_function",
        "fit",
        "get_params",
        "predict",
        "predict_proba",
        "score",
        "set_params",
    ],
    "psann.PSANNRegressor": ["fit", "get_params", "predict", "score", "set_params"],
    "psann.ResConvPSANNRegressor": ["fit", "get_params", "predict", "score", "set_params"],
    "psann.ResPSANNRegressor": ["fit", "get_params", "predict", "score", "set_params"],
    "psann.SGRPSANNRegressor": ["fit", "get_params", "predict", "score", "set_params"],
    "psann.WaveResNetRegressor": ["fit", "get_params", "predict", "score", "set_params"],
    "psann.InferenceRuntime": [
        "create_session",
        "explain",
        "make_explainer",
        "metadata",
        "predict",
    ],
    "psann.InferenceSession": ["close", "predict_sequence", "reset", "step"],
    "psann.TrainingRun": ["evaluate", "export"],
}


def _parameters(value: Any) -> list[str]:
    return [name for name in inspect.signature(value).parameters if name not in {"self", "cls"}]


def _module_contract(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    experimental = MODULE_EXPERIMENTAL[module_name]
    stable = [name for name in module.__all__ if name not in experimental]
    signatures: dict[str, list[str]] = {}
    exemptions: dict[str, str] = {}
    for name in stable:
        value = getattr(module, name)
        try:
            signatures[name] = _parameters(value)
        except (TypeError, ValueError):
            if callable(value):
                exemptions[name] = "inspect.signature is unavailable for this public type"
    return {
        "stable_exports": stable,
        "experimental_exports": [name for name in module.__all__ if name in experimental],
        "signatures": signatures,
        "signature_exemptions": exemptions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acknowledge-api-change",
        choices=("workplace-v1",),
        required=True,
        help="Confirm that the diff will receive compatibility-policy review.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "docs" / "workplace_public_api.json",
    )
    arguments = parser.parse_args()
    payload = {
        "candidate": "1.1.0rc1",
        "contract": arguments.acknowledge_api_change,
        "modules": {name: _module_contract(name) for name in ("psann", "psann.platform")},
        "required_members": REQUIRED_MEMBERS,
        "stability": "stable unless listed as experimental",
        "version": 2,
    }
    arguments.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote reviewed API snapshot candidate to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
