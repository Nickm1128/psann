"""Verify the current and legacy machine-readable public-API freezes."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if SOURCE_ROOT.is_dir() and os.environ.get("PSANN_API_CHECK_INSTALLED") != "1":
    sys.path.insert(0, str(SOURCE_ROOT))

import psann


def _parameter_names(value: Any) -> list[str]:
    return [name for name in inspect.signature(value).parameters if name not in {"self", "cls"}]


def _read_payload(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("API manifest must contain a JSON object.")
    return payload


def _module(name: str) -> Any:
    return importlib.import_module(name)


def check_manifest(path: str | Path) -> list[str]:
    """Return current API-freeze violations without mutating the checkout."""

    payload = _read_payload(path)
    failures: list[str] = []
    modules = payload.get("modules")
    if not isinstance(modules, Mapping):
        return ["manifest.modules must be an object"]

    for module_name, raw_contract in modules.items():
        if not isinstance(raw_contract, Mapping):
            failures.append(f"{module_name}: module contract must be an object")
            continue
        module = _module(str(module_name))
        actual_exports = list(getattr(module, "__all__", ()))
        stable = raw_contract.get("stable_exports")
        experimental = raw_contract.get("experimental_exports", [])
        signatures = raw_contract.get("signatures")
        exemptions = raw_contract.get("signature_exemptions", {})
        if not isinstance(stable, list) or not isinstance(experimental, list):
            failures.append(f"{module_name}: export inventories must be arrays")
            continue
        if not isinstance(signatures, Mapping) or not isinstance(exemptions, Mapping):
            failures.append(f"{module_name}: signatures and exemptions must be objects")
            continue

        inventory = [*stable, *experimental]
        duplicates = sorted({name for name in inventory if inventory.count(name) > 1})
        if duplicates:
            failures.append(f"{module_name}: duplicate inventory names {duplicates!r}")
        missing_from_manifest = sorted(set(actual_exports) - set(inventory))
        removed_from_exports = sorted(set(inventory) - set(actual_exports))
        if missing_from_manifest:
            failures.append(
                f"{module_name}: exports missing from manifest inventory "
                f"{missing_from_manifest!r}"
            )
        if removed_from_exports:
            failures.append(
                f"{module_name}: manifest names missing from module exports "
                f"{removed_from_exports!r}"
            )

        for name in stable:
            if name not in actual_exports:
                continue
            try:
                value = getattr(module, name)
            except (AttributeError, ImportError) as exc:
                failures.append(f"{module_name}.{name}: cannot import ({exc})")
                continue
            expected_parameters = signatures.get(name)
            exempt_reason = exemptions.get(name)
            if expected_parameters is not None and exempt_reason is not None:
                failures.append(f"{module_name}.{name}: signature is both frozen and exempt")
                continue
            try:
                observed = _parameter_names(value)
            except (TypeError, ValueError):
                if callable(value) and (
                    not isinstance(exempt_reason, str) or not exempt_reason.strip()
                ):
                    failures.append(f"{module_name}.{name}: callable signature is not inventoried")
                continue
            if expected_parameters is None:
                if callable(value):
                    failures.append(f"{module_name}.{name}: callable signature is not frozen")
                continue
            if observed != list(expected_parameters):
                failures.append(
                    f"{module_name}.{name}: parameters changed; "
                    f"expected={list(expected_parameters)!r}, observed={observed!r}"
                )

        unknown_signatures = sorted(set(signatures) - set(stable))
        unknown_exemptions = sorted(set(exemptions) - set(stable))
        if unknown_signatures:
            failures.append(
                f"{module_name}: signatures for non-stable names {unknown_signatures!r}"
            )
        if unknown_exemptions:
            failures.append(
                f"{module_name}: exemptions for non-stable names {unknown_exemptions!r}"
            )

    members = payload.get("required_members", {})
    if not isinstance(members, Mapping):
        failures.append("manifest.required_members must be an object")
        return failures
    for qualified_owner, expected_members in members.items():
        module_name, _, owner = str(qualified_owner).rpartition(".")
        value = getattr(_module(module_name), owner, None) if module_name else None
        for member in expected_members:
            if value is None or not callable(getattr(value, member, None)):
                failures.append(f"{qualified_owner}.{member}: missing callable member")
    return failures


def check_legacy_manifest(path: str | Path) -> list[str]:
    """Verify that the retained 0.12.7 public estimator surface remains compatible."""

    payload = _read_payload(path)
    failures: list[str] = []
    exports = payload.get("exports")
    estimators = payload.get("principal_estimators")
    if not isinstance(exports, list):
        return ["legacy manifest.exports must be an array"]
    if not isinstance(estimators, Mapping):
        return ["legacy manifest.principal_estimators must be an object"]
    missing = sorted(set(exports) - set(psann.__all__))
    if missing:
        failures.append(f"psann 0.12.7 exports removed from current API: {missing!r}")
    for name, raw_contract in estimators.items():
        if not isinstance(raw_contract, Mapping):
            failures.append(f"legacy {name}: estimator contract must be an object")
            continue
        value = getattr(psann, str(name), None)
        if value is None:
            failures.append(f"legacy {name}: estimator is missing")
            continue
        expected = list(raw_contract.get("parameters", ()))
        observed = _parameter_names(value)
        positions = [observed.index(parameter) for parameter in expected if parameter in observed]
        removed = [parameter for parameter in expected if parameter not in observed]
        if removed:
            failures.append(f"legacy {name}: parameters removed {removed!r}")
        elif positions != sorted(positions):
            failures.append(f"legacy {name}: parameter order changed")
        for member in raw_contract.get("required_members", ()):
            if not callable(getattr(value, member, None)):
                failures.append(f"legacy {name}.{member}: missing callable member")
    return failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=REPO_ROOT / "docs" / "workplace_public_api.json",
        type=Path,
    )
    parser.add_argument(
        "--legacy-manifest",
        default=REPO_ROOT / "docs" / "public_api_0_12_7.json",
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    failures = [
        *check_manifest(arguments.manifest),
        *check_legacy_manifest(arguments.legacy_manifest),
    ]
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}")
        return 1
    payload = _read_payload(arguments.manifest)
    stable_count = sum(len(contract["stable_exports"]) for contract in payload["modules"].values())
    legacy = _read_payload(arguments.legacy_manifest)
    print(
        f"public API freeze {payload['contract']} passed "
        f"({stable_count} scoped stable exports; {len(legacy['exports'])} legacy exports)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["check_legacy_manifest", "check_manifest", "main"]
