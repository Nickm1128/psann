"""In-memory migrations for supported native artifact manifest versions."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from typing import Any

from .artifact_schema import (
    ARTIFACT_FORMAT,
    ARTIFACT_FORMAT_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ArtifactFormatError,
    ArtifactVersionError,
)

ManifestMigration = Callable[[Mapping[str, Any]], dict[str, Any]]
_MIGRATIONS: dict[str, ManifestMigration] = {}


def register_artifact_migration(
    source_version: str,
    migration: ManifestMigration,
    *,
    replace: bool = False,
) -> None:
    """Register one forward migration used before current-schema validation."""

    version = str(source_version).strip()
    if not version:
        raise ValueError("source_version cannot be empty.")
    if version in _MIGRATIONS and not replace:
        raise ValueError(f"Artifact migration from {version!r} is already registered.")
    if not callable(migration):
        raise TypeError("migration must be callable.")
    _MIGRATIONS[version] = migration


def _migrate_0_9(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate the preview 0.9 manifest while retaining additive metadata."""

    migrated = copy.deepcopy(dict(manifest))
    migrated["artifact_format"] = migrated.pop("format", ARTIFACT_FORMAT)
    migrated["artifact_format_version"] = ARTIFACT_FORMAT_VERSION
    migrated["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    package_version = migrated.pop("package_version", None)
    if "package" not in migrated:
        migrated["package"] = {
            "name": "psann",
            "version": str(package_version or "0.12.0"),
        }
    if "model" not in migrated:
        migrated["model"] = {
            "backbone": migrated.pop("backbone", None),
            "task": migrated.pop("task", None),
            "plugin": migrated.pop("plugin", None),
        }
    migrated.setdefault("required_extensions", [])
    migrated.setdefault("metadata", {})
    migrated.setdefault("registry", {})
    return migrated


register_artifact_migration("0.9", _migrate_0_9)


def manifest_version(manifest: Mapping[str, Any]) -> str:
    """Return the declared version from current or historical field names."""

    value = manifest.get("artifact_format_version", manifest.get("format_version"))
    if not isinstance(value, str) or not value.strip():
        raise ArtifactFormatError("Artifact manifest must declare artifact_format_version.")
    return value.strip()


def migrate_manifest(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Migrate a supported manifest to the current schema in memory."""

    current = copy.deepcopy(dict(manifest))
    applied: list[str] = []
    seen: set[str] = set()
    while True:
        version = manifest_version(current)
        if version == ARTIFACT_FORMAT_VERSION:
            return current, tuple(applied)
        if version in seen:
            raise ArtifactVersionError(f"Artifact migration cycle detected at version {version!r}.")
        seen.add(version)
        migration = _MIGRATIONS.get(version)
        if migration is None:
            supported = sorted({ARTIFACT_FORMAT_VERSION, *_MIGRATIONS})
            raise ArtifactVersionError(
                f"Unsupported artifact format version {version!r}; "
                f"supported={supported!r}. Upgrade PSANN if this artifact was "
                "created by a newer release."
            )
        current = migration(current)
        applied.append(version)


__all__ = [
    "ManifestMigration",
    "manifest_version",
    "migrate_manifest",
    "register_artifact_migration",
]
