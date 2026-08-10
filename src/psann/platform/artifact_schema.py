"""Versioned JSON contract for native PSANN deployment artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ARTIFACT_FORMAT = "psann.model"
ARTIFACT_FORMAT_VERSION = "1.0"
MANIFEST_SCHEMA_VERSION = "1.0"
SUPPORTED_ARTIFACT_VERSIONS = ("0.9", "1.0")

MANIFEST_PATH = "manifest.json"
MODEL_CONFIG_PATH = "config/model.json"
FITTED_CONFIG_PATH = "config/fitted.json"
INPUT_SCHEMA_PATH = "schema/input.json"
OUTPUT_SCHEMA_PATH = "schema/output.json"
PREPROCESSING_PATH = "preprocessing/state.json"
WEIGHTS_PATH = "weights/model.pt"
CHECKSUMS_PATH = "checksums.sha256"
MODEL_CARD_PATH = "model-card.md"

REQUIRED_PAYLOAD_PATHS = frozenset(
    {
        MANIFEST_PATH,
        MODEL_CONFIG_PATH,
        FITTED_CONFIG_PATH,
        INPUT_SCHEMA_PATH,
        OUTPUT_SCHEMA_PATH,
        PREPROCESSING_PATH,
        WEIGHTS_PATH,
    }
)
ALLOWED_PAYLOAD_PATHS = REQUIRED_PAYLOAD_PATHS | {MODEL_CARD_PATH}

# This is intentionally dependency-free. It is published for registry and tooling
# integrations while runtime validation below provides the same required-field rules.
ARTIFACT_MANIFEST_JSON_SCHEMA: Mapping[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Nickm1128/psann/schemas/artifact-manifest-1.0.json",
    "title": "PSANN native model artifact manifest",
    "type": "object",
    "required": [
        "artifact_format",
        "artifact_format_version",
        "manifest_schema_version",
        "artifact_id",
        "created_at",
        "package",
        "runtime",
        "requirements",
        "model",
        "training",
        "capabilities",
        "experimental",
        "files",
        "required_extensions",
    ],
    "properties": {
        "artifact_format": {"const": ARTIFACT_FORMAT},
        "artifact_format_version": {
            "type": "string",
            "pattern": r"^[0-9]+\.[0-9]+$",
        },
        "manifest_schema_version": {"const": MANIFEST_SCHEMA_VERSION},
        "artifact_id": {"type": "string", "minLength": 1},
        "created_at": {"type": "string", "minLength": 1},
        "package": {
            "type": "object",
            "required": ["name", "version"],
            "properties": {
                "name": {"const": "psann"},
                "version": {"type": "string", "minLength": 1},
            },
        },
        "runtime": {
            "type": "object",
            "required": ["python", "numpy", "torch"],
            "properties": {
                "python": {"type": "string", "minLength": 1},
                "numpy": {"type": "string", "minLength": 1},
                "torch": {"type": "string", "minLength": 1},
            },
        },
        "requirements": {
            "type": "object",
            "required": ["python_min", "psann_min", "torch_min"],
        },
        "model": {
            "type": "object",
            "required": ["backbone", "task", "plugin"],
        },
        "training": {
            "type": "object",
            "required": ["run_id"],
        },
        "capabilities": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "uniqueItems": True,
        },
        "experimental": {"type": "boolean"},
        "files": {"type": "object"},
        "required_extensions": {"type": "array"},
        "metadata": {"type": "object"},
        "registry": {"type": "object"},
    },
    # Additive optional metadata is forward compatible within the artifact major.
    "additionalProperties": True,
}


class ArtifactError(RuntimeError):
    """Base error for native artifact validation, loading, and migration."""


class ArtifactFormatError(ArtifactError):
    """Raised when the bundle layout or JSON contract is invalid."""


class ArtifactChecksumError(ArtifactError):
    """Raised when a payload checksum is missing or does not match."""


class ArtifactVersionError(ArtifactError):
    """Raised when an artifact or runtime version is incompatible."""


class ArtifactExtensionError(ArtifactError):
    """Raised when a required registered plugin or transform is unavailable."""


class LegacyCheckpointTrustError(ArtifactError):
    """Raised when legacy pickle loading was not explicitly trusted."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactFormatError(f"manifest.{field} must be a JSON object.")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactFormatError(f"manifest.{field} must be a non-empty string.")
    return value


def _string_sequence(value: Any, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ArtifactFormatError(f"manifest.{field} must be an array of non-empty strings.")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ArtifactFormatError(f"manifest.{field} cannot contain duplicates.")
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the current manifest schema without importing a schema library."""

    required = ARTIFACT_MANIFEST_JSON_SCHEMA["required"]
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ArtifactFormatError(
            "Artifact manifest is missing required field(s): " + ", ".join(missing) + "."
        )
    if manifest.get("artifact_format") != ARTIFACT_FORMAT:
        raise ArtifactFormatError(
            f"manifest.artifact_format must be {ARTIFACT_FORMAT!r}; "
            f"received {manifest.get('artifact_format')!r}."
        )
    _string(manifest.get("artifact_format_version"), "artifact_format_version")
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactVersionError(
            "Unsupported manifest schema version "
            f"{manifest.get('manifest_schema_version')!r}; "
            f"supported={MANIFEST_SCHEMA_VERSION!r}."
        )
    _string(manifest.get("artifact_id"), "artifact_id")
    _string(manifest.get("created_at"), "created_at")

    package = _mapping(manifest.get("package"), "package")
    if package.get("name") != "psann":
        raise ArtifactFormatError("manifest.package.name must be 'psann'.")
    _string(package.get("version"), "package.version")

    runtime = _mapping(manifest.get("runtime"), "runtime")
    for name in ("python", "numpy", "torch"):
        _string(runtime.get(name), f"runtime.{name}")

    requirements = _mapping(manifest.get("requirements"), "requirements")
    for name in ("python_min", "psann_min", "torch_min"):
        _string(requirements.get(name), f"requirements.{name}")

    model = _mapping(manifest.get("model"), "model")
    _string(model.get("backbone"), "model.backbone")
    if model.get("task") not in {
        "regression",
        "binary",
        "multiclass",
        "multilabel",
    }:
        raise ArtifactFormatError(
            "manifest.model.task must be regression, binary, multiclass, or multilabel."
        )
    plugin = model.get("plugin")
    if plugin is not None:
        plugin_mapping = _mapping(plugin, "model.plugin")
        _string(plugin_mapping.get("identifier"), "model.plugin.identifier")
        if plugin_mapping.get("version") is not None:
            _string(plugin_mapping.get("version"), "model.plugin.version")

    training = _mapping(manifest.get("training"), "training")
    if training.get("run_id") is not None:
        _string(training.get("run_id"), "training.run_id")
    _string_sequence(manifest.get("capabilities"), "capabilities")
    if not isinstance(manifest.get("experimental"), bool):
        raise ArtifactFormatError("manifest.experimental must be a boolean.")

    files = _mapping(manifest.get("files"), "files")
    required_files = set(REQUIRED_PAYLOAD_PATHS - {MANIFEST_PATH})
    allowed_files = set(ALLOWED_PAYLOAD_PATHS - {MANIFEST_PATH})
    if not required_files <= set(files) or not set(files) <= allowed_files:
        expected = sorted(required_files)
        raise ArtifactFormatError(
            "manifest.files must declare every required payload and only supported "
            f"optional payloads; required={expected!r}, received={sorted(files)!r}."
        )
    for path, role in files.items():
        _string(path, f"files path {path!r}")
        _string(role, f"files.{path}")

    extensions = manifest.get("required_extensions")
    if not isinstance(extensions, list):
        raise ArtifactFormatError("manifest.required_extensions must be an array.")
    for index, extension in enumerate(extensions):
        value = _mapping(extension, f"required_extensions[{index}]")
        if value.get("kind") not in {
            "backbone_plugin",
            "categorical_encoder",
            "missing_value_imputer",
        }:
            raise ArtifactFormatError(
                f"manifest.required_extensions[{index}].kind is not supported."
            )
        _string(value.get("identifier"), f"required_extensions[{index}].identifier")

    for optional_mapping in ("metadata", "registry"):
        if optional_mapping in manifest:
            _mapping(manifest[optional_mapping], optional_mapping)


__all__ = [
    "ALLOWED_PAYLOAD_PATHS",
    "ARTIFACT_FORMAT",
    "ARTIFACT_FORMAT_VERSION",
    "ARTIFACT_MANIFEST_JSON_SCHEMA",
    "ArtifactChecksumError",
    "ArtifactError",
    "ArtifactExtensionError",
    "ArtifactFormatError",
    "ArtifactVersionError",
    "CHECKSUMS_PATH",
    "FITTED_CONFIG_PATH",
    "INPUT_SCHEMA_PATH",
    "LegacyCheckpointTrustError",
    "MANIFEST_PATH",
    "MANIFEST_SCHEMA_VERSION",
    "MODEL_CARD_PATH",
    "MODEL_CONFIG_PATH",
    "OUTPUT_SCHEMA_PATH",
    "PREPROCESSING_PATH",
    "REQUIRED_PAYLOAD_PATHS",
    "SUPPORTED_ARTIFACT_VERSIONS",
    "WEIGHTS_PATH",
    "validate_manifest",
]
