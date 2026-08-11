"""Atomic ZIP I/O and checksum validation for native PSANN artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .artifact_migrations import manifest_version, migrate_manifest
from .artifact_schema import (
    ALLOWED_PAYLOAD_PATHS,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    REQUIRED_PAYLOAD_PATHS,
    ArtifactChecksumError,
    ArtifactFormatError,
    validate_manifest,
)

_MAX_METADATA_BYTES = 16 * 1024 * 1024
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
_CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")


@dataclass(frozen=True)
class ValidatedArtifact:
    """Validated manifest and lightweight metadata read without tensor loading."""

    path: Path
    manifest: Mapping[str, Any]
    original_version: str
    migrations: tuple[str, ...]


def json_bytes(value: Any) -> bytes:
    """Encode deterministic JSON while rejecting NaN and non-JSON values."""

    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactFormatError(f"Artifact metadata must be finite and JSON-safe: {exc}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactFormatError(f"JSON object contains duplicate key {key!r}.")
        result[key] = value
    return result


def parse_json(payload: bytes, *, member: str) -> Mapping[str, Any]:
    if len(payload) > _MAX_METADATA_BYTES:
        raise ArtifactFormatError(
            f"Artifact metadata member {member!r} exceeds {_MAX_METADATA_BYTES} bytes."
        )
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except ArtifactFormatError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactFormatError(f"Artifact member {member!r} is invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ArtifactFormatError(f"Artifact member {member!r} must contain a JSON object.")
    return value


def _safe_member_name(name: str) -> None:
    if not name or "\\" in name or ":" in name:
        raise ArtifactFormatError(f"Unsafe artifact member path {name!r}.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArtifactFormatError(f"Unsafe artifact member path {name!r}.")
    if path.as_posix() != name:
        raise ArtifactFormatError(f"Artifact member path is not canonical: {name!r}.")


def _validated_names(archive: zipfile.ZipFile) -> set[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ArtifactFormatError("Artifact bundle contains duplicate member names.")
    for name in names:
        _safe_member_name(name)
    name_set = set(names)
    required = set(REQUIRED_PAYLOAD_PATHS) | {CHECKSUMS_PATH}
    missing = sorted(required - name_set)
    if missing:
        raise ArtifactFormatError(
            "Artifact bundle is incomplete; missing member(s): " + ", ".join(missing) + "."
        )
    allowed = set(ALLOWED_PAYLOAD_PATHS) | {CHECKSUMS_PATH}
    unexpected = sorted(name_set - allowed)
    if unexpected:
        raise ArtifactFormatError(
            "Artifact bundle contains unsupported member(s): " + ", ".join(unexpected) + "."
        )

    total_size = 0
    for info in archive.infolist():
        if info.is_dir():
            raise ArtifactFormatError(f"Artifact member {info.filename!r} cannot be a directory.")
        if info.file_size < 0 or info.file_size > _MAX_MEMBER_BYTES:
            raise ArtifactFormatError(
                f"Artifact member {info.filename!r} exceeds the safe size limit."
            )
        if info.filename != "weights/model.pt" and info.file_size > _MAX_METADATA_BYTES:
            raise ArtifactFormatError(
                f"Artifact metadata member {info.filename!r} exceeds the safe size limit."
            )
        total_size += info.file_size
    if total_size > _MAX_TOTAL_BYTES:
        raise ArtifactFormatError("Artifact uncompressed payload exceeds the safe size limit.")
    return name_set


def _checksum_records(payload: bytes) -> Mapping[str, str]:
    if len(payload) > _MAX_METADATA_BYTES:
        raise ArtifactChecksumError("Artifact checksum index exceeds the safe size limit.")
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ArtifactChecksumError("Artifact checksum index is not UTF-8.") from exc
    records: dict[str, str] = {}
    for line in lines:
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ArtifactChecksumError("Artifact checksum index contains an invalid record.")
        digest, name = match.groups()
        _safe_member_name(name)
        if name in records:
            raise ArtifactChecksumError(f"Artifact checksum index repeats member {name!r}.")
        records[name] = digest
    return records


def _stream_sha256(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name, mode="r") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checksums(archive: zipfile.ZipFile, names: set[str]) -> None:
    records = _checksum_records(archive.read(CHECKSUMS_PATH))
    expected_names = names - {CHECKSUMS_PATH}
    if set(records) != expected_names:
        missing = sorted(expected_names - set(records))
        unexpected = sorted(set(records) - expected_names)
        raise ArtifactChecksumError(
            "Artifact checksum index does not cover every payload member: "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )
    for name in sorted(expected_names):
        observed = _stream_sha256(archive, name)
        if observed != records[name]:
            raise ArtifactChecksumError(
                f"Artifact checksum mismatch for {name!r}; the bundle is corrupt or incomplete."
            )


def inspect_bundle(path: str | os.PathLike[str]) -> ValidatedArtifact:
    """Validate structure/checksums and parse the manifest without loading tensors."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ArtifactFormatError(f"Model artifact does not exist: {source}")
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            names = _validated_names(archive)
            _validate_checksums(archive, names)
            raw_manifest = parse_json(archive.read(MANIFEST_PATH), member=MANIFEST_PATH)
    except (ArtifactChecksumError, ArtifactFormatError):
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ArtifactFormatError(f"Invalid PSANN model artifact {source}: {exc}") from exc

    original_version = manifest_version(raw_manifest)
    migrated, migrations = migrate_manifest(raw_manifest)
    validate_manifest(migrated)
    return ValidatedArtifact(
        path=source,
        manifest=migrated,
        original_version=original_version,
        migrations=migrations,
    )


def read_member(
    artifact: ValidatedArtifact,
    member: str,
    *,
    parse_as_json: bool = False,
) -> bytes | Mapping[str, Any]:
    """Read one member after revalidating the complete bundle."""

    if member not in ALLOWED_PAYLOAD_PATHS:
        raise ArtifactFormatError(f"Unsupported artifact member request {member!r}.")
    validated = inspect_bundle(artifact.path)
    if validated.manifest.get("artifact_id") != artifact.manifest.get("artifact_id"):
        raise ArtifactFormatError("Artifact changed between validation and member read.")
    try:
        with zipfile.ZipFile(artifact.path, mode="r") as archive:
            payload = archive.read(member)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ArtifactFormatError(f"Artifact member {member!r} could not be read: {exc}") from exc
    return parse_json(payload, member=member) if parse_as_json else payload


def _checksum_index(payloads: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in sorted(payloads)
    ).encode("utf-8")


def write_bundle(
    path: str | os.PathLike[str],
    payloads: Mapping[str, bytes],
) -> Path:
    """Atomically write an exact, checksummed artifact payload set."""

    names = set(payloads)
    missing = sorted(set(REQUIRED_PAYLOAD_PATHS) - names)
    if missing:
        raise ArtifactFormatError(
            "Cannot write incomplete artifact; missing member(s): " + ", ".join(missing) + "."
        )
    unexpected = sorted(names - set(ALLOWED_PAYLOAD_PATHS))
    if unexpected:
        raise ArtifactFormatError(
            "Cannot write unsupported artifact member(s): " + ", ".join(unexpected) + "."
        )
    for name in names:
        _safe_member_name(name)

    target = Path(path).resolve()
    if target.suffix.lower() != ".psann":
        raise ValueError(
            f"Native deployment artifacts must use the '.psann' extension; received {target.name!r}."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    all_payloads = dict(payloads)
    all_payloads[CHECKSUMS_PATH] = _checksum_index(payloads)

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(file_descriptor)
    temporary = Path(temp_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            for name in sorted(all_payloads):
                archive.writestr(name, all_payloads[name])
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def all_payloads(artifact: ValidatedArtifact) -> dict[str, bytes]:
    """Read every non-checksum payload from a previously validated bundle."""

    validated = inspect_bundle(artifact.path)
    if validated.manifest.get("artifact_id") != artifact.manifest.get("artifact_id"):
        raise ArtifactFormatError("Artifact changed between validation and migration.")
    try:
        with zipfile.ZipFile(artifact.path, mode="r") as archive:
            return {
                name: archive.read(name) for name in archive.namelist() if name != CHECKSUMS_PATH
            }
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactFormatError(f"Artifact payloads could not be read: {exc}") from exc


__all__ = [
    "ValidatedArtifact",
    "all_payloads",
    "inspect_bundle",
    "json_bytes",
    "parse_json",
    "read_member",
    "write_bundle",
]
