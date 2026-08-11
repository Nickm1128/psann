from __future__ import annotations

import hashlib
import io
import json
import os
import random
import tempfile
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch

from ._version import __version__

TRAINING_CHECKPOINT_FORMAT = "psann.training-checkpoint"
TRAINING_CHECKPOINT_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_STATE_NAME = "state.pt"
_CHECKSUM_NAME = "checksums.sha256"
_REQUIRED_MEMBERS = {_MANIFEST_NAME, _STATE_NAME, _CHECKSUM_NAME}
_MAX_METADATA_BYTES = 1024 * 1024
_MAX_STATE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = _MAX_STATE_BYTES + 2 * _MAX_METADATA_BYTES


class TrainingCheckpointError(RuntimeError):
    """Raised when a resumable training checkpoint is invalid or incompatible."""


def _validated_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise TrainingCheckpointError("Training checkpoint contains duplicate members.")
    observed = set(names)
    missing = sorted(_REQUIRED_MEMBERS - observed)
    if missing:
        raise TrainingCheckpointError(
            f"Training checkpoint is missing required member(s): {', '.join(missing)}."
        )
    unexpected = sorted(observed - _REQUIRED_MEMBERS)
    if unexpected:
        raise TrainingCheckpointError(
            f"Training checkpoint contains unexpected member(s): {', '.join(unexpected)}."
        )

    by_name = {info.filename: info for info in infos}
    total_size = 0
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            raise TrainingCheckpointError(
                f"Training checkpoint member {info.filename!r} is not a regular unencrypted file."
            )
        limit = _MAX_STATE_BYTES if info.filename == _STATE_NAME else _MAX_METADATA_BYTES
        if info.file_size < 0 or info.file_size > limit:
            raise TrainingCheckpointError(
                f"Training checkpoint member {info.filename!r} exceeds its safe size limit."
            )
        total_size += info.file_size
    if total_size > _MAX_TOTAL_BYTES:
        raise TrainingCheckpointError("Training checkpoint exceeds the safe total-size limit.")
    return by_name


def _read_manifest(archive: zipfile.ZipFile) -> Mapping[str, Any]:
    try:
        value = json.loads(archive.read(_MANIFEST_NAME).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingCheckpointError(f"Training checkpoint manifest is invalid: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TrainingCheckpointError("Training checkpoint manifest must be a JSON object.")
    return value


def _stream_sha256(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member, mode="r") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _safe_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_safe_value(item) for item in value)
    if isinstance(value, list):
        return [_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool, bytes, torch.Tensor)) or value is None:
        return value
    raise TypeError(
        "Training checkpoints may only contain tensors, NumPy arrays, and primitive "
        f"containers; received {type(value).__name__}."
    )


def _numpy_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _numpy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_numpy_value(item) for item in value)
    if isinstance(value, list):
        return [_numpy_value(item) for item in value]
    return value


def capture_rng_state(
    *,
    data_loader_generator: Optional[torch.Generator] = None,
) -> dict[str, Any]:
    numpy_state = cast(
        tuple[str, np.ndarray, int, int, float],
        np.random.get_state(),
    )
    cuda_state = (
        [state.detach().cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state().detach().cpu(),
        "cuda": cuda_state,
        "data_loader": (
            data_loader_generator.get_state().detach().cpu()
            if data_loader_generator is not None
            else None
        ),
    }


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    data_loader_generator: Optional[torch.Generator] = None,
) -> None:
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(tuple(python_state))

    numpy_state = state.get("numpy")
    if isinstance(numpy_state, Mapping):
        keys = numpy_state.get("keys")
        if isinstance(keys, torch.Tensor):
            keys_array = keys.detach().cpu().numpy().astype(np.uint32, copy=False)
        else:
            keys_array = np.asarray(keys, dtype=np.uint32)
        np.random.set_state(
            (
                str(numpy_state.get("algorithm", "MT19937")),
                keys_array,
                int(numpy_state.get("position", 0)),
                int(numpy_state.get("has_gauss", 0)),
                float(numpy_state.get("cached_gaussian", 0.0)),
            )
        )

    torch_state = state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.detach().cpu())

    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_state])

    loader_state = state.get("data_loader")
    if data_loader_generator is not None and isinstance(loader_state, torch.Tensor):
        data_loader_generator.set_state(loader_state.detach().cpu())


def _manifest_bytes(*, state_checksum: str) -> bytes:
    manifest = {
        "format": TRAINING_CHECKPOINT_FORMAT,
        "format_version": TRAINING_CHECKPOINT_VERSION,
        "package_version": __version__,
        "created_at_unix": time.time(),
        "state_file": _STATE_NAME,
        "state_sha256": state_checksum,
        "trusted_for_deployment": False,
    }
    return json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")


def _serialize_state(state: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(_safe_value(dict(state)), buffer)
    return buffer.getvalue()


def save_training_checkpoint(path: str | os.PathLike[str], state: Mapping[str, Any]) -> Path:
    """Atomically write a checksummed, restricted-load training checkpoint."""

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    state_bytes = _serialize_state(state)
    if len(state_bytes) > _MAX_STATE_BYTES:
        raise TrainingCheckpointError(
            f"Training checkpoint state exceeds the {_MAX_STATE_BYTES}-byte safe size limit."
        )
    state_checksum = hashlib.sha256(state_bytes).hexdigest()
    manifest_bytes = _manifest_bytes(state_checksum=state_checksum)
    checksum_bytes = f"{state_checksum}  {_STATE_NAME}\n".encode("utf-8")

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(_MANIFEST_NAME, manifest_bytes)
            archive.writestr(_STATE_NAME, state_bytes)
            archive.writestr(_CHECKSUM_NAME, checksum_bytes)
        with temp_path.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return target


def is_training_checkpoint(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path)
    if not candidate.is_file() or not zipfile.is_zipfile(candidate):
        return False
    try:
        with zipfile.ZipFile(candidate, mode="r") as archive:
            _validated_members(archive)
            manifest = _read_manifest(archive)
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        TrainingCheckpointError,
    ):
        return False
    return manifest.get("format") == TRAINING_CHECKPOINT_FORMAT


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Validate and restricted-load a `.psann-train` checkpoint."""

    source = Path(path).resolve()
    if not source.is_file():
        raise TrainingCheckpointError(f"Training checkpoint does not exist: {source}")
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            _validated_members(archive)
            manifest = _read_manifest(archive)
            if manifest.get("format") != TRAINING_CHECKPOINT_FORMAT:
                raise TrainingCheckpointError("File is not a PSANN resumable training checkpoint.")
            version = manifest.get("format_version")
            if version != TRAINING_CHECKPOINT_VERSION:
                raise TrainingCheckpointError(
                    "Unsupported training checkpoint format version "
                    f"{version!r}; expected {TRAINING_CHECKPOINT_VERSION}."
                )
            if manifest.get("state_file") != _STATE_NAME:
                raise TrainingCheckpointError(
                    f"Training checkpoint manifest must reference {_STATE_NAME!r}."
                )
            try:
                checksum_record = archive.read(_CHECKSUM_NAME).decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise TrainingCheckpointError(
                    "Training checkpoint checksum index must be UTF-8 text."
                ) from exc
            expected_checksum = str(manifest.get("state_sha256", ""))
            if len(expected_checksum) != 64 or any(
                character not in "0123456789abcdef" for character in expected_checksum.lower()
            ):
                raise TrainingCheckpointError(
                    "Training checkpoint manifest contains an invalid state SHA-256."
                )
            expected_record = f"{expected_checksum}  {_STATE_NAME}"
            if checksum_record != expected_record:
                raise TrainingCheckpointError(
                    "Training checkpoint checksum index does not match its manifest."
                )
            observed_checksum = _stream_sha256(archive, _STATE_NAME)
            if observed_checksum != expected_checksum:
                raise TrainingCheckpointError(
                    "Training checkpoint checksum mismatch; the state is corrupt or incomplete."
                )
            state_bytes = archive.read(_STATE_NAME)
    except TrainingCheckpointError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise TrainingCheckpointError(f"Invalid training checkpoint {source}: {exc}") from exc

    try:
        loaded = torch.load(
            io.BytesIO(state_bytes),
            map_location=map_location,
            weights_only=True,
        )
    except Exception as exc:
        raise TrainingCheckpointError(
            f"Training checkpoint state could not be restricted-loaded: {exc}"
        ) from exc
    if not isinstance(loaded, dict):
        raise TrainingCheckpointError("Training checkpoint state must be a dictionary.")
    return loaded


def restore_scaler_state(value: Any) -> Any:
    """Convert checkpoint tensors back into NumPy scaler state."""

    return _numpy_value(value)


class TrainingCheckpointManager:
    """Maintain atomic latest, best, and bounded periodic checkpoints."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        periodic_every: int = 0,
        keep_periodic: int = 3,
    ) -> None:
        if periodic_every < 0:
            raise ValueError("checkpoint_every must be >= 0.")
        if keep_periodic < 1:
            raise ValueError("checkpoint_keep must be >= 1.")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.periodic_every = int(periodic_every)
        self.keep_periodic = int(keep_periodic)

    def save(
        self,
        state: Mapping[str, Any],
        *,
        epoch: int,
        improved: bool,
    ) -> list[tuple[str, Path]]:
        written: list[tuple[str, Path]] = []
        latest = save_training_checkpoint(self.directory / "latest.psann-train", state)
        written.append(("latest", latest))
        if improved:
            best = save_training_checkpoint(self.directory / "best.psann-train", state)
            written.append(("best", best))
        if self.periodic_every and epoch % self.periodic_every == 0:
            periodic = save_training_checkpoint(
                self.directory / f"epoch_{epoch:06d}.psann-train",
                state,
            )
            written.append(("periodic", periodic))
            self._prune_periodic()
        return written

    def _prune_periodic(self) -> None:
        checkpoints = sorted(self.directory.glob("epoch_*.psann-train"))
        for obsolete in checkpoints[: -self.keep_periodic]:
            obsolete.unlink()


__all__ = [
    "TRAINING_CHECKPOINT_FORMAT",
    "TRAINING_CHECKPOINT_VERSION",
    "TrainingCheckpointError",
    "TrainingCheckpointManager",
    "capture_rng_state",
    "is_training_checkpoint",
    "load_training_checkpoint",
    "restore_rng_state",
    "restore_scaler_state",
    "save_training_checkpoint",
]
