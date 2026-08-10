"""Privacy-safe fingerprints, retention contracts, and optional operation hooks."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import torch

HookErrorPolicy = Literal["raise", "warn"]
OperationalSink = Callable[["OperationalEvent"], None]

_SECRET_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|credential|password|private[_-]?key|secret|token)($|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[oprsu]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
    re.compile(
        r"\b(?:api[_ -]?key|authorization|credential|password|private[_ -]?key|"
        r"secret|token)\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
)


class OperationalMetadataError(ValueError):
    """Raised when operational metadata contains prohibited sensitive material."""


def _secret_value(value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in _SECRET_VALUE_PATTERNS)


def sensitive_paths(value: Any, *, path: str = "metadata") -> tuple[str, ...]:
    """Return paths that look like secret-bearing keys or credential values."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            child = f"{path}.{name}"
            if _SECRET_KEY.search(name):
                found.append(child)
            else:
                found.extend(sensitive_paths(item, path=child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            found.extend(sensitive_paths(item, path=f"{path}[{index}]"))
    elif isinstance(value, str) and _secret_value(value):
        found.append(path)
    return tuple(found)


def validate_no_secrets(value: Any, *, field: str = "metadata") -> None:
    """Reject metadata/model-card content that resembles credentials."""

    paths = sensitive_paths(value, path=field)
    if paths:
        raise OperationalMetadataError(
            "Sensitive credential-like material is prohibited in operational metadata: "
            + ", ".join(paths)
            + "."
        )


def redact_sensitive(value: Any) -> Any:
    """Return a recursively redacted copy suitable for operational events."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            result[name] = "[REDACTED]" if _SECRET_KEY.search(name) else redact_sensitive(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str) and _secret_value(value):
        return "[REDACTED]"
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _update_array(hasher: Any, value: Any, *, label: str) -> None:
    array = np.asarray(value)
    hasher.update(label.encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    if array.dtype.kind in {"O", "S", "U"}:
        payload = json.dumps(
            array.tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        for start in range(0, len(payload), 1024 * 1024):
            hasher.update(payload[start : start + 1024 * 1024])
        return
    contiguous = np.ascontiguousarray(array)
    view = memoryview(contiguous).cast("B")
    for start in range(0, len(view), 1024 * 1024):
        hasher.update(view[start : start + 1024 * 1024])


def fingerprint_data(inputs: Any, targets: Any, context: Any | None = None) -> str:
    """Hash data content and shape without retaining or returning raw values."""

    hasher = hashlib.sha256()
    hasher.update(b"psann-data-fingerprint-v1")
    _update_array(hasher, inputs, label="inputs")
    _update_array(hasher, targets, label="targets")
    if context is not None:
        _update_array(hasher, context, label="context")
    return f"sha256:{hasher.hexdigest()}"


def fingerprint_model(model: Any) -> str:
    """Hash a fitted model state deterministically without serializing Python objects."""

    core = getattr(model, "estimator_", model)
    module = getattr(core, "model_", None)
    if not isinstance(module, torch.nn.Module):
        raise TypeError("Model fingerprinting requires a fitted Torch-backed estimator.")
    hasher = hashlib.sha256()
    hasher.update(b"psann-model-fingerprint-v1")
    for name, tensor in sorted(module.state_dict().items()):
        hasher.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        _update_array(hasher, array, label="tensor")
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class RetentionPolicy:
    """Serializable maximum-retention contract for workplace outputs."""

    history_days: int = 90
    checkpoint_days: int = 30
    explanation_days: int = 30
    service_log_days: int = 14
    redact_raw_inputs: bool = True
    redact_targets: bool = True
    redact_context: bool = True

    def __post_init__(self) -> None:
        for name in ("history_days", "checkpoint_days", "explanation_days", "service_log_days"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "history_days": self.history_days,
            "checkpoint_days": self.checkpoint_days,
            "explanation_days": self.explanation_days,
            "service_log_days": self.service_log_days,
            "redact_raw_inputs": self.redact_raw_inputs,
            "redact_targets": self.redact_targets,
            "redact_context": self.redact_context,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RetentionPolicy":
        return cls(**dict(value))


@dataclass(frozen=True)
class OperationalEvent:
    """Dependency-free event passed to explicitly configured workplace sinks."""

    kind: str
    timestamp: str
    run_id: str | None = None
    model_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        run_id: str | None = None,
        model_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "OperationalEvent":
        if not str(kind).strip():
            raise ValueError("Operational event kind cannot be empty.")
        return cls(
            kind=str(kind),
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            model_id=model_id,
            metadata=redact_sensitive(dict(metadata or {})),
        )


@dataclass(frozen=True)
class OperationalHooks:
    """Optional hooks for experiment tracking, registries, and monitoring."""

    experiment_tracker: OperationalSink | None = None
    registry_publisher: OperationalSink | None = None
    monitor: OperationalSink | None = None
    error_policy: HookErrorPolicy = "raise"

    def __post_init__(self) -> None:
        if self.error_policy not in {"raise", "warn"}:
            raise ValueError("Operational hook error_policy must be 'raise' or 'warn'.")
        for name in ("experiment_tracker", "registry_publisher", "monitor"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None.")

    def emit(self, event: OperationalEvent, *, registry: bool = False) -> None:
        sinks = [self.experiment_tracker, self.monitor]
        if registry:
            sinks.append(self.registry_publisher)
        for sink in sinks:
            if sink is None:
                continue
            try:
                sink(event)
            except Exception as exc:
                if self.error_policy == "raise":
                    raise
                warnings.warn(
                    f"Operational hook failed for event {event.kind!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )


__all__ = [
    "OperationalEvent",
    "OperationalHooks",
    "OperationalMetadataError",
    "RetentionPolicy",
    "fingerprint_data",
    "fingerprint_model",
    "redact_sensitive",
    "sensitive_paths",
    "validate_no_secrets",
]
