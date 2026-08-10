"""Explicit workplace accelerator and numeric-precision policy."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal

import torch

AcceleratorStatus = Literal["stable", "experimental", "unsupported"]
AcceleratorOperation = Literal["training", "inference", "export", "explanation"]


@dataclass(frozen=True)
class AcceleratorCapability:
    """One resolved device/dtype capability decision."""

    device: str
    dtype: str
    operation: AcceleratorOperation
    amp: bool
    compile: bool
    status: AcceleratorStatus
    reason: str
    evidence_required: bool

    @property
    def supported(self) -> bool:
        return self.status != "unsupported"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "operation": self.operation,
            "amp": self.amp,
            "compile": self.compile,
            "status": self.status,
            "reason": self.reason,
            "evidence_required": self.evidence_required,
        }


def canonical_dtype(value: str | torch.dtype) -> str:
    """Return a stable dtype identifier or reject an unknown value."""

    aliases: dict[Any, str] = {
        "float32": "float32",
        "fp32": "float32",
        "float": "float32",
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    key: Any = value.strip().lower() if isinstance(value, str) else value
    try:
        return aliases[key]
    except (KeyError, TypeError) as exc:
        raise ValueError("dtype must be float32/fp32, float16/fp16, or bfloat16/bf16.") from exc


def accelerator_capability(
    device: str,
    dtype: str | torch.dtype,
    *,
    operation: AcceleratorOperation,
    amp: bool = False,
    compile: bool = False,
) -> AcceleratorCapability:
    """Resolve the documented workplace support tier without probing hardware."""

    device_name = str(device).split(":", 1)[0].strip().lower()
    dtype_name = canonical_dtype(dtype)
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cpu":
        supported = dtype_name == "float32" and not amp and not compile
        return AcceleratorCapability(
            device="cpu",
            dtype=dtype_name,
            operation=operation,
            amp=amp,
            compile=compile,
            status="stable" if supported else "unsupported",
            reason=(
                "CPU float32 without AMP/compile is the blocking workplace baseline."
                if supported
                else "Stable CPU execution is limited to float32 without AMP or compile."
            ),
            evidence_required=supported,
        )
    if device_name == "cuda":
        if operation in {"inference", "export", "explanation"}:
            supported = dtype_name == "float32" and not amp
        else:
            supported = (dtype_name == "float32" and not amp) or (
                dtype_name in {"float16", "bfloat16"} and amp
            )
        if amp and compile:
            supported = False
        if compile and operation != "training":
            supported = False
        return AcceleratorCapability(
            device="cuda",
            dtype=dtype_name,
            operation=operation,
            amp=amp,
            compile=compile,
            status="stable" if supported else "unsupported",
            reason=(
                "CUDA float32 and CUDA training AMP float16/bfloat16 are scheduled."
                if supported
                else "The requested CUDA dtype/AMP/compile combination is not certified."
            ),
            evidence_required=supported,
        )
    if device_name == "mps":
        supported = (
            operation in {"training", "inference"}
            and dtype_name == "float32"
            and not amp
            and not compile
        )
        return AcceleratorCapability(
            device="mps",
            dtype=dtype_name,
            operation=operation,
            amp=amp,
            compile=compile,
            status="experimental" if supported else "unsupported",
            reason=(
                "MPS float32 is observation-only and has no stable compatibility guarantee."
                if supported
                else "MPS export, explanation, AMP, reduced precision, and compile are not certified."
            ),
            evidence_required=False,
        )
    return AcceleratorCapability(
        device=device_name,
        dtype=dtype_name,
        operation=operation,
        amp=amp,
        compile=compile,
        status="unsupported",
        reason=f"Device {device_name!r} is outside the workplace support matrix.",
        evidence_required=False,
    )


def accelerator_support_matrix() -> tuple[AcceleratorCapability, ...]:
    """Return the stable and experimental matrix used by docs and automation."""

    return (
        accelerator_capability("cpu", "float32", operation="training"),
        accelerator_capability("cpu", "float32", operation="inference"),
        accelerator_capability("cuda", "float32", operation="training"),
        accelerator_capability("cuda", "float16", operation="training", amp=True),
        accelerator_capability("cuda", "bfloat16", operation="training", amp=True),
        accelerator_capability("cuda", "float32", operation="inference"),
        accelerator_capability("cuda", "float32", operation="export"),
        accelerator_capability("cuda", "float32", operation="explanation"),
        accelerator_capability("mps", "float32", operation="training"),
        accelerator_capability("mps", "float32", operation="inference"),
    )


def _device_available(device: torch.device) -> bool:
    if device.type == "cpu":
        return True
    if device.type == "cuda":
        return torch.cuda.is_available()
    if device.type == "mps":
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    return False


def resolve_workplace_device(
    requested: str | torch.device | None,
    *,
    fallback_policy: str,
    operation: AcceleratorOperation,
) -> tuple[torch.device, dict[str, Any] | None]:
    """Resolve an available workplace device with explicit fallback metadata."""

    if fallback_policy not in {"warn", "error"}:
        raise ValueError("fallback_policy must be 'warn' or 'error'.")
    if requested is None or requested == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            resolved = torch.device(requested)
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError(f"Invalid device request {requested!r}: {exc}") from exc
    capability = accelerator_capability(
        resolved.type,
        "float32",
        operation=operation,
    )
    if capability.status == "unsupported":
        raise ValueError(capability.reason)
    if _device_available(resolved):
        return resolved, None
    reason = f"requested {resolved.type!r} device is unavailable"
    if fallback_policy == "error":
        raise RuntimeError(f"{reason} and fallback_policy='error'.")
    warnings.warn(f"{reason}; using CPU.", RuntimeWarning, stacklevel=3)
    return (
        torch.device("cpu"),
        {
            "component": "device",
            "requested": str(resolved),
            "effective": "cpu",
            "reason": reason,
        },
    )


def runtime_accelerator_evidence() -> dict[str, Any]:
    """Return privacy-safe runtime evidence for scheduled accelerator reports."""

    mps = getattr(torch.backends, "mps", None)
    evidence: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count() if torch.cuda.is_available() else 0),
        "cuda_bf16_supported": bool(
            torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        ),
        "mps_available": bool(mps is not None and mps.is_available()),
        "matrix": [entry.to_dict() for entry in accelerator_support_matrix()],
    }
    if torch.cuda.is_available():
        evidence["cuda_devices"] = [
            {
                "name": torch.cuda.get_device_properties(index).name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    return evidence


__all__ = [
    "AcceleratorCapability",
    "accelerator_capability",
    "accelerator_support_matrix",
    "canonical_dtype",
    "resolve_workplace_device",
    "runtime_accelerator_evidence",
]
