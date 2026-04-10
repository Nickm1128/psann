from __future__ import annotations

import contextlib
from typing import Any, Iterable, Optional

import torch


def _autocast_context(
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> Any:
    """Return an autocast context compatible with current torch version/device."""

    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        try:
            return amp_mod.autocast(device.type, dtype=dtype)
        except TypeError:
            return amp_mod.autocast(dtype=dtype)  # type: ignore[call-arg]
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if hasattr(torch, "autocast"):
        try:
            return torch.autocast(device.type, dtype=dtype)  # type: ignore[attr-defined]
        except TypeError:  # pragma: no cover - defensive
            return torch.autocast("cuda", dtype=dtype)  # type: ignore[attr-defined]
    return contextlib.nullcontext()


@contextlib.contextmanager
def _guard_cuda_capture() -> Iterable[None]:
    """Temporarily neutralise CUDA graph capture checks when the driver is unavailable."""

    if not torch.cuda.is_available():
        yield
        return

    patched = False
    original_capture = None
    original_sync = None
    try:
        try:
            torch.cuda.is_current_stream_capturing()
        except RuntimeError:
            original_capture = torch.cuda.is_current_stream_capturing
            original_sync = torch.cuda.synchronize
            torch.cuda.is_current_stream_capturing = lambda: False
            torch.cuda.synchronize = lambda *args, **kwargs: None
            patched = True
        yield
    finally:
        if patched:
            if original_capture is not None:
                torch.cuda.is_current_stream_capturing = original_capture
            if original_sync is not None:
                torch.cuda.synchronize = original_sync
