"""Deprecated legacy façade for canonical episodic AMP helpers."""

from ..episodic.amp import _autocast_context, _guard_cuda_capture

__all__ = ["_autocast_context", "_guard_cuda_capture"]
