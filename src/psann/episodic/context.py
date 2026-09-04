from __future__ import annotations

import inspect
import warnings
from typing import Any, Mapping, Optional

import numpy as np
import torch

from ..types import ContextExtractor

_NUMPY_CONTEXT_FALLBACK_WARNED_IDS: set[int] = set()


def _coerce_context_output(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert arbitrary context outputs into a detached tensor on ``device``."""

    target_dtype = dtype if dtype is not None else torch.float32

    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor.to(device)

    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float32)
        tensor = torch.from_numpy(array)
        return tensor.to(device=device, dtype=target_dtype)

    if isinstance(value, Mapping):
        for key in ("price_matrix", "prices", "returns", "context"):
            if key in value:
                try:
                    return _coerce_context_output(value[key], device=device, dtype=target_dtype)
                except TypeError:
                    pass
        for item in value.values():
            try:
                return _coerce_context_output(item, device=device, dtype=target_dtype)
            except TypeError:
                continue
        raise TypeError("context_extractor mapping did not contain tensor-compatible values.")

    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return _coerce_context_output(item, device=device, dtype=target_dtype)
            except TypeError:
                continue
        raise TypeError("context_extractor sequence did not contain tensor-compatible values.")

    raise TypeError(f"Unsupported context_extractor output type '{type(value).__name__}'.")


def _call_context_extractor(
    extractor: Optional[ContextExtractor],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Invoke ``extractor`` with best-effort dtype/device handling for HISSO training."""

    if extractor is None:
        return inputs.detach()

    fallback_to_numpy = False
    try:
        context = extractor(inputs)
    except TypeError as first_exc:
        if not isinstance(inputs, torch.Tensor):
            raise first_exc
        fallback_to_numpy = True
        inputs_np = inputs.detach().cpu().numpy()
        try:
            context = extractor(inputs_np)
        except Exception as second_exc:  # pragma: no cover - defensive fallback
            raise first_exc from second_exc

    if fallback_to_numpy:
        warning_key = id(extractor)
        if warning_key not in _NUMPY_CONTEXT_FALLBACK_WARNED_IDS:
            _NUMPY_CONTEXT_FALLBACK_WARNED_IDS.add(warning_key)
            stacklevel = 2
            stack = inspect.stack()
            try:
                for idx, frame_info in enumerate(stack[1:], start=1):
                    module_name = str(frame_info.frame.f_globals.get("__name__", ""))
                    file_name = str(frame_info.filename).replace("\\", "/").lower()
                    is_psann_module = (
                        module_name in {"psann", "src.psann"}
                        or module_name.startswith("psann.")
                        or module_name.startswith("src.psann.")
                        or "/src/psann/" in file_name
                    )
                    if not is_psann_module:
                        stacklevel = idx + 1
                        break
            finally:
                del stack
            warnings.warn(
                "HISSO context_extractor fell back to NumPy input after rejecting torch.Tensor; "
                "this can trigger host/device transfers (especially on CUDA). "
                "Update the extractor to accept torch.Tensor and return a tensor on the same "
                "device/dtype.",
                RuntimeWarning,
                stacklevel=stacklevel,
            )

    if isinstance(context, tuple):
        context = context[0]
    if context is None:
        raise TypeError("context_extractor returned None; expected tensor-like output.")

    if not isinstance(inputs, torch.Tensor):
        raise TypeError("HISSO context extraction requires tensor inputs.")

    device = inputs.device
    dtype = inputs.dtype if inputs.dtype is not None else torch.float32
    return _coerce_context_output(context, device=device, dtype=dtype)
