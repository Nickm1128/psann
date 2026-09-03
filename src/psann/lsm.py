from __future__ import annotations

"""Deprecated compatibility facade for low-level LSM classes."""

import warnings
from typing import Any

from ._lsm.common import TensorLike
from ._lsm.conv import LSMConv2d as _LSMConv2d
from ._lsm.conv import LSMConv2dExpander as _LSMConv2dExpander
from ._lsm.conv import MaskedConv2d as _MaskedConv2d
from ._lsm.dense import LSM as _LSM
from ._lsm.dense import LSMExpander as _LSMExpander
from ._lsm.dense import MaskedLinear as _MaskedLinear

__all__ = [
    "TensorLike",
    "MaskedLinear",
    "LSM",
    "LSMExpander",
    "MaskedConv2d",
    "LSMConv2d",
    "LSMConv2dExpander",
]

_SUPPRESS_DEPRECATION_WARNING = False
_COMPAT_EXPORTS = {
    "MaskedLinear": _MaskedLinear,
    "LSM": _LSM,
    "LSMExpander": _LSMExpander,
    "MaskedConv2d": _MaskedConv2d,
    "LSMConv2d": _LSMConv2d,
    "LSMConv2dExpander": _LSMConv2dExpander,
}


def _set_deserialization_warning_suppressed(value: bool) -> None:
    """Internal checkpoint-reader guard; never part of the public API."""

    global _SUPPRESS_DEPRECATION_WARNING
    _SUPPRESS_DEPRECATION_WARNING = value


def __getattr__(name: str) -> Any:
    try:
        value = _COMPAT_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    if not _SUPPRESS_DEPRECATION_WARNING:
        warnings.warn(
            "psann.lsm is deprecated; import LSM helpers from psann.preprocessing.",
            DeprecationWarning,
            stacklevel=2,
        )
    return value
