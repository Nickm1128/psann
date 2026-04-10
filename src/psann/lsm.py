from __future__ import annotations

"""Public LSM expanders and sparse helper modules."""

from ._lsm.common import TensorLike
from ._lsm.conv import LSMConv2d, LSMConv2dExpander, MaskedConv2d
from ._lsm.dense import LSM, LSMExpander, MaskedLinear

__all__ = [
    "TensorLike",
    "MaskedLinear",
    "LSM",
    "LSMExpander",
    "MaskedConv2d",
    "LSMConv2d",
    "LSMConv2dExpander",
]

for _cls in (MaskedLinear, LSM, LSMExpander, MaskedConv2d, LSMConv2d, LSMConv2dExpander):
    _cls.__module__ = __name__
