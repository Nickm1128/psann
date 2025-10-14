from __future__ import annotations

from typing import Iterable, Optional, Tuple, TypedDict


class ActivationConfig(TypedDict, total=False):
    amplitude_init: float
    frequency_init: float
    decay_init: float
    learnable: Iterable[str] | str
    decay_mode: str
    bounds: dict[str, Tuple[Optional[float], Optional[float]]]

