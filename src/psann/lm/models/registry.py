"""Model base registry for PSANN-LM.

Maps string identifiers (e.g., "respsann", "waveresnet") to transformer
constructors. This keeps the public API decoupled from implementation
details and enables simple extensibility.
"""

from __future__ import annotations

from typing import Callable, Dict


_REGISTRY: Dict[str, Callable[..., object]] = {}


def register_base(name: str, factory: Callable[..., object]) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Base name cannot be empty")
    if not callable(factory):
        raise TypeError("Factory must be callable")
    _REGISTRY[key] = factory


def get_base(name: str) -> Callable[..., object]:
    key = name.strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown base {name!r}. Available: {available}")
    return _REGISTRY[key]


def list_bases() -> list[str]:  # pragma: no cover - trivial
    return sorted(_REGISTRY)


# Pre-register default bases once modules are importable.
try:  # pragma: no cover - import side-effect wiring
    from .transformer_respsann import build_respsann_transformer, build_sgrpsann_transformer
    from .transformer_vanilla import build_vanilla_transformer
    from .transformer_waveresnet import build_waveresnet_transformer

    register_base("respsann", build_respsann_transformer)
    register_base("sgrpsann", build_sgrpsann_transformer)
    register_base("transformer", build_vanilla_transformer)
    register_base("waveresnet", build_waveresnet_transformer)
except Exception:
    # Allow import during partial scaffolding
    pass
