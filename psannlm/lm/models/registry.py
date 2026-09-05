"""0.x base spellings delegating to the canonical typed registry."""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from ...architectures.compat import BASE_KINDS, compatibility_warning, legacy_lm_config
from ...architectures.registry import (
    build_lm_model,
    legacy_factory,
    legacy_factory_names,
    register_legacy_factory,
)


def get_base(name: str) -> Callable[..., nn.Module]:
    key = name.strip().lower()
    registered = legacy_factory(key)
    if key not in BASE_KINDS and registered is None:
        raise KeyError(f"Unknown base {name!r}. Available: {tuple(BASE_KINDS)}")

    def factory(**kwargs: Any) -> nn.Module:
        if registered is not None:
            compatibility_warning(
                "get_base is deprecated; this external 0.x factory retains its original keyword contract."
            )
            return registered(**kwargs)
        return build_lm_model(legacy_lm_config(key, kwargs)).model

    return factory


def list_bases() -> list[str]:
    compatibility_warning("list_bases is deprecated; use available_lm_architectures.")
    return sorted(set(BASE_KINDS) | set(legacy_factory_names()))


def register_base(name: str, factory: Callable[..., nn.Module], *, replace: bool = False) -> None:
    """Keep external 0.x factory signatures in the canonical registry's legacy namespace."""
    key = name.strip().lower()
    if not key:
        raise ValueError("registry.name must be nonempty.")
    if key in BASE_KINDS and not replace:
        raise ValueError(f"registry.{key} is already registered; replacement must be explicit.")
    register_legacy_factory(key, factory, replace=replace)
    compatibility_warning(
        "register_base is deprecated; replace_lm_builder replaces an existing typed kind. "
        "External 0.x names retain their legacy factory contract."
    )
