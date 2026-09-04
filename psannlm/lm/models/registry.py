"""0.x base spellings delegating to the canonical typed registry."""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from ...architectures.compat import BASE_KINDS, compatibility_warning, legacy_lm_config
from ...architectures.registry import (
    LMBuildRequest,
    LMBuildResult,
    LMCapabilities,
    build_lm_model,
    register_lm_builder,
)


def get_base(name: str) -> Callable[..., nn.Module]:
    key = name.strip().lower()
    if key not in BASE_KINDS:
        raise KeyError(f"Unknown base {name!r}. Available: {tuple(BASE_KINDS)}")

    def factory(**kwargs: Any) -> nn.Module:
        return build_lm_model(legacy_lm_config(key, kwargs)).model

    return factory


def list_bases() -> list[str]:
    compatibility_warning("list_bases is deprecated; use available_lm_architectures.")
    return sorted(BASE_KINDS)


def register_base(name: str, factory: Callable[..., nn.Module], *, replace: bool = False) -> None:
    """Adapt a 0.x factory replacement to the one canonical request registry."""
    key = name.strip().lower()
    kind = BASE_KINDS.get(key, key)

    def builder(request: LMBuildRequest) -> LMBuildResult:
        config = request.config
        model = factory(config=config)
        return LMBuildResult(model, LMCapabilities(config.architecture.kind))

    register_lm_builder(kind, builder, replace=replace)
    compatibility_warning(
        "register_base is deprecated; factories now receive config=LMConfig. Use register_lm_builder."
    )
