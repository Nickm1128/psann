"""Single typed registry for built-in and explicitly replaced LM builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from .config import LMConfig, normalize_lm_config


@dataclass(frozen=True)
class LMBuildRequest:
    config: LMConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", normalize_lm_config(self.config, for_build=True))


@dataclass(frozen=True)
class LMCapabilities:
    kind: str
    kv_cache: bool = True
    gradient_checkpointing: bool = True
    positional_encodings: tuple[str, ...] = ("rope", "alibi", "sinusoidal")
    trainer_identifier: str = "psannlm.trainer"
    export_identifier: str = "psannlm.model"


@dataclass(frozen=True)
class LMBuildResult:
    model: nn.Module
    capabilities: LMCapabilities


Builder = Callable[[LMBuildRequest], LMBuildResult]


def _transformer(request: LMBuildRequest) -> LMBuildResult:
    from ..lm.models.transformer_vanilla import VanillaTransformer

    return LMBuildResult(VanillaTransformer(request.config), LMCapabilities("transformer"))


def _residual(request: LMBuildRequest) -> LMBuildResult:
    from ..lm.models.transformer_respsann import ResPSANNTransformer

    return LMBuildResult(ResPSANNTransformer(request.config), LMCapabilities("residual"))


def _wave(request: LMBuildRequest) -> LMBuildResult:
    from ..lm.models.transformer_waveresnet import WaveResNetTransformer

    return LMBuildResult(WaveResNetTransformer(request.config), LMCapabilities("wave"))


def _geometric(request: LMBuildRequest) -> LMBuildResult:
    from ..lm.models.transformer_geosparse import GeoSparseTransformer

    return LMBuildResult(GeoSparseTransformer(request.config), LMCapabilities("geometric-sparse"))


_BUILDERS: dict[str, Builder] = {
    "transformer": _transformer,
    "residual": _residual,
    "wave": _wave,
    "geometric-sparse": _geometric,
}


def register_lm_builder(name: str, builder: Builder, *, replace: bool = False) -> None:
    if not isinstance(name, str) or not name or name.strip() != name:
        raise ValueError("registry.name must be a nonempty canonical name.")
    if not callable(builder):
        raise TypeError("registry.builder must be callable.")
    if name in _BUILDERS and not replace:
        raise ValueError(f"registry.{name} is already registered; replacement must be explicit.")
    _BUILDERS[name] = builder


def available_lm_architectures() -> tuple[str, ...]:
    return tuple(name for name in _BUILDERS if not name.startswith("legacy:"))


@dataclass(frozen=True)
class LegacyFactoryRegistration:
    """Opaque 0.x extension stored in the same registry, outside typed dispatch.

    External 0.x factories retain their own keyword contract. Maintained model
    construction uses typed builders; registering this adapter does not replace
    a canonical architecture or claim canonical persistence for arbitrary modules.
    """

    factory: Callable[..., nn.Module]

    def __call__(self, request: LMBuildRequest) -> LMBuildResult:
        raise ValueError(
            "registry legacy extension requires get_base; register_lm_builder accepts typed requests."
        )


def register_legacy_factory(name: str, factory: Callable[..., nn.Module], *, replace: bool) -> None:
    if not callable(factory):
        raise TypeError("registry.factory must be callable.")
    register_lm_builder("legacy:" + name, LegacyFactoryRegistration(factory), replace=replace)


def legacy_factory(name: str) -> Callable[..., nn.Module] | None:
    entry = _BUILDERS.get("legacy:" + name)
    return entry.factory if isinstance(entry, LegacyFactoryRegistration) else None


def legacy_factory_names() -> tuple[str, ...]:
    return tuple(name.removeprefix("legacy:") for name in _BUILDERS if name.startswith("legacy:"))


def build_lm_model(config: object) -> LMBuildResult:
    request = LMBuildRequest(normalize_lm_config(config, for_build=True))
    result = _BUILDERS[request.config.architecture.kind](request)
    if not isinstance(result, LMBuildResult) or not isinstance(result.model, nn.Module):
        raise TypeError("registry builder must return LMBuildResult with a torch module.")
    if result.capabilities.kind != request.config.architecture.kind:
        raise ValueError("registry capabilities.kind disagrees with configuration.")
    setattr(result.model, "lm_config", request.config)
    setattr(result.model, "lm_capabilities", result.capabilities)
    return result
