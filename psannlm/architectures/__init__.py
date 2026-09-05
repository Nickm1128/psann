"""Canonical immutable LM policies and the shared model construction path."""

from .config import (
    LMActivationInitializationConfig,
    LMArchitectureConfig,
    LMConfig,
    LMGeometryExecutionConfig,
    LMTemporalConfig,
    normalize_architecture,
    normalize_lm_config,
    to_mapping,
)
from .registry import (
    LMBuildRequest,
    LMBuildResult,
    LMCapabilities,
    available_lm_architectures,
    build_lm_model,
    register_lm_builder as register_lm_builder,
    replace_lm_builder,
)

__all__ = [
    "LMActivationInitializationConfig",
    "LMArchitectureConfig",
    "LMConfig",
    "LMGeometryExecutionConfig",
    "LMTemporalConfig",
    "normalize_architecture",
    "normalize_lm_config",
    "to_mapping",
    "LMBuildRequest",
    "LMBuildResult",
    "LMCapabilities",
    "available_lm_architectures",
    "build_lm_model",
    "replace_lm_builder",
]
