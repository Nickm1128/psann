"""Canonical typed preprocessing boundary."""

from .._lsm.conv import LSMConv2d, LSMConv2dExpander
from .._lsm.dense import LSM, LSMExpander
from .config import (
    LSMConfig,
    LSMPretrainingConfig,
    ModulePreprocessorConfig,
    PreprocessorConfig,
    PreprocessorLike,
    PreprocessorTrainingConfig,
    normalize_preprocessor,
    normalize_legacy_lsm,
    preprocessor_to_mapping,
)
from .runtime import (
    PreprocessorBuildRequest,
    PreprocessorBuildResult,
    PreprocessorCapabilities,
    declared_preprocessor_capabilities,
    prepare_preprocessor,
    validate_preprocessor_capability,
)

__all__ = [
    "LSM",
    "LSMConfig",
    "LSMConv2d",
    "LSMConv2dExpander",
    "LSMExpander",
    "LSMPretrainingConfig",
    "ModulePreprocessorConfig",
    "PreprocessorConfig",
    "PreprocessorBuildRequest",
    "PreprocessorBuildResult",
    "PreprocessorCapabilities",
    "declared_preprocessor_capabilities",
    "PreprocessorLike",
    "PreprocessorTrainingConfig",
    "normalize_preprocessor",
    "normalize_legacy_lsm",
    "preprocessor_to_mapping",
    "prepare_preprocessor",
    "validate_preprocessor_capability",
]
