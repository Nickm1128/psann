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
    preprocessor_to_mapping,
)
from .runtime import (
    PreprocessorBuildRequest,
    PreprocessorBuildResult,
    PreprocessorCapabilities,
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
    "PreprocessorLike",
    "PreprocessorTrainingConfig",
    "normalize_preprocessor",
    "preprocessor_to_mapping",
    "prepare_preprocessor",
    "validate_preprocessor_capability",
]
