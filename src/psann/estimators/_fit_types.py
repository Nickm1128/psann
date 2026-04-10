from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Protocol, Tuple, Union

import numpy as np
import torch.nn as nn

from ..hisso import HISSOOptions, HISSOTrainerConfig
from ..types import NoiseSpec

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


ValidationPair = Tuple[np.ndarray, np.ndarray]
ValidationTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]
ValidationInput = Union[ValidationPair, ValidationTriple]


@dataclass
class NormalisedFitArgs:
    """Canonical view of the arguments supplied to ``fit``."""

    X: np.ndarray
    y: Optional[np.ndarray]
    context: Optional[np.ndarray]
    validation: Optional[ValidationInput]
    hisso: bool
    hisso_options: Optional[HISSOOptions]
    noisy: Optional[NoiseSpec]
    verbose: int
    lr_max: Optional[float]
    lr_min: Optional[float]


@dataclass
class PreparedInputState:
    """Intermediate artefacts produced after scaler/shape handling."""

    X_flat: np.ndarray
    X_cf: Optional[np.ndarray]
    context: Optional[np.ndarray]
    input_shape: Tuple[int, ...]
    internal_shape_cf: Optional[Tuple[int, ...]]
    scaler_transform: Optional[Callable[[np.ndarray], np.ndarray]]
    train_inputs: np.ndarray
    train_context: Optional[np.ndarray]
    train_targets: Optional[np.ndarray]
    y_vector: Optional[np.ndarray]
    y_cf: Optional[np.ndarray]
    context_dim: Optional[int]
    primary_dim: int
    output_dim: int


@dataclass
class ModelBuildRequest:
    """Bundle of information required to construct the estimator core."""

    estimator: "PSANNRegressor"
    prepared: PreparedInputState
    primary_dim: int
    lsm_module: Optional[nn.Module]
    lsm_output_dim: Optional[int]
    preserve_shape: bool


@dataclass
class HISSOTrainingPlan:
    """Precomputed artefacts required to launch HISSO training."""

    inputs: np.ndarray
    primary_dim: int
    trainer_config: HISSOTrainerConfig
    allow_full_window: bool
    options: HISSOOptions
    lsm_module: Optional[nn.Module]


class ModelFactory(Protocol):
    def __call__(self, request: ModelBuildRequest) -> nn.Module: ...


class PreprocFactory(Protocol):
    def __call__(self, request: ModelBuildRequest) -> Optional[nn.Module]: ...


class HISSOPlanFactory(Protocol):
    def __call__(
        self,
        estimator: "PSANNRegressor",
        request: ModelBuildRequest,
        *,
        fit_args: NormalisedFitArgs,
    ) -> Optional[HISSOTrainingPlan]: ...


@dataclass
class FitVariantHooks:
    """Declarative hooks that let estimator variants share the pipeline."""

    build_model: ModelFactory
    build_preprocessor: Optional[PreprocFactory] = None
    build_hisso_plan: Optional[HISSOPlanFactory] = None

    def wants_hisso(self) -> bool:
        return self.build_hisso_plan is not None
