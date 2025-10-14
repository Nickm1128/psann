from __future__ import annotations

from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
    Self,
)

import numpy as np
import torch


class ActivationConfig(TypedDict, total=False):
    amplitude_init: float
    frequency_init: float
    decay_init: float
    learnable: Iterable[str] | str
    decay_mode: str
    bounds: dict[str, Tuple[Optional[float], Optional[float]]]


LossCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
LossLike = Union[str, LossCallable]

NoiseSpec = Union[float, np.ndarray]
ArrayLike = Union[np.ndarray, torch.Tensor]
RewardFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ContextExtractor = Callable[[torch.Tensor], torch.Tensor]


@runtime_checkable
class TransformerProtocol(Protocol):
    """Minimal protocol for scaler/transformer style objects."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Self:
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    # Some scalers expose inverse_transform; leave optional to keep protocol lenient.


ScalerSpec = Union[str, TransformerProtocol]


class HISSOFitParams(TypedDict, total=False):
    hisso_window: Optional[int]
    hisso_reward_fn: Optional[RewardFn]
    hisso_context_extractor: Optional[ContextExtractor]
    hisso_primary_transform: Optional[str]
    hisso_transition_penalty: Optional[float]
    hisso_trans_cost: Optional[float]
    hisso_supervised: Optional[Mapping[str, Any] | bool]

