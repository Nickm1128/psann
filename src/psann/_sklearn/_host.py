"""Structural host contract for the shared estimator lifecycle mixins.

Annotations describe constructor and fitted state owned by the estimator. This
protocol is used only for static checking; it does not alter the runtime MRO.
"""

from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
)
import numpy as np
import torch
import torch.nn as nn
from ..attention import AttentionConfig
from ..state import StateConfig
from ..preproc import PreprocessorLike
from ..types import LossLike, NoiseSpec, ScalerSpec
from ..estimators._fit_types import FitVariantHooks, PreparedInputState
from .shared import ValidationDataLike

if TYPE_CHECKING:
    from .base import PSANNRegressor
    from .inference import _PreparedPrediction


class EstimatorHost(Protocol):
    _amp_scaler_: Optional[Any]
    _attention_shape_: Optional[Tuple[int, int]]
    _context_builder_callable_: Optional[Callable[[np.ndarray], np.ndarray]]
    _context_dim_: Optional[int]
    _hisso_cache_: Optional[np.ndarray]
    _hisso_cfg_: Optional[Any]
    _hisso_context_extractor_: Optional[Any]
    _hisso_options_: Optional[Any]
    _hisso_reward_fn_: Optional[Any]
    _hisso_trained_: bool
    _hisso_trainer_: Optional[Any]
    _internal_input_shape_cf_: Optional[Tuple[int, ...]]
    _keep_column_output_: bool
    _lr_scheduler_: Optional[Any]
    _lsm_module_: Optional[nn.Module]
    _model_device_: Optional[torch.device]
    _optimizer_: Optional[torch.optim.Optimizer]
    _output_dim_: Optional[int]
    _output_shape_tuple_: Optional[Tuple[int, ...]]
    _primary_dim_: Optional[int]
    _scaler_fitted_: bool
    _scaler_kind_: Optional[str]
    _scaler_spec_: Optional[Dict[str, Any]]
    _scaler_state_: Optional[Dict[str, Any]]
    _stream_last_lr_: Optional[float]
    _stream_loss_: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    _stream_model_token_: Optional[int]
    _stream_opt_: Optional[torch.optim.Optimizer]
    _target_cf_shape_: Optional[Tuple[int, ...]]
    _target_scaler_fitted_: bool
    _target_scaler_kind_: Optional[str]
    _target_scaler_spec_: Optional[Dict[str, Any]]
    _target_scaler_state_: Optional[Dict[str, Any]]
    _target_vector_dim_: Optional[int]
    _train_inputs_layout_: str
    _training_state_token_: int
    _use_channel_first_train_inputs_: bool
    activation: Mapping[str, Any]
    activation_type: str
    amp: bool
    amp_dtype: Optional[Union[str, torch.dtype]]
    attention: AttentionConfig
    batch_size: int
    compile: bool
    compile_backend: str
    compile_dynamic: bool
    compile_fullgraph: bool
    compile_mode: str
    context_builder: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]]
    context_builder_params: Dict[str, Any]
    conv_channels: int
    conv_kernel_size: int
    data_format: str
    device: str | torch.device
    early_stopping: bool
    epochs: int
    hidden_layers: int
    hidden_units: int
    hidden_width: int
    input_shape_: Optional[Tuple[int, ...]]
    loss: LossLike
    loss_params: Optional[Dict[str, Any]]
    loss_reduction: str
    lr: float
    lsm: Optional[PreprocessorLike]
    lsm_lr: Optional[float]
    lsm_pretrain_epochs: int
    lsm_train: bool
    model_: nn.Module
    num_workers: int
    optimizer: str
    output_shape: Optional[Tuple[int, ...]]
    patience: int
    per_element: bool
    preserve_shape: bool
    random_state: Optional[int]
    scaler: Optional[ScalerSpec]
    scaler_params: Optional[Dict[str, Any]]
    state: Optional[StateConfig]
    state_reset: str
    stateful: bool
    stream_lr: Optional[float]
    target_scaler: Optional[ScalerSpec]
    target_scaler_params: Optional[Dict[str, Any]]
    w0: float
    warm_start: bool
    weight_decay: float

    def enable_conv_stem(
        self, *, data_format: Optional[str] = None, per_element: Optional[bool] = None
    ) -> "PSANNRegressor": ...

    def get_params(self, deep: bool = True) -> Any: ...

    def set_params(self, **params: Any) -> Any: ...

    def _attention_enabled(self) -> bool: ...

    def gradient_hook(self, _: nn.Module) -> None: ...

    def epoch_callback(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        improved: bool,
        patience_left: Optional[int],
    ) -> None: ...

    def _after_model_built(self) -> None: ...

    def _device(self) -> torch.device: ...

    def _infer_input_shape(self, X: np.ndarray) -> tuple: ...

    def _flatten(self, X: np.ndarray) -> np.ndarray: ...

    def _resolve_lsm_module(
        self, data: Any, *, preserve_shape: bool
    ) -> Tuple[Optional[nn.Module], Optional[int]]: ...

    def _build_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> nn.Module: ...

    def _build_dense_backbone(
        self, input_dim: int, output_dim: int, *, state_cfg: Optional[Dict[str, Any]] = None
    ) -> nn.Module: ...

    def _build_token_backbone(
        self, token_dim: int, embed_dim: int, *, state_cfg: Optional[Dict[str, Any]] = None
    ) -> nn.Module: ...

    def _build_attention_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]],
        input_shape: Optional[Tuple[int, ...]],
    ) -> nn.Module: ...

    def _infer_conv_embed_dim(self, core: nn.Module) -> int: ...

    def _wrap_with_attention_conv(
        self, core: nn.Module, spatial_shape: Optional[Tuple[int, ...]], *, segmentation_head: bool
    ) -> nn.Module: ...

    def _build_conv_core(
        self,
        spatial_ndim: int,
        in_channels: int,
        output_dim: int,
        *,
        segmentation_head: bool,
        spatial_shape: Optional[Tuple[int, ...]] = None,
    ) -> nn.Module: ...

    def _make_optimizer(self, model: torch.nn.Module, lr: Optional[float] = None) -> Any: ...

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer: ...

    def _make_loss(self) -> Any: ...

    def _make_per_element_fit_hooks(self) -> FitVariantHooks: ...

    def _make_conv_fit_hooks(
        self, *, prepared: PreparedInputState, verbose: int
    ) -> FitVariantHooks: ...

    def _make_flatten_fit_hooks(
        self, *, prepared: PreparedInputState, verbose: int
    ) -> FitVariantHooks: ...

    def _make_fit_hooks(self, *, prepared: PreparedInputState, verbose: int) -> FitVariantHooks: ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
        context: Optional[np.ndarray] = None,
        validation_data: Optional[ValidationDataLike] = None,
        verbose: int = 0,
        noisy: Optional[NoiseSpec] = None,
        hisso: bool = False,
        hisso_window: Optional[int] = None,
        hisso_batch_episodes: Optional[int] = None,
        hisso_updates_per_epoch: Optional[int] = None,
        hisso_reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        hisso_context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        hisso_primary_transform: Optional[str] = None,
        hisso_transition_penalty: Optional[float] = None,
        hisso_trans_cost: Optional[float] = None,
        hisso_supervised: Optional[Mapping[str, Any] | bool] = None,
        lr_max: Optional[float] = None,
        lr_min: Optional[float] = None,
    ) -> Any: ...

    @staticmethod
    def _normalize_param_aliases(params: Dict[str, Any]) -> Dict[str, Any]: ...

    def _ensure_model_device(self, device: torch.device) -> None: ...

    def _get_context_builder(self) -> Optional[Callable[[np.ndarray], np.ndarray]]: ...

    @staticmethod
    def _build_cosine_context_callable(
        *,
        frequencies: Optional[Union[int, Iterable[float]]] = None,
        include_sin: bool = True,
        include_cos: bool = True,
        normalise_input: bool = False,
    ) -> Callable[[np.ndarray], np.ndarray]: ...

    def _auto_context(self, features_2d: np.ndarray) -> Optional[np.ndarray]: ...

    def _make_internal_scaler(self) -> Optional[Dict[str, Any]]: ...

    def _scaler_fit_update(
        self, X2d: np.ndarray
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]: ...

    def _make_internal_target_scaler(self) -> Optional[Dict[str, Any]]: ...

    def _target_scaler_fit_update(
        self, y2d: np.ndarray
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]: ...

    def _apply_fitted_target_scaler(self, y2d: np.ndarray) -> np.ndarray: ...

    def _inverse_fitted_target_scaler(self, y2d: np.ndarray) -> np.ndarray: ...

    def _apply_fitted_target_scaler_like(self, y: np.ndarray) -> np.ndarray: ...

    def _inverse_fitted_target_scaler_like(self, y: np.ndarray) -> np.ndarray: ...

    def _inverse_fitted_target_scaler_tensor(self, values: torch.Tensor) -> torch.Tensor: ...

    def _scaler_inverse_tensor(
        self, X_ep: torch.Tensor, *, feature_dim: int = -1
    ) -> torch.Tensor: ...

    def _apply_fitted_scaler(self, X2d: np.ndarray) -> np.ndarray: ...

    def _ensure_fitted(self) -> None: ...

    def reset_state(self) -> None: ...

    def step(
        self,
        x: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        update_params: bool = False,
        update_state: bool = True,
    ) -> Any: ...

    def predict_sequence(
        self,
        X: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
        reset_state: bool = False,
        return_sequence: bool = False,
        update_state: bool = True,
    ) -> Any: ...

    def predict_sequence_online(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
        reset_state: bool = True,
        return_sequence: bool = True,
        update_state: bool = True,
    ) -> np.ndarray: ...

    def _sequence_rollout(
        self,
        X_seq: np.ndarray,
        *,
        context_seq: Optional[np.ndarray],
        targets: Optional[np.ndarray],
        reset_state: bool,
        update_params: bool,
        update_state: bool,
        return_sequence: bool,
    ) -> Any: ...

    def _coerce_sequence_inputs(self, sequence: np.ndarray) -> np.ndarray: ...

    def _coerce_sequence_context(self, context: np.ndarray, steps: int) -> np.ndarray: ...

    def _coerce_sequence_targets(self, targets: np.ndarray, steps: int) -> np.ndarray: ...

    def _ensure_streaming_ready(self) -> None: ...

    def _coerce_stream_target(
        self, target: np.ndarray, reference: torch.Tensor, device: torch.device
    ) -> torch.Tensor: ...

    def _apply_stream_update(
        self, inputs_np: np.ndarray, *, context_np: Optional[np.ndarray], target: np.ndarray
    ) -> None: ...

    def _build_serialized_payload(self, model_cpu: torch.nn.Module) -> Dict[str, Any]: ...

    def save(self, path: str) -> None: ...

    def _prepare_inference_inputs(
        self, X: np.ndarray, context: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]: ...

    def _reshape_predictions(self, preds: np.ndarray, meta: Dict[str, Any]) -> np.ndarray: ...

    def _run_model(
        self,
        inputs_np: np.ndarray,
        *,
        context_np: Optional[np.ndarray] = None,
        state_updates: bool = False,
        sequence: bool = False,
    ) -> np.ndarray: ...

    def predict(self, X: np.ndarray, *, context: Optional[np.ndarray] = None) -> np.ndarray: ...

    def _predict_with_prepared_inputs(
        self,
        X: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
        sequence: bool = False,
        reset_state: bool = False,
        update_state: bool = False,
    ) -> _PreparedPrediction: ...

    def score(
        self, X: np.ndarray, y: np.ndarray, *, context: Optional[np.ndarray] = None
    ) -> float: ...
