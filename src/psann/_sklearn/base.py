from __future__ import annotations

import copy
import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .._aliases import resolve_int_alias
from ..attention import AttentionConfig, ensure_attention_config
from ..preproc import PreprocessorLike
from ..state import StateConfig, ensure_state_config
from ..types import ActivationConfig, LossLike, ScalerSpec
from .builders import _PSANNRegressorBuilderMixin
from .inference import _PSANNRegressorInferenceMixin
from .scaling import _PSANNRegressorScalingMixin
from .sequence import _PSANNRegressorSequenceMixin
from .serialization import _PSANNRegressorSerializationMixin
from .shared import BaseEstimator, RegressorMixin


class PSANNRegressor(
    _PSANNRegressorSerializationMixin,
    _PSANNRegressorSequenceMixin,
    _PSANNRegressorBuilderMixin,
    _PSANNRegressorInferenceMixin,
    _PSANNRegressorScalingMixin,
    BaseEstimator,
    RegressorMixin,
):
    """Sklearn-style regressor wrapper around a PSANN network (PyTorch).

    Parameters mirror the README's proposed API.

    Shapes and dtype:
    - X: float32 array shaped (N, F) for flattened inputs, or (N, C, ...) / (N, ..., C)
      when preserve_shape=True.
    - y: float32 array shaped (N,) or (N, target_dim); when per_element=True, y can match
      the spatial layout of X.

    Defaults are chosen for CPU-friendly quick runs; set device="cuda" to train on GPU.
    """

    def __init__(
        self,
        *,
        hidden_layers: int = 2,
        hidden_width: Optional[int] = None,
        hidden_units: Optional[int] = None,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        activation: Optional[ActivationConfig] = None,
        device: str | torch.device = "auto",
        random_state: Optional[int] = None,
        early_stopping: bool = False,
        patience: int = 20,
        num_workers: int = 0,
        loss: LossLike = "mse",
        loss_params: Optional[Dict[str, Any]] = None,
        loss_reduction: str = "mean",
        w0: float = 30.0,
        preserve_shape: bool = False,
        data_format: str = "channels_first",
        conv_kernel_size: int = 1,
        conv_channels: Optional[int] = None,
        per_element: bool = False,
        activation_type: str = "psann",
        attention: Optional[AttentionConfig | Mapping[str, Any]] = None,
        stateful: bool = False,
        state: Optional[Union[StateConfig, Mapping[str, Any]]] = None,
        state_reset: str = "batch",  # 'batch' | 'epoch' | 'none'
        stream_lr: Optional[float] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        lsm: Optional[PreprocessorLike] = None,
        lsm_train: bool = False,
        lsm_pretrain_epochs: int = 0,
        lsm_lr: Optional[float] = None,
        warm_start: bool = False,
        scaler: Optional[ScalerSpec] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
        target_scaler: Optional[ScalerSpec] = None,
        target_scaler_params: Optional[Dict[str, Any]] = None,
        amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        compile_fullgraph: bool = False,
        compile_dynamic: bool = False,
        context_builder: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = None,
        context_builder_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.hidden_layers = int(hidden_layers)

        hidden_units_res = resolve_int_alias(
            primary_value=hidden_units,
            alias_value=hidden_width,
            primary_name="hidden_units",
            alias_name="hidden_width",
            context="PSANNRegressor",
            default=64,
        )
        user_set_hidden_units = hidden_units_res.provided_primary
        units = hidden_units_res.value if hidden_units_res.value is not None else 64
        self.hidden_units = units
        self.hidden_width = units

        user_set_conv = conv_channels is not None
        if user_set_conv and not preserve_shape:
            warnings.warn(
                "`conv_channels` has no effect when preserve_shape=False; ignoring value.",
                UserWarning,
                stacklevel=2,
            )

        conv_val = conv_channels if user_set_conv else units
        if conv_val is None:
            conv_val = units
        conv_val = int(conv_val)
        if user_set_conv and user_set_hidden_units and conv_val != units:
            warnings.warn(
                "`conv_channels` differs from `hidden_units`; using `conv_channels` for convolutional paths.",
                UserWarning,
                stacklevel=2,
            )
        self.conv_channels = conv_val

        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.optimizer = str(optimizer)
        self.weight_decay = float(weight_decay)
        self.activation = activation or {}
        self.device = device
        self.random_state = random_state
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.num_workers = int(num_workers)
        self.loss = loss
        self.loss_params = loss_params
        self.loss_reduction = loss_reduction
        self.w0 = float(w0)
        self.preserve_shape = bool(preserve_shape)
        self.data_format = str(data_format)
        self.conv_kernel_size = int(conv_kernel_size)
        self.per_element = bool(per_element)
        self.activation_type = activation_type
        self.attention = ensure_attention_config(attention)
        self.stateful = bool(stateful)
        self.state = ensure_state_config(state)
        self.state_reset = state_reset
        self.stream_lr = stream_lr
        self.output_shape = output_shape
        self.lsm = lsm
        self.lsm_train = bool(lsm_train)
        self.lsm_pretrain_epochs = int(lsm_pretrain_epochs)
        self.lsm_lr = lsm_lr
        self.warm_start = bool(warm_start)
        # Optional input scaler (minmax/standard or custom object with fit/transform)
        self.scaler = scaler
        self.scaler_params = scaler_params or None
        # Optional target scaler (minmax/standard or custom object with fit/transform)
        self.target_scaler = target_scaler
        self.target_scaler_params = target_scaler_params or None
        self.amp = bool(amp)
        self.amp_dtype = amp_dtype
        self.compile = bool(compile)
        self.compile_backend = str(compile_backend)
        self.compile_mode = str(compile_mode)
        self.compile_fullgraph = bool(compile_fullgraph)
        self.compile_dynamic = bool(compile_dynamic)
        self.context_builder = context_builder
        self.context_builder_params = (
            copy.deepcopy(context_builder_params) if context_builder_params is not None else {}
        )
        self._context_builder_callable_: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._use_channel_first_train_inputs_ = False
        self._preproc_cfg_ = {
            "lsm": lsm,
            "train": bool(lsm_train),
            "pretrain_epochs": int(lsm_pretrain_epochs),
        }
        self._lsm_module_ = None
        self._hisso_cache_: Optional[np.ndarray] = None
        self._hisso_trainer_: Optional[Any] = None
        self._hisso_options_: Optional[Any] = None
        self._hisso_reward_fn_: Optional[Any] = None
        self._hisso_context_extractor_: Optional[Any] = None
        self._hisso_cfg_: Optional[Any] = None
        self._hisso_trained_ = False

        # Training state caches
        self._optimizer_: Optional[torch.optim.Optimizer] = None
        self._lr_scheduler_: Optional[Any] = None
        self._amp_scaler_: Optional[Any] = None
        self._training_state_token_: int = 0
        self._stream_opt_: Optional[torch.optim.Optimizer] = None
        self._stream_loss_: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
        self._stream_model_token_: Optional[int] = None
        self._stream_last_lr_: Optional[float] = None

        # Fitted scaler state (set during fit)
        self._scaler_kind_: Optional[str] = None
        self._scaler_state_: Optional[Dict[str, Any]] = None
        self._target_scaler_kind_: Optional[str] = None
        self._target_scaler_state_: Optional[Dict[str, Any]] = None

        # Inference metadata
        self._train_inputs_layout_: str = "flat"
        self._primary_dim_: Optional[int] = None
        self._output_dim_: Optional[int] = None
        self._target_cf_shape_: Optional[Tuple[int, ...]] = None
        self._target_vector_dim_: Optional[int] = None
        self._output_shape_tuple_: Optional[Tuple[int, ...]] = (
            tuple(output_shape) if output_shape is not None else None
        )
        self._context_dim_: Optional[int] = None
        self._model_device_: Optional[torch.device] = None
        self._attention_shape_: Optional[Tuple[int, int]] = None

    @classmethod
    def with_conv_stem(
        cls,
        *,
        conv_channels: Optional[int] = None,
        conv_kernel_size: Optional[int] = None,
        per_element: bool = False,
        data_format: str = "channels_first",
        preserve_shape: bool = True,
        **kwargs: Any,
    ) -> "PSANNRegressor":
        """Instantiate an estimator configured with the convolutional fit path.

        This helper mirrors the historical ``*ConvPSANNRegressor`` classes by
        enabling ``preserve_shape`` training, ensuring channel-first tensors are
        used during optimisation, and forwarding the configured convolutional
        parameters. The returned estimator can be trained on 1D/2D/3D inputs
        without switching to a separate subclass.
        """

        params = dict(kwargs)
        params.setdefault("preserve_shape", preserve_shape)
        params.setdefault("data_format", data_format)
        params.setdefault("per_element", per_element)
        if conv_channels is not None:
            params["conv_channels"] = conv_channels
        if conv_kernel_size is not None:
            params["conv_kernel_size"] = int(conv_kernel_size)
        estimator = cls(**params)
        estimator.enable_conv_stem(
            data_format=estimator.data_format,
            per_element=estimator.per_element,
        )
        return estimator

    def enable_conv_stem(
        self,
        *,
        data_format: Optional[str] = None,
        per_element: Optional[bool] = None,
    ) -> "PSANNRegressor":
        """Switch the estimator to the convolutional training pipeline."""

        if data_format is not None:
            fmt = str(data_format).lower()
            if fmt not in {"channels_first", "channels_last"}:
                raise ValueError("data_format must be 'channels_first' or 'channels_last'.")
            self.data_format = fmt
        self.preserve_shape = True
        if per_element is not None:
            self.per_element = bool(per_element)
        self._use_channel_first_train_inputs_ = True
        self._conv_stem_config_ = {
            "data_format": self.data_format,
            "per_element": bool(self.per_element),
            "conv_kernel_size": int(self.conv_kernel_size),
            "conv_channels": int(self.conv_channels),
        }
        return self

    def set_params(self, **params: Any):
        if not params:
            return self
        normalised = self._normalize_param_aliases(params)
        reset_builder = False
        if "context_builder" in normalised:
            reset_builder = True
        if "context_builder_params" in normalised:
            reset_builder = True
            params_value = normalised.get("context_builder_params")
            if params_value is None:
                normalised["context_builder_params"] = {}
            else:
                normalised["context_builder_params"] = copy.deepcopy(params_value)
        result = super().set_params(**normalised)
        if reset_builder:
            self._context_builder_callable_ = None
            if getattr(self, "context_builder", None) is None and "context_dim" not in normalised:
                self._context_dim_ = None
                if hasattr(self, "context_dim") and "context_dim" not in normalised:
                    try:
                        setattr(self, "context_dim", None)
                    except Exception:
                        pass
        return result

    def _attention_enabled(self) -> bool:
        cfg = getattr(self, "attention", None)
        return bool(cfg and cfg.is_enabled())

    def gradient_hook(self, _: nn.Module) -> None:
        """Hook executed after backward before the optimiser step."""
        return None

    def epoch_callback(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        improved: bool,
        patience_left: Optional[int],
    ) -> None:
        """Hook executed at the end of each epoch."""
        return None

    def _after_model_built(self) -> None:
        """Extension point invoked after the core model has been (re)constructed."""
        return None


__all__ = ["PSANNRegressor"]
