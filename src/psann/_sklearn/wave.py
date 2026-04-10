from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn

from ..attention import AttentionConfig, build_attention_module
from ..conv import PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet
from ..layers import SpectralGate1D
from ..models import WaveResNet
from ..nn import WithPreprocessor
from ..preproc import PreprocessorLike
from ..state import StateConfig
from ..types import ActivationConfig, LossLike, ScalerSpec
from .base import PSANNRegressor
from .shared import _WaveResNetConvModel, _WaveResNetSpectralDenseModel


class WaveResNetRegressor(PSANNRegressor):
    """Sklearn-style regressor that wraps the WaveResNet backbone."""

    def __init__(
        self,
        *,
        hidden_layers: int = 6,
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
        state_reset: str = "batch",
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
        first_layer_w0: float = 30.0,
        hidden_w0: float = 1.0,
        norm: Literal["none", "weight", "rms"] = "none",
        use_film: bool = True,
        use_phase_shift: bool = True,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        context_builder: Optional[Union[str, Callable[[np.ndarray], np.ndarray]]] = None,
        context_builder_params: Optional[Dict[str, Any]] = None,
        residual_alpha_init: float = 0.0,
        grad_clip_norm: Optional[float] = 5.0,
        first_layer_w0_initial: Optional[float] = 10.0,
        hidden_w0_initial: Optional[float] = 0.5,
        w0_warmup_epochs: int = 10,
        progressive_depth_initial: Optional[int] = None,
        progressive_depth_interval: int = 15,
        progressive_depth_growth: int = 1,
        # Optional spectral gating (sequence inputs)
        use_spectral_gate: bool = False,
        k_fft: int = 64,
        gate_type: str = "rfft",
        gate_groups: str = "depthwise",
        gate_init: float = 0.0,
        gate_strength: float = 1.0,
    ) -> None:
        if per_element:
            raise ValueError("WaveResNetRegressor does not support per_element=True.")
        if not preserve_shape:
            if conv_channels is not None:
                warnings.warn(
                    "conv_channels has no effect for WaveResNetRegressor when preserve_shape=False; ignoring value.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if conv_kernel_size != 1:
                warnings.warn(
                    "conv_kernel_size has no effect for WaveResNetRegressor when preserve_shape=False; ignoring value.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if preserve_shape and lsm is not None:
            raise ValueError(
                "WaveResNetRegressor does not support lsm preprocessors when preserve_shape=True."
            )

        norm_value = str(norm).lower()
        if norm_value not in {"none", "weight", "rms"}:
            raise ValueError("norm must be one of {'none', 'weight', 'rms'}.")

        if context_dim is None:
            context_val = None
        else:
            context_val = int(context_dim)
            if context_val <= 0:
                raise ValueError("context_dim must be positive when provided.")

        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive when provided.")

        if first_layer_w0_initial is not None and first_layer_w0_initial <= 0:
            raise ValueError("first_layer_w0_initial must be positive when provided.")
        if hidden_w0_initial is not None and hidden_w0_initial <= 0:
            raise ValueError("hidden_w0_initial must be positive when provided.")

        warmup_epochs = int(w0_warmup_epochs)
        if warmup_epochs < 0:
            raise ValueError("w0_warmup_epochs must be non-negative.")

        if k_fft <= 0:
            raise ValueError("k_fft must be positive.")
        gate_type_value = str(gate_type).lower()
        if gate_type_value not in {"rfft", "fourier_features"}:
            raise ValueError("gate_type must be 'rfft' or 'fourier_features'.")
        gate_groups_value = str(gate_groups).lower()
        if gate_groups_value not in {"depthwise", "full"}:
            raise ValueError("gate_groups must be 'depthwise' or 'full'.")
        if gate_strength < 0:
            raise ValueError("gate_strength must be >= 0.")

        if progressive_depth_initial is not None:
            init_layers = int(progressive_depth_initial)
            if init_layers <= 0:
                raise ValueError("progressive_depth_initial must be positive when provided.")
            if init_layers > int(hidden_layers):
                raise ValueError("progressive_depth_initial cannot exceed hidden_layers.")
        else:
            init_layers = None

        grow_interval = int(progressive_depth_interval)
        if init_layers is not None and grow_interval <= 0:
            raise ValueError(
                "progressive_depth_interval must be positive when progressive depth is enabled."
            )

        growth = int(progressive_depth_growth)
        if init_layers is not None and growth <= 0:
            raise ValueError(
                "progressive_depth_growth must be positive when progressive depth is enabled."
            )

        if stateful or state is not None:
            warnings.warn(
                "WaveResNetRegressor does not support stateful configurations; ignoring state/stateful.",
                RuntimeWarning,
                stacklevel=2,
            )
        stateful_flag = False
        state_cfg = None

        super().__init__(
            hidden_layers=hidden_layers,
            hidden_width=hidden_width,
            hidden_units=hidden_units,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            activation=activation,
            device=device,
            random_state=random_state,
            early_stopping=early_stopping,
            patience=patience,
            num_workers=num_workers,
            loss=loss,
            loss_params=loss_params,
            loss_reduction=loss_reduction,
            w0=w0,
            preserve_shape=preserve_shape,
            data_format=data_format,
            conv_kernel_size=conv_kernel_size,
            conv_channels=conv_channels,
            per_element=per_element,
            activation_type=activation_type,
            attention=attention,
            stateful=stateful_flag,
            state=state_cfg,
            state_reset=state_reset,
            stream_lr=stream_lr,
            output_shape=output_shape,
            lsm=lsm,
            lsm_train=lsm_train,
            lsm_pretrain_epochs=lsm_pretrain_epochs,
            lsm_lr=lsm_lr,
            warm_start=warm_start,
            scaler=scaler,
            scaler_params=scaler_params,
            target_scaler=target_scaler,
            target_scaler_params=target_scaler_params,
            context_builder=context_builder,
            context_builder_params=context_builder_params,
        )

        self.first_layer_w0 = float(first_layer_w0)
        self.hidden_w0 = float(hidden_w0)
        self.first_layer_w0_initial = (
            float(first_layer_w0_initial)
            if first_layer_w0_initial is not None
            else self.first_layer_w0
        )
        self.hidden_w0_initial = (
            float(hidden_w0_initial) if hidden_w0_initial is not None else self.hidden_w0
        )
        self.w0_warmup_epochs = warmup_epochs
        self.norm = norm_value
        self.use_film = bool(use_film)
        self.use_phase_shift = bool(use_phase_shift)
        self.dropout = float(dropout)
        self.context_dim = context_val
        self._context_dim_ = context_val
        self.residual_alpha_init = float(residual_alpha_init)
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

        self._w0_schedule_active = False
        self._w0_schedule_step = 0

        self.progressive_depth_initial = init_layers
        self.progressive_depth_interval = grow_interval
        self.progressive_depth_growth = growth
        self._progressive_depth_active = False
        self._progressive_depth_current = int(self.hidden_layers)
        self._progressive_next_expand_epoch: Optional[int] = None

        self._wave_hidden_dim = int(self.hidden_units)
        self.use_spectral_gate = bool(use_spectral_gate)
        self.k_fft = int(k_fft)
        self.gate_type = gate_type_value
        self.gate_groups = gate_groups_value
        self.gate_init = float(gate_init)
        self.gate_strength = float(gate_strength)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
        context: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> "WaveResNetRegressor":
        validation_data = kwargs.get("validation_data")

        builder_active = self._get_context_builder() is not None

        inferred_context_dim: Optional[int] = None
        if context is not None:
            ctx_arr = np.asarray(context, dtype=np.float32)
            if ctx_arr.ndim == 1:
                ctx_arr = ctx_arr.reshape(-1, 1)
            inferred_context_dim = int(ctx_arr.shape[1])

        validation_context_dim: Optional[int] = None
        if validation_data is not None and isinstance(validation_data, (tuple, list)):
            val_tuple = tuple(validation_data)
            if len(val_tuple) == 3 and val_tuple[2] is not None:
                val_ctx = np.asarray(val_tuple[2], dtype=np.float32)
                if val_ctx.ndim == 1:
                    val_ctx = val_ctx.reshape(-1, 1)
                validation_context_dim = int(val_ctx.shape[1])

        if inferred_context_dim is not None:
            if self.context_dim is not None and int(self.context_dim) != inferred_context_dim:
                raise ValueError(
                    f"Provided context feature dimension {inferred_context_dim} does not match "
                    f"configured context_dim={self.context_dim}."
                )
            self.context_dim = inferred_context_dim
            self._context_dim_ = inferred_context_dim
        elif self.context_dim is not None:
            if not builder_active:
                raise ValueError(
                    f"WaveResNetRegressor expects a context array matching context_dim={self.context_dim}; "
                    "received context=None."
                )
            self._context_dim_ = int(self.context_dim)
        elif builder_active:
            self._context_dim_ = None
        else:
            self._context_dim_ = None

        if validation_context_dim is not None:
            expected_dim = self.context_dim
            if expected_dim is None:
                if builder_active:
                    self.context_dim = validation_context_dim
                    self._context_dim_ = validation_context_dim
                else:
                    raise ValueError(
                        "Validation context was provided but estimator was constructed without context_dim. "
                        "Specify context during fit to enable context-aware validation."
                    )
            elif validation_context_dim != int(expected_dim):
                raise ValueError(
                    f"Validation context dimension {validation_context_dim} does not match expected {expected_dim}."
                )

        fitted = cast(
            "WaveResNetRegressor",
            super().fit(X, y, context=context, **kwargs),
        )
        if self._context_dim_ is not None:
            self.context_dim = int(self._context_dim_)
        return fitted

    def _build_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> nn.Module:
        if state_cfg is not None:
            warnings.warn(
                "WaveResNetRegressor ignores state_cfg; WaveResNet does not expose external state.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not self.use_spectral_gate:
            return super()._build_dense_core(
                input_dim,
                output_dim,
                state_cfg=None,
                input_shape=input_shape,
            )

        if self._attention_enabled():
            warnings.warn(
                "WaveResNetRegressor ignores attention when use_spectral_gate=True for flattened inputs.",
                RuntimeWarning,
                stacklevel=2,
            )

        if input_shape is None or len(input_shape) < 2:
            return self._build_dense_backbone(
                input_dim,
                output_dim,
                state_cfg=None,
            )
        seq_dims = input_shape[:-1]
        token_dim = int(input_shape[-1])
        seq_len = int(math.prod(seq_dims)) if seq_dims else 1
        expected = seq_len * token_dim
        if expected != int(input_dim):
            raise ValueError(
                "WaveResNetRegressor input shape does not match input_dim "
                f"(expected {expected}, received {input_dim})."
            )
        if seq_len <= 1:
            return self._build_dense_backbone(
                input_dim,
                output_dim,
                state_cfg=None,
            )

        wave_core = self._build_dense_backbone(
            input_dim,
            output_dim,
            state_cfg=None,
        )
        spectral_gate = SpectralGate1D(
            token_dim,
            k_fft=self.k_fft,
            gate_type=self.gate_type,
            gate_groups=self.gate_groups,
            gate_init=self.gate_init,
            gate_strength=self.gate_strength,
        )
        return _WaveResNetSpectralDenseModel(
            wave_core,
            spectral_gate,
            seq_len=seq_len,
            token_dim=token_dim,
        )

    def _build_dense_backbone(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        if state_cfg is not None:
            warnings.warn(
                "WaveResNetRegressor ignores state_cfg; WaveResNet does not expose external state.",
                RuntimeWarning,
                stacklevel=2,
            )
        init_first, init_hidden = self._initial_w0_values()
        depth = int(self.hidden_layers)
        if self._progressive_enabled():
            depth = int(self.progressive_depth_initial)
        activation_cfg = copy.deepcopy(self.activation)
        return WaveResNet(
            input_dim=int(input_dim),
            hidden_dim=int(self._wave_hidden_dim),
            depth=depth,
            output_dim=int(output_dim),
            first_layer_w0=init_first,
            hidden_w0=init_hidden,
            context_dim=self.context_dim,
            norm=self.norm,
            use_film=self.use_film,
            use_phase_shift=self.use_phase_shift,
            dropout=self.dropout,
            residual_alpha_init=self.residual_alpha_init,
            activation_config=activation_cfg,
        )

    def _build_conv_core(
        self,
        spatial_ndim: int,
        in_channels: int,
        output_dim: int,
        *,
        segmentation_head: bool,
        spatial_shape: Optional[Tuple[int, ...]] = None,
        state_cfg: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        if segmentation_head:
            raise ValueError(
                "WaveResNetRegressor does not support per_element=True in convolutional mode."
            )
        if spatial_shape is None:
            raise ValueError(
                "WaveResNetRegressor requires known spatial dimensions for preserve_shape inputs."
            )
        conv_map = {
            1: PSANNConv1dNet,
            2: PSANNConv2dNet,
            3: PSANNConv3dNet,
        }
        conv_cls = conv_map.get(int(spatial_ndim))
        if conv_cls is None:
            raise ValueError(
                f"Unsupported spatial dimensionality {spatial_ndim}; expected 1, 2, or 3."
            )
        conv_channels = int(self.conv_channels)
        conv_core = conv_cls(
            int(in_channels),
            out_dim=conv_channels,
            hidden_layers=self.hidden_layers,
            conv_channels=conv_channels,
            hidden_channels=conv_channels,
            kernel_size=self.conv_kernel_size,
            act_kw=self.activation,
            activation_type=self.activation_type,
            w0=self.w0,
            segmentation_head=False,
        )
        embed_dim = self._infer_conv_embed_dim(conv_core)
        seq_len = int(math.prod(spatial_shape)) if spatial_shape else 1
        attn_module: Optional[nn.Module] = None
        if self._attention_enabled():
            attn_module = build_attention_module(self.attention, embed_dim)
            if attn_module is not None:
                self._attention_shape_ = (seq_len, embed_dim)
        wave_input_dim = seq_len * embed_dim
        wave_core = self._build_dense_backbone(
            wave_input_dim,
            output_dim,
            state_cfg=state_cfg,
        )
        spectral_gate: Optional[SpectralGate1D] = None
        if self.use_spectral_gate:
            if spatial_ndim != 1:
                raise ValueError(
                    "WaveResNetRegressor spectral gating is only supported for 1D preserve_shape inputs."
                )
            if seq_len > 1:
                spectral_gate = SpectralGate1D(
                    embed_dim,
                    k_fft=self.k_fft,
                    gate_type=self.gate_type,
                    gate_groups=self.gate_groups,
                    gate_init=self.gate_init,
                    gate_strength=self.gate_strength,
                )
        return _WaveResNetConvModel(
            conv_core,
            wave_core,
            spatial_shape=spatial_shape,
            attention_module=attn_module,
            spectral_gate=spectral_gate,
        )

    def _initial_w0_values(self) -> Tuple[float, float]:
        return float(self.first_layer_w0_initial), float(self.hidden_w0_initial)

    def _target_w0_values(self) -> Tuple[float, float]:
        return float(self.first_layer_w0), float(self.hidden_w0)

    def _current_w0_values(self) -> Tuple[float, float]:
        if not self._use_w0_warmup():
            return self._target_w0_values()
        total = max(self.w0_warmup_epochs, 1)
        step = min(int(self._w0_schedule_step), total)
        ratio = float(step) / float(total)
        init_first, init_hidden = self._initial_w0_values()
        target_first, target_hidden = self._target_w0_values()
        first = init_first + (target_first - init_first) * ratio
        hidden = init_hidden + (target_hidden - init_hidden) * ratio
        return first, hidden

    def _use_w0_warmup(self) -> bool:
        init_first, init_hidden = self._initial_w0_values()
        target_first, target_hidden = self._target_w0_values()
        return self.w0_warmup_epochs > 0 and (
            not math.isclose(init_first, target_first)
            or not math.isclose(init_hidden, target_hidden)
        )

    def _progressive_enabled(self) -> bool:
        return self.progressive_depth_initial is not None and self.progressive_depth_initial < int(
            self.hidden_layers
        )

    def _wave_core(self) -> Optional[WaveResNet]:
        model = getattr(self, "model_", None)
        if isinstance(model, WaveResNet):
            return model
        if isinstance(model, _WaveResNetSpectralDenseModel):
            return model.wave
        if isinstance(model, _WaveResNetConvModel):
            return model.wave
        if isinstance(model, WithPreprocessor):
            core = model.core
            if isinstance(core, WaveResNet):
                return core
            if isinstance(core, _WaveResNetSpectralDenseModel):
                return core.wave
            if isinstance(core, _WaveResNetConvModel):
                return core.wave
        return None

    def _apply_w0_values(self, first_w0: float, hidden_w0: float) -> None:
        core = self._wave_core()
        if core is None:
            return
        value_first = float(first_w0)
        value_hidden = float(hidden_w0)
        core.stem_w0 = value_first
        for block in core.blocks:
            if hasattr(block, "w0"):
                block.w0 = value_hidden

    def _reset_w0_schedule(self) -> None:
        self._w0_schedule_step = 0
        if not self._use_w0_warmup():
            self._w0_schedule_active = False
            target_first, target_hidden = self._target_w0_values()
            self._apply_w0_values(target_first, target_hidden)
            return
        self._w0_schedule_active = True
        init_first, init_hidden = self._initial_w0_values()
        self._apply_w0_values(init_first, init_hidden)

    def _reset_progressive_depth(self) -> None:
        core = self._wave_core()
        if core is None:
            self._progressive_depth_active = False
            self._progressive_depth_current = int(self.hidden_layers)
            self._progressive_next_expand_epoch = None
            return
        self._progressive_depth_current = int(core.depth)
        if not self._progressive_enabled():
            self._progressive_depth_active = False
            self._progressive_next_expand_epoch = None
            return
        if self._progressive_depth_current >= int(self.hidden_layers):
            self._progressive_depth_active = False
            self._progressive_next_expand_epoch = None
            return
        self._progressive_depth_active = True
        self._progressive_next_expand_epoch = int(self.progressive_depth_interval)

    def _expand_progressive_depth(self) -> None:
        core = self._wave_core()
        if core is None or not self._progressive_depth_active:
            return
        target_depth = int(self.hidden_layers)
        if self._progressive_depth_current >= target_depth:
            self._progressive_depth_active = False
            self._progressive_next_expand_epoch = None
            return
        growth = min(
            int(self.progressive_depth_growth),
            target_depth - self._progressive_depth_current,
        )
        new_blocks = core.add_blocks(growth)
        self._progressive_depth_current += growth
        if self._progressive_depth_current >= target_depth:
            self._progressive_depth_active = False
            self._progressive_next_expand_epoch = None
        elif self.progressive_depth_interval > 0:
            next_epoch = (
                None
                if self._progressive_next_expand_epoch is None
                else int(self._progressive_next_expand_epoch)
            )
            self._progressive_next_expand_epoch = (
                None if next_epoch is None else next_epoch + int(self.progressive_depth_interval)
            )

        # Ensure new parameters are clipped into the warmup schedule
        first_w0, hidden_w0 = self._current_w0_values()
        self._apply_w0_values(first_w0, hidden_w0)

        optimizer = getattr(self, "_optimizer_", None)
        if optimizer is not None and new_blocks:
            new_params = []
            for block in new_blocks:
                new_params.extend(list(block.parameters()))
            if new_params:
                reference_group = optimizer.param_groups[0]
                param_group = {k: v for k, v in reference_group.items() if k != "params"}
                param_group["params"] = new_params
                optimizer.add_param_group(param_group)

    def _update_w0_schedule(self, next_epoch: int) -> None:
        if not self._w0_schedule_active:
            return
        total = max(self.w0_warmup_epochs, 1)
        step = min(int(next_epoch), total)
        init_first, init_hidden = self._initial_w0_values()
        target_first, target_hidden = self._target_w0_values()
        ratio = float(step) / float(total)
        new_first = init_first + (target_first - init_first) * ratio
        new_hidden = init_hidden + (target_hidden - init_hidden) * ratio
        self._apply_w0_values(new_first, new_hidden)
        self._w0_schedule_step = step
        if step >= total:
            self._w0_schedule_active = False

    def _after_model_built(self) -> None:
        super()._after_model_built()
        rebuilt = bool(getattr(self, "_model_rebuilt_", True))
        if rebuilt:
            self._reset_w0_schedule()
            self._reset_progressive_depth()
        else:
            self._w0_schedule_active = False
            self._progressive_depth_active = False
            self._progressive_next_expand_epoch = None

    def gradient_hook(self, model: nn.Module) -> None:
        if self.grad_clip_norm is None:
            return
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(self.grad_clip_norm))

    def epoch_callback(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        improved: bool,
        patience_left: Optional[int],
    ) -> None:
        if self._w0_schedule_active:
            self._update_w0_schedule(epoch + 1)
        if (
            self._progressive_depth_active
            and self._progressive_next_expand_epoch is not None
            and (epoch + 1) >= int(self._progressive_next_expand_epoch)
        ):
            self._expand_progressive_depth()


__all__ = ["WaveResNetRegressor"]
