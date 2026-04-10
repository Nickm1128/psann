from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..attention import AttentionConfig
from ..nn import SGRPSANNSequenceNet
from ..preproc import PreprocessorLike
from ..state import StateConfig
from ..types import ActivationConfig, LossLike, ScalerSpec
from .base import PSANNRegressor


class SGRPSANNRegressor(PSANNRegressor):
    """Sklearn-style regressor with phase-shifted sine blocks and spectral gating."""

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
        # SGR-specific
        phase_init: float = 0.0,
        phase_trainable: bool = True,
        use_spectral_gate: bool = True,
        k_fft: int = 64,
        gate_type: str = "rfft",
        gate_groups: str = "depthwise",
        gate_init: float = 0.0,
        gate_strength: float = 1.0,
        pool: str = "last",
    ) -> None:
        if preserve_shape:
            raise ValueError("SGRPSANNRegressor does not support preserve_shape=True.")
        if per_element:
            raise ValueError("SGRPSANNRegressor does not support per_element=True.")
        if lsm is not None:
            warnings.warn(
                "SGRPSANNRegressor does not support LSM preprocessors; ignoring lsm settings.",
                RuntimeWarning,
                stacklevel=2,
            )
            lsm = None
            lsm_train = False
            lsm_pretrain_epochs = 0
            lsm_lr = None
        if str(activation_type).lower() != "psann":
            raise ValueError("SGRPSANNRegressor requires activation_type='psann'.")
        if k_fft <= 0:
            raise ValueError("k_fft must be positive.")
        gate_type = str(gate_type).lower()
        if gate_type not in {"rfft", "fourier_features"}:
            raise ValueError("gate_type must be 'rfft' or 'fourier_features'.")
        gate_groups = str(gate_groups).lower()
        if gate_groups not in {"depthwise", "full"}:
            raise ValueError("gate_groups must be 'depthwise' or 'full'.")
        if gate_strength < 0:
            raise ValueError("gate_strength must be >= 0.")
        pool = str(pool).lower()
        if pool not in {"mean", "last"}:
            raise ValueError("pool must be 'mean' or 'last'.")

        if stateful or state is not None:
            warnings.warn(
                "SGRPSANNRegressor does not support stateful configurations; ignoring state/stateful.",
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
        )

        self.phase_init = float(phase_init)
        self.phase_trainable = bool(phase_trainable)
        self.use_spectral_gate = bool(use_spectral_gate)
        self.k_fft = int(k_fft)
        self.gate_type = gate_type
        self.gate_groups = gate_groups
        self.gate_init = float(gate_init)
        self.gate_strength = float(gate_strength)
        self.pool = pool

    def _build_sequence_core(
        self,
        *,
        seq_len: int,
        token_dim: int,
        output_dim: int,
    ) -> nn.Module:
        return SGRPSANNSequenceNet(
            seq_len=seq_len,
            token_dim=token_dim,
            output_dim=int(output_dim),
            hidden_layers=self.hidden_layers,
            hidden_units=self.hidden_units,
            hidden_width=self.hidden_width,
            act_kw=self.activation,
            activation_type=self.activation_type,
            w0=self.w0,
            phase_init=self.phase_init,
            phase_trainable=self.phase_trainable,
            use_spectral_gate=self.use_spectral_gate,
            k_fft=self.k_fft,
            gate_type=self.gate_type,
            gate_groups=self.gate_groups,
            gate_init=self.gate_init,
            gate_strength=self.gate_strength,
            pool=self.pool,
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
                "SGRPSANNRegressor does not support stateful configurations; ignoring state_cfg.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self._build_sequence_core(seq_len=1, token_dim=int(input_dim), output_dim=output_dim)

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
                "SGRPSANNRegressor does not support stateful configurations; ignoring state_cfg.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self._attention_enabled():
            warnings.warn(
                "SGRPSANNRegressor ignores attention; spectral gating uses the sequence axis.",
                RuntimeWarning,
                stacklevel=2,
            )
        if input_shape is None or len(input_shape) < 1:
            raise ValueError("SGRPSANNRegressor requires input_shape with at least one dimension.")
        seq_dims = input_shape[:-1]
        token_dim = int(input_shape[-1])
        seq_len = int(math.prod(seq_dims)) if seq_dims else 1
        expected = seq_len * token_dim
        if expected != int(input_dim):
            raise ValueError(
                "SGRPSANNRegressor input shape does not match input_dim "
                f"(expected {expected}, received {input_dim})."
            )
        return self._build_sequence_core(
            seq_len=seq_len,
            token_dim=token_dim,
            output_dim=output_dim,
        )

    def _build_conv_core(
        self,
        spatial_ndim: int,
        in_channels: int,
        output_dim: int,
        *,
        segmentation_head: bool,
    ) -> nn.Module:
        raise ValueError("SGRPSANNRegressor does not support preserve_shape inputs.")


__all__ = ["SGRPSANNRegressor"]
