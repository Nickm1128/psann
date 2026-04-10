from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ..attention import AttentionConfig
from ..nn_geo_sparse import GeoSparseNet
from ..preproc import PreprocessorLike
from ..state import StateConfig
from ..types import ActivationConfig, LossLike, ScalerSpec
from .base import PSANNRegressor


class GeoSparseRegressor(PSANNRegressor):
    """Sklearn-style regressor using the GeoSparseNet backbone.

    Shapes and dtype:
    - X: float32 array shaped (N, H, W) or (N, H * W) with shape=(H, W) provided.
    - y: float32 array shaped (N,) or (N, target_dim).

    Note: hidden_layers controls the sparse depth. hidden_units/hidden_width are unused.
    """

    def __init__(
        self,
        *,
        hidden_layers: int = 4,
        hidden_width: Optional[int] = None,
        hidden_units: Optional[int] = None,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        activation: Optional[ActivationConfig | Mapping[str, Any]] = None,
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
        amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        compile_fullgraph: bool = False,
        compile_dynamic: bool = False,
        # geo-specific
        shape: Optional[Tuple[int, int]] = None,
        k: int = 8,
        pattern: str = "local",
        radius: int = 1,
        offsets: Optional[Sequence[Tuple[int, int]]] = None,
        wrap_mode: str = "clamp",
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
        bias: bool = True,
        compute_mode: str = "gather",
        geo_seed: Optional[int] = None,
    ) -> None:
        if preserve_shape:
            warnings.warn(
                "GeoSparseRegressor ignores preserve_shape; using flattened inputs.",
                RuntimeWarning,
                stacklevel=2,
            )
        if per_element:
            warnings.warn(
                "GeoSparseRegressor does not support per_element; ignoring.",
                RuntimeWarning,
                stacklevel=2,
            )
        if attention is not None:
            warnings.warn(
                "GeoSparseRegressor ignores attention for now.",
                RuntimeWarning,
                stacklevel=2,
            )
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
            preserve_shape=False,
            data_format=data_format,
            conv_kernel_size=conv_kernel_size,
            conv_channels=conv_channels,
            per_element=False,
            activation_type=activation_type,
            attention=None,
            stateful=stateful,
            state=state,
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
            amp=amp,
            amp_dtype=amp_dtype,
            compile=compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
        )
        self.geo_shape = tuple(shape) if shape is not None else None
        self.geo_k = int(k)
        self.geo_pattern = str(pattern)
        self.geo_radius = int(radius)
        self.geo_offsets = list(offsets) if offsets is not None else None
        self.geo_wrap_mode = str(wrap_mode)
        self.geo_norm = str(norm)
        self.geo_drop_path_max = float(drop_path_max)
        self.geo_residual_alpha_init = float(residual_alpha_init)
        self.geo_bias = bool(bias)
        self.geo_compute_mode = str(compute_mode)
        self.geo_seed = geo_seed if geo_seed is not None else random_state

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
                "GeoSparseRegressor does not support stateful configurations; ignoring state_cfg.",
                RuntimeWarning,
                stacklevel=2,
            )
        shape = self._resolve_geo_shape(input_dim, input_shape)
        return GeoSparseNet(
            int(input_dim),
            int(output_dim),
            shape=shape,
            depth=int(self.hidden_layers),
            k=int(self.geo_k),
            pattern=self.geo_pattern,
            radius=int(self.geo_radius),
            offsets=self.geo_offsets,
            wrap_mode=self.geo_wrap_mode,
            activation_type=self.activation_type,
            activation_config=self.activation,
            norm=self.geo_norm,
            drop_path_max=self.geo_drop_path_max,
            residual_alpha_init=self.geo_residual_alpha_init,
            bias=self.geo_bias,
            compute_mode=self.geo_compute_mode,
            seed=self.geo_seed,
        )

    def _build_conv_core(
        self,
        spatial_ndim: int,
        in_channels: int,
        output_dim: int,
        *,
        segmentation_head: bool,
        spatial_shape: Optional[Tuple[int, ...]] = None,
    ) -> nn.Module:
        raise ValueError("GeoSparseRegressor does not support preserve_shape inputs.")

    def _resolve_geo_shape(
        self,
        input_dim: int,
        input_shape: Optional[Tuple[int, ...]],
    ) -> Tuple[int, int]:
        shape = self.geo_shape
        if shape is None and input_shape is not None and len(input_shape) == 2:
            shape = (int(input_shape[0]), int(input_shape[1]))
        if shape is None:
            raise ValueError(
                "GeoSparseRegressor requires shape=(H, W) or inputs with shape (N, H, W)."
            )
        if len(shape) != 2:
            raise ValueError("shape must be (H, W).")
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            raise ValueError("shape dimensions must be positive.")
        if int(input_dim) != height * width:
            raise ValueError(
                "input_dim must match shape height * width "
                f"(expected {height * width}, received {input_dim})."
            )
        return height, width


__all__ = ["GeoSparseRegressor"]
