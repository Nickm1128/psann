from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..attention import AttentionConfig
from ..conv import ResidualPSANNConv2dNet
from ..estimators._fit_utils import (
    ModelBuildRequest,
    build_model_from_hooks,
    maybe_run_hisso,
    normalise_fit_args,
    prepare_inputs_and_scaler,
)
from ..nn import ResidualPSANNNet
from ..preproc import PreprocessorLike
from ..state import StateConfig
from ..types import ActivationConfig, LossLike, NoiseSpec, ScalerSpec
from ..utils import seed_all
from .base import PSANNRegressor
from .shared import ValidationDataLike


class ResPSANNRegressor(PSANNRegressor):
    """Sklearn-style regressor using ResidualPSANNNet core.

    Adds residual-specific args while keeping .fit/.predict API identical,
    including HISSO training.
    """

    def __init__(
        self,
        *,
        hidden_layers: int = 8,
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
        # maintained for parity; not used in residual core
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
        # residual-specific
        w0_first: float = 12.0,
        w0_hidden: float = 1.0,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
    ) -> None:
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
        )
        self.w0_first = float(w0_first)
        self.w0_hidden = float(w0_hidden)
        self.norm = str(norm)
        self.drop_path_max = float(drop_path_max)
        self.residual_alpha_init = float(residual_alpha_init)
        if self.preserve_shape:
            self._use_channel_first_train_inputs_ = True

    def _build_dense_backbone(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        if state_cfg is not None:
            warnings.warn(
                "ResidualPSANNNet does not currently support stateful configurations; ignoring state_cfg.",
                RuntimeWarning,
                stacklevel=2,
            )
        return ResidualPSANNNet(
            int(input_dim),
            int(output_dim),
            hidden_layers=self.hidden_layers,
            hidden_units=self.hidden_units,
            hidden_width=self.hidden_width,
            act_kw=self.activation,
            activation_type=self.activation_type,
            w0_first=self.w0_first,
            w0_hidden=self.w0_hidden,
            norm=self.norm,
            drop_path_max=self.drop_path_max,
            residual_alpha_init=self.residual_alpha_init,
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
        if int(spatial_ndim) == 2:
            return ResidualPSANNConv2dNet(
                int(in_channels),
                int(output_dim),
                hidden_layers=self.hidden_layers,
                conv_channels=self.conv_channels,
                hidden_channels=self.conv_channels,
                kernel_size=self.conv_kernel_size,
                act_kw=self.activation,
                activation_type=self.activation_type,
                w0_first=self.w0_first,
                w0_hidden=self.w0_hidden,
                norm=self.norm,
                drop_path_max=self.drop_path_max,
                residual_alpha_init=self.residual_alpha_init,
                segmentation_head=bool(segmentation_head),
            )
        return super()._build_conv_core(
            spatial_ndim,
            in_channels,
            output_dim,
            segmentation_head=segmentation_head,
            spatial_shape=spatial_shape,
        )


class ResConvPSANNRegressor(ResPSANNRegressor):
    """Residual 2D convolutional PSANN regressor with HISSO support."""

    def __init__(
        self,
        *,
        hidden_layers: int = 6,
        hidden_width: Optional[int] = None,
        hidden_units: Optional[int] = None,
        epochs: int = 200,
        batch_size: int = 64,
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
        preserve_shape: bool = True,
        data_format: str = "channels_first",
        conv_kernel_size: int = 3,
        conv_channels: Optional[int] = None,
        per_element: bool = False,
        activation_type: str = "psann",
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
        w0_first: float = 12.0,
        w0_hidden: float = 1.0,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
    ) -> None:
        if not preserve_shape:
            warnings.warn(
                "ResConvPSANNRegressor forces preserve_shape=True; overriding provided value.",
                UserWarning,
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
            preserve_shape=True,
            data_format=data_format,
            conv_kernel_size=conv_kernel_size,
            conv_channels=conv_channels,
            per_element=per_element,
            activation_type=activation_type,
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
            w0_first=w0_first,
            w0_hidden=w0_hidden,
            norm=norm,
            drop_path_max=drop_path_max,
            residual_alpha_init=residual_alpha_init,
        )
        self._use_channel_first_train_inputs_ = True
        self.data_format = data_format

    def _build_conv_core(
        self,
        spatial_ndim: int,
        in_channels: int,
        output_dim: int,
        *,
        segmentation_head: bool,
        spatial_shape: Optional[Tuple[int, ...]] = None,
    ) -> nn.Module:
        return super()._build_conv_core(
            spatial_ndim,
            in_channels,
            output_dim,
            segmentation_head=segmentation_head,
            spatial_shape=spatial_shape,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
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
    ) -> "ResConvPSANNRegressor":
        if not hisso:
            return super().fit(
                X,
                y,
                validation_data=validation_data,
                verbose=verbose,
                noisy=noisy,
                hisso=False,
                hisso_window=hisso_window,
                hisso_batch_episodes=hisso_batch_episodes,
                hisso_updates_per_epoch=hisso_updates_per_epoch,
                hisso_reward_fn=hisso_reward_fn,
                hisso_context_extractor=hisso_context_extractor,
                hisso_primary_transform=hisso_primary_transform,
                hisso_transition_penalty=hisso_transition_penalty,
                hisso_trans_cost=hisso_trans_cost,
                hisso_supervised=hisso_supervised,
                lr_max=lr_max,
                lr_min=lr_min,
            )

        if self.per_element:
            raise ValueError("hisso=True currently supports per_element=False.")

        seed_all(self.random_state)

        fit_args = normalise_fit_args(
            self,
            X,
            y,
            validation_data=validation_data,
            noisy=noisy,
            verbose=verbose,
            lr_max=lr_max,
            lr_min=lr_min,
            hisso=True,
            hisso_kwargs={
                "hisso_window": hisso_window,
                "hisso_batch_episodes": hisso_batch_episodes,
                "hisso_updates_per_epoch": hisso_updates_per_epoch,
                "hisso_reward_fn": hisso_reward_fn,
                "hisso_context_extractor": hisso_context_extractor,
                "hisso_primary_transform": hisso_primary_transform,
                "hisso_transition_penalty": hisso_transition_penalty,
                "hisso_trans_cost": hisso_trans_cost,
                "hisso_supervised": hisso_supervised,
            },
        )

        verbose = fit_args.verbose
        self._keep_column_output_ = bool(fit_args.y is not None and fit_args.y.ndim > 1)

        prepared_state, primary_dim = prepare_inputs_and_scaler(self, fit_args)
        primary_dim = int(primary_dim)
        self._primary_dim_ = primary_dim
        self._output_dim_ = int(prepared_state.output_dim)
        self._train_inputs_layout_ = "cf"
        self._target_cf_shape_ = (
            tuple(prepared_state.y_cf.shape[1:])
            if prepared_state.y_cf is not None
            else self._target_cf_shape_
        )
        self._target_vector_dim_ = (
            int(prepared_state.y_vector.shape[1])
            if prepared_state.y_vector is not None
            else self._target_vector_dim_
        )

        lsm_data = prepared_state.train_inputs
        lsm_model, lsm_channels = self._resolve_lsm_module(lsm_data, preserve_shape=True)

        hooks = self._make_fit_hooks(prepared=prepared_state, verbose=verbose)

        request = ModelBuildRequest(
            estimator=self,
            prepared=prepared_state,
            primary_dim=primary_dim,
            lsm_module=lsm_model,
            lsm_output_dim=lsm_channels,
            preserve_shape=True,
        )

        rebuild = not (self.warm_start and isinstance(getattr(self, "model_", None), nn.Module))
        if rebuild:
            self.model_ = build_model_from_hooks(hooks, request)

        device = self._device()
        self._ensure_model_device(device)

        result = maybe_run_hisso(hooks, request, fit_args=fit_args)
        if result is None:
            raise RuntimeError("HISSO requested but no variant hook was provided.")
        return self


__all__ = ["ResPSANNRegressor", "ResConvPSANNRegressor"]
