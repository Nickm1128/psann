from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..attention import build_attention_module
from ..conv import PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet
from ..estimators._fit_utils import (
    FitVariantHooks,
    HISSOTrainingPlan,
    ModelBuildRequest,
    NormalisedFitArgs,
    PreparedInputState,
    build_hisso_training_plan,
    build_model_from_hooks,
    maybe_run_hisso,
    normalise_fit_args,
    prepare_inputs_and_scaler,
    run_supervised_training,
)
from ..estimators._fit_utils import _build_optimizer as _build_optimizer_helper
from ..nn import PSANNNet, WithPreprocessor
from ..preproc import build_preprocessor
from ..types import NoiseSpec
from ..utils import choose_device, seed_all
from .shared import ValidationDataLike, _AttentionConvModel, _AttentionDenseModel

if TYPE_CHECKING:
    from .base import PSANNRegressor
    from .residual import ResConvPSANNRegressor


class _PSANNRegressorBuilderMixin:
    def _device(self) -> torch.device:
        return choose_device(self.device)

    def _infer_input_shape(self, X: np.ndarray) -> tuple:
        if X.ndim < 2:
            raise ValueError("X must be at least 2D (batch, features...)")
        return tuple(X.shape[1:])

    def _flatten(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], -1).astype(np.float32, copy=False)

    def _resolve_lsm_module(
        self,
        data: Any,
        *,
        preserve_shape: bool,
    ) -> Tuple[Optional[nn.Module], Optional[int]]:
        if self.lsm is None:
            self._lsm_module_ = None
            return None, None

        preproc, module = build_preprocessor(
            self.lsm,
            allow_train=data is not None,
            pretrain_epochs=self.lsm_pretrain_epochs,
            data=data,
        )
        if preproc is None:
            self._lsm_module_ = None
            return None, None

        self.lsm = preproc
        lsm_module = (
            module if module is not None else (preproc if isinstance(preproc, nn.Module) else None)
        )
        if lsm_module is None or not hasattr(lsm_module, "forward"):
            raise RuntimeError(
                "Provided lsm preprocessor must expose a torch.nn.Module. Fit the expander or pass an nn.Module."
            )

        self._lsm_module_ = lsm_module
        if lsm_module is not None and preproc is not None:
            if hasattr(preproc, "score_reconstruction") and not hasattr(
                lsm_module, "score_reconstruction"
            ):
                setattr(lsm_module, "score_reconstruction", preproc.score_reconstruction)
        attr = "out_channels" if preserve_shape else "output_dim"
        dim = getattr(lsm_module, attr, getattr(preproc, attr, None))
        return lsm_module, int(dim) if dim is not None else None

    def _build_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> nn.Module:
        if self._attention_enabled():
            return self._build_attention_dense_core(
                input_dim,
                output_dim,
                state_cfg=state_cfg,
                input_shape=input_shape,
            )
        return self._build_dense_backbone(
            input_dim,
            output_dim,
            state_cfg=state_cfg,
        )

    def _build_dense_backbone(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        return PSANNNet(
            int(input_dim),
            int(output_dim),
            hidden_layers=self.hidden_layers,
            hidden_units=self.hidden_units,
            hidden_width=self.hidden_width,
            act_kw=self.activation,
            state_cfg=state_cfg,
            activation_type=self.activation_type,
            w0=self.w0,
        )

    def _build_token_backbone(
        self,
        token_dim: int,
        embed_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        return self._build_dense_backbone(
            token_dim,
            embed_dim,
            state_cfg=state_cfg,
        )

    def _build_attention_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[Dict[str, Any]],
        input_shape: Optional[Tuple[int, ...]],
    ) -> nn.Module:
        if input_shape is None or len(input_shape) < 2:
            raise ValueError(
                "attention requires inputs with shape (batch, ..., features); "
                "provide tensors with at least two non-batch dimensions."
            )
        seq_dims = input_shape[:-1]
        token_dim = int(input_shape[-1])
        seq_len = int(math.prod(seq_dims)) if seq_dims else 1
        expected = seq_len * token_dim
        if expected != int(input_dim):
            raise ValueError(
                "attention expected input_dim matching seq_len * token_dim "
                f"(inferred={expected}, received={input_dim}). "
                "Ensure the last axis of X holds per-token features."
            )
        attn_module = build_attention_module(self.attention, int(self.hidden_units))
        if attn_module is None:
            return self._build_dense_backbone(
                input_dim,
                output_dim,
                state_cfg=state_cfg,
            )
        token_backbone = self._build_token_backbone(
            token_dim,
            int(self.hidden_units),
            state_cfg=state_cfg,
        )
        self._attention_shape_ = (seq_len, token_dim)
        return _AttentionDenseModel(
            token_backbone,
            attn_module,
            seq_len=seq_len,
            token_dim=token_dim,
            embed_dim=int(self.hidden_units),
            output_dim=int(output_dim),
            pool="mean",
        )

    def _infer_conv_embed_dim(self, core: nn.Module) -> int:
        for attr in ("conv_channels", "hidden_channels"):
            if hasattr(core, attr):
                return int(getattr(core, attr))
        raise ValueError("Unable to infer convolutional token dimension for attention.")

    def _wrap_with_attention_conv(
        self,
        core: nn.Module,
        spatial_shape: Optional[Tuple[int, ...]],
        *,
        segmentation_head: bool,
    ) -> nn.Module:
        if spatial_shape is None:
            raise ValueError(
                "attention requires known spatial dimensions for preserve_shape inputs; "
                "ensure training inputs include spatial axes."
            )
        embed_dim = self._infer_conv_embed_dim(core)
        attn_module = build_attention_module(self.attention, embed_dim)
        if attn_module is None:
            return core
        seq_len = int(math.prod(spatial_shape)) if spatial_shape else 1
        self._attention_shape_ = (seq_len, embed_dim)
        return _AttentionConvModel(
            core,
            attn_module,
            spatial_shape=spatial_shape,
            segmentation_head=segmentation_head,
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
        core = conv_cls(
            int(in_channels),
            int(output_dim),
            hidden_layers=self.hidden_layers,
            conv_channels=self.conv_channels,
            hidden_channels=self.conv_channels,
            kernel_size=self.conv_kernel_size,
            act_kw=self.activation,
            activation_type=self.activation_type,
            w0=self.w0,
            segmentation_head=bool(segmentation_head),
        )
        if self._attention_enabled():
            return self._wrap_with_attention_conv(
                core, spatial_shape, segmentation_head=segmentation_head
            )
        return core

    def _make_optimizer(self, model: torch.nn.Module, lr: Optional[float] = None):
        lr = float(self.lr if lr is None else lr)
        if self.optimizer.lower() == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        if self.optimizer.lower() == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """Compatibility helper for warm-start flows expecting estimator-owned builders."""

        return _build_optimizer_helper(self, model)

    def _make_loss(self):
        # Built-in strings
        if isinstance(self.loss, str):
            name = self.loss.lower()
            params = self.loss_params or {}
            reduction = self.loss_reduction
            if name in ("l1", "mae"):
                return torch.nn.L1Loss(reduction=reduction)
            if name in ("mse", "l2"):
                return torch.nn.MSELoss(reduction=reduction)
            if name in ("smooth_l1", "huber_smooth"):
                beta = float(params.get("beta", 1.0))
                return torch.nn.SmoothL1Loss(beta=beta, reduction=reduction)
            if name in ("huber",):
                delta = float(params.get("delta", 1.0))
                return torch.nn.HuberLoss(delta=delta, reduction=reduction)
            raise ValueError(
                f"Unknown loss '{self.loss}'. Supported: mse, l1/mae, smooth_l1, huber, or a callable."
            )

        # Callable custom loss; may return tensor (any shape) or float
        if callable(self.loss):
            user_fn = self.loss
            params = self.loss_params or {}
            reduction = self.loss_reduction

            def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                out = user_fn(pred, target, **params) if params else user_fn(pred, target)
                if not isinstance(out, torch.Tensor):
                    out = torch.as_tensor(out, dtype=pred.dtype, device=pred.device)
                if out.ndim == 0:
                    return out
                if reduction == "mean":
                    return out.mean()
                if reduction == "sum":
                    return out.sum()
                if reduction == "none":
                    return out
                raise ValueError(f"Unsupported reduction '{reduction}' for custom loss")

            return _loss

        raise TypeError("loss must be a string or a callable returning a scalar tensor")

    def _make_per_element_fit_hooks(self) -> FitVariantHooks:
        def build_model(request: ModelBuildRequest) -> nn.Module:
            prepared = request.prepared
            X_cf = prepared.train_inputs
            if not isinstance(X_cf, np.ndarray):
                X_cf = prepared.X_cf
            if X_cf is None:
                raise ValueError(
                    "PreparedInputState missing channel-first inputs for per-element training."
                )
            nd = int(X_cf.ndim) - 2
            if nd < 1:
                raise ValueError("per_element=True expects inputs with spatial dimensions.")

            in_channels = int(X_cf.shape[1])
            spatial_shape = tuple(X_cf.shape[2:])
            lsm_module = request.lsm_module
            if lsm_module is not None:
                if request.lsm_output_dim is not None:
                    in_channels = int(request.lsm_output_dim)
                elif hasattr(lsm_module, "out_channels"):
                    in_channels = int(getattr(lsm_module, "out_channels"))
                elif hasattr(self.lsm, "out_channels"):
                    in_channels = int(getattr(self.lsm, "out_channels"))

            core = self._build_conv_core(
                nd,
                in_channels,
                int(prepared.output_dim),
                segmentation_head=True,
                spatial_shape=spatial_shape,
            )

            preproc = lsm_module
            if preproc is not None and not self.lsm_train:
                for param in preproc.parameters():
                    param.requires_grad = False
            return WithPreprocessor(preproc, core)

        return FitVariantHooks(build_model=build_model)

    def _make_conv_fit_hooks(
        self,
        *,
        prepared: PreparedInputState,
        verbose: int,
    ) -> FitVariantHooks:
        internal_shape = prepared.internal_shape_cf
        if internal_shape is None:
            raise ValueError("PreparedInputState missing channels-first shape for conv training.")
        spatial_ndim = max(1, len(internal_shape) - 1)
        base_channels = int(internal_shape[0])
        spatial_shape = tuple(internal_shape[1:])

        def build_model(request: ModelBuildRequest) -> nn.Module:
            lsm_module = request.lsm_module
            in_channels = base_channels
            if lsm_module is not None:
                if request.lsm_output_dim is not None:
                    in_channels = int(request.lsm_output_dim)
                elif hasattr(lsm_module, "out_channels"):
                    in_channels = int(getattr(lsm_module, "out_channels"))
                elif hasattr(self.lsm, "out_channels"):
                    in_channels = int(getattr(self.lsm, "out_channels"))

            core = self._build_conv_core(
                spatial_ndim,
                in_channels,
                int(prepared.output_dim),
                segmentation_head=bool(self.per_element),
                spatial_shape=spatial_shape,
            )

            preproc = lsm_module
            if preproc is not None and not self.lsm_train:
                for param in preproc.parameters():
                    param.requires_grad = False
            return WithPreprocessor(preproc, core)

        def build_hisso_plan(
            estimator_ref: "ResConvPSANNRegressor",
            request: ModelBuildRequest,
            *,
            fit_args: NormalisedFitArgs,
        ) -> Optional[HISSOTrainingPlan]:
            if estimator_ref.per_element:
                raise ValueError(
                    "hisso=True currently supports per_element=False for ResConvPSANNRegressor"
                )
            prepared_local = request.prepared
            inputs_cf = prepared_local.train_inputs
            if not isinstance(inputs_cf, np.ndarray):
                inputs_cf = prepared_local.X_cf
            if inputs_cf is None:
                raise ValueError(
                    "PreparedInputState missing channel-first inputs for conv HISSO training."
                )
            if fit_args.hisso_options is None:
                raise ValueError("HISSO options were not prepared despite hisso=True.")
            return build_hisso_training_plan(
                estimator_ref,
                train_inputs=inputs_cf,
                primary_dim=int(request.primary_dim),
                fit_args=fit_args,
                options=fit_args.hisso_options,
                lsm_module=request.lsm_module,
            )

        return FitVariantHooks(
            build_model=build_model,
            build_hisso_plan=build_hisso_plan,
        )

    def _make_flatten_fit_hooks(
        self,
        *,
        prepared: PreparedInputState,
        verbose: int,
    ) -> FitVariantHooks:
        train_inputs = prepared.train_inputs
        if train_inputs is None:
            raise ValueError("PreparedInputState missing flattened training inputs.")
        if not isinstance(train_inputs, np.ndarray):
            raise ValueError(
                "PreparedInputState.train_inputs must be a numpy array for flat training."
            )

        def build_model(request: ModelBuildRequest) -> nn.Module:
            prepared_local = request.prepared
            inputs_arr = prepared_local.train_inputs
            if inputs_arr is None or not isinstance(inputs_arr, np.ndarray):
                raise ValueError("PreparedInputState missing train_inputs for model construction.")
            if request.lsm_output_dim is not None:
                input_dim = int(request.lsm_output_dim)
            else:
                input_dim = int(inputs_arr.shape[1])
            if self._attention_enabled() and request.lsm_module is not None:
                raise ValueError(
                    "attention is currently incompatible with lsm preprocessors; "
                    "attach attention after the LSM or disable lsm."
                )
            core = self._build_dense_core(
                input_dim,
                int(prepared_local.output_dim),
                state_cfg=(self.state if self.stateful else None),
                input_shape=prepared_local.input_shape,
            )
            preproc = request.lsm_module
            if preproc is not None and not self.lsm_train:
                for param in preproc.parameters():
                    param.requires_grad = False
            if preproc is None:
                return core
            return WithPreprocessor(preproc, core)

        def build_hisso_plan(
            estimator_ref: "PSANNRegressor",
            request: ModelBuildRequest,
            *,
            fit_args: NormalisedFitArgs,
        ) -> Optional[HISSOTrainingPlan]:
            if not fit_args.hisso:
                return None
            if fit_args.hisso_options is None:
                raise ValueError("HISSO options were not prepared despite hisso=True.")
            inputs_arr = request.prepared.train_inputs
            if inputs_arr is None or not isinstance(inputs_arr, np.ndarray):
                inputs_arr = request.prepared.X_flat
                if inputs_arr is None:
                    raise ValueError("PreparedInputState missing inputs for HISSO planning.")
            return build_hisso_training_plan(
                estimator_ref,
                train_inputs=inputs_arr,
                primary_dim=int(request.primary_dim),
                fit_args=fit_args,
                options=fit_args.hisso_options,
                lsm_module=request.lsm_module,
            )

        return FitVariantHooks(
            build_model=build_model,
            build_hisso_plan=build_hisso_plan,
        )

    def _make_fit_hooks(
        self,
        *,
        prepared: PreparedInputState,
        verbose: int,
    ) -> FitVariantHooks:
        if self.preserve_shape and self.per_element:
            return self._make_per_element_fit_hooks()
        if self.preserve_shape and getattr(self, "_use_channel_first_train_inputs_", False):
            return self._make_conv_fit_hooks(prepared=prepared, verbose=verbose)
        return self._make_flatten_fit_hooks(prepared=prepared, verbose=verbose)

    # Estimator API

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
    ):
        """Fit the estimator.

        Parameters
        - X: np.ndarray
            Training inputs. Shapes:
              - MLP/flattened: (N, F1, ..., Fk) flattened internally to (N, prod(F*))
              - preserve_shape=True: (N, C, ...) or (N, ..., C) depending on data_format
        - y: np.ndarray
            Targets. Shapes:
              - vector/pooled head: (N, T) or (N,) where T=prod(output_shape) if provided
              - per_element=True: (N, C_out, ...) or (N, ..., C_out) matching X spatial dims
        - validation_data: optional (X_val, y_val) tuple for early stopping/logging
        - verbose: 0/1 to control epoch logging
        - noisy: optional Gaussian input noise std; scalar, per-feature vector, or tensor matching input shape
        - hisso_supervised: optional bool or dict to run a supervised warm start before HISSO (requires providing 'y')
        - hisso: if True, train via Horizon-Informed Sampling Strategy Optimization (episodic reward)
        - hisso_window: episode/window length for HISSO (default 64)
        - hisso_batch_episodes: number of episodes sampled per HISSO optimizer update (default 32)
        - hisso_updates_per_epoch: number of HISSO optimizer updates per epoch (defaults to compatibility schedule when omitted)
        - hisso_primary_transform: optional transform ('identity' | 'softmax' | 'tanh') applied to primary outputs before reward evaluation
        - hisso_transition_penalty: optional float penalty applied between HISSO steps (deprecated alias `hisso_trans_cost` still accepted)
        """
        seed_all(self.random_state)

        if hisso and self.per_element:
            raise ValueError("hisso=True currently supports per_element=False.")

        # Reset HISSO-specific runtime artefacts before launching a new fit.
        self._hisso_options_ = None
        self._hisso_reward_fn_ = None
        self._hisso_context_extractor_ = None
        self._hisso_cfg_ = None
        self._hisso_trainer_ = None
        self._hisso_trained_ = False

        if not self.warm_start:
            self._scaler_state_ = None
            self._scaler_spec_ = None
            self._scaler_kind_ = None
            self._scaler_fitted_ = False
            self._target_scaler_state_ = None
            self._target_scaler_spec_ = None
            self._target_scaler_kind_ = None
            self._target_scaler_fitted_ = False

        fit_args = normalise_fit_args(
            self,
            X,
            y,
            context=context,
            validation_data=validation_data,
            noisy=noisy,
            verbose=verbose,
            lr_max=lr_max,
            lr_min=lr_min,
            hisso=hisso,
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
        hisso = fit_args.hisso

        X = fit_args.X
        y = fit_args.y
        self._keep_column_output_ = bool(y is not None and y.ndim > 1)

        prepared_state, primary_dim = prepare_inputs_and_scaler(
            self,
            fit_args,
        )
        primary_dim = int(primary_dim)
        self._primary_dim_ = primary_dim
        self._output_dim_ = int(prepared_state.output_dim)
        layout = (
            "cf"
            if (
                self.preserve_shape
                and (self.per_element or getattr(self, "_use_channel_first_train_inputs_", False))
            )
            else "flat"
        )
        self._train_inputs_layout_ = layout
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
        self._context_dim_ = prepared_state.context_dim
        if prepared_state.context_dim is not None and hasattr(self, "context_dim"):
            try:
                setattr(self, "context_dim", int(prepared_state.context_dim))
            except Exception:
                pass

        preserve_inputs = bool(self.preserve_shape and self.per_element)
        lsm_data = prepared_state.train_inputs
        lsm_model, lsm_dim = self._resolve_lsm_module(lsm_data, preserve_shape=preserve_inputs)

        hooks = self._make_fit_hooks(prepared=prepared_state, verbose=verbose)

        request = ModelBuildRequest(
            estimator=self,
            prepared=prepared_state,
            primary_dim=primary_dim,
            lsm_module=lsm_model,
            lsm_output_dim=lsm_dim,
            preserve_shape=bool(self.preserve_shape),
        )

        rebuild = not (self.warm_start and isinstance(getattr(self, "model_", None), nn.Module))
        if rebuild:
            self.model_ = build_model_from_hooks(hooks, request)
            self._model_device_ = None
        self._model_rebuilt_ = bool(rebuild)

        device = self._device()
        self._ensure_model_device(device)
        self._after_model_built()

        if hisso:
            result = maybe_run_hisso(hooks, request, fit_args=fit_args)
            if result is None:
                raise RuntimeError("HISSO requested but no variant hook was provided.")
            return self

        run_supervised_training(
            self,
            self.model_,
            prepared_state,
            fit_args=fit_args,
        )
        return self


__all__ = ["_PSANNRegressorBuilderMixin"]
