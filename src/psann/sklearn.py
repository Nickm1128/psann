from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Callable, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn

try:  # Optional scikit-learn import for API compatibility
    from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
    from sklearn.metrics import r2_score as _sk_r2_score  # type: ignore
except Exception:  # Fallbacks if sklearn isn't installed at runtime

    class BaseEstimator:  # minimal stub
        def get_params(self, deep: bool = True):
            # Return non-private, non-callable attributes
            params = {}
            for k, v in self.__dict__.items():
                if k.endswith("_"):
                    continue
                if not k.startswith("_") and not callable(v):
                    params[k] = v
            return params

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class RegressorMixin:
        pass

    def _sk_r2_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1.0 - (u / v if v != 0 else np.nan)


from ._aliases import resolve_int_alias
from .conv import PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet, ResidualPSANNConv2dNet
from .models import WaveResNet
from .nn import PSANNNet, ResidualPSANNNet, WithPreprocessor
from .preproc import PreprocessorLike, build_preprocessor
from .state import StateConfig, ensure_state_config
from .types import ActivationConfig, LossLike, NoiseSpec, ScalerSpec
from .utils import choose_device, seed_all

ValidationDataLike = Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]
from .estimators._fit_utils import (
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
from .estimators._fit_utils import (
    _build_optimizer as _build_optimizer_helper,
)


class PSANNRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-style regressor wrapper around a PSANN network (PyTorch).

    Parameters mirror the README's proposed API.
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
        self.context_builder = context_builder
        self.context_builder_params = context_builder_params or {}
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

    @staticmethod
    def _normalize_param_aliases(params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        hidden_width_raw = out.pop("hidden_width", None)
        hidden_units_res = resolve_int_alias(
            primary_value=out.pop("hidden_units", None),
            alias_value=hidden_width_raw,
            primary_name="hidden_units",
            alias_name="hidden_width",
            context="PSANNRegressor.set_params",
        )
        if hidden_units_res.value is not None:
            out["hidden_units"] = hidden_units_res.value
            out["hidden_width"] = hidden_units_res.value
        elif hidden_width_raw is not None:
            out["hidden_width"] = hidden_width_raw

        conv_channels_res = resolve_int_alias(
            primary_value=out.pop("conv_channels", None),
            alias_value=out.pop("hidden_channels", None),
            primary_name="conv_channels",
            alias_name="hidden_channels",
            context="PSANNRegressor.set_params",
            default=out.get("hidden_units"),
        )
        if conv_channels_res.value is not None:
            out["conv_channels"] = conv_channels_res.value
        else:
            out.pop("conv_channels", None)
        return out

    def _get_context_builder(self) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        builder = getattr(self, "_context_builder_callable_", None)
        if builder is not None:
            return builder
        spec = self.context_builder
        if spec is None:
            return None
        if callable(spec):
            builder = spec
        elif isinstance(spec, str):
            key = spec.strip().lower()
            if key == "cosine":
                builder = self._build_cosine_context_callable(**self.context_builder_params)
            else:
                raise ValueError(f"Unknown context_builder option: {spec!r}")
        else:
            raise TypeError("context_builder must be None, a string, or a callable.")
        self._context_builder_callable_ = builder
        return builder

    @staticmethod
    def _build_cosine_context_callable(
        *,
        frequencies: Optional[Union[int, Iterable[float]]] = None,
        include_sin: bool = True,
        include_cos: bool = True,
        normalise_input: bool = False,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if not include_sin and not include_cos:
            raise ValueError("cosine context builder requires include_sin or include_cos.")
        if frequencies is None:
            freqs: list[float] = [1.0]
        elif isinstance(frequencies, int):
            if frequencies <= 0:
                raise ValueError("frequencies integer must be positive.")
            freqs = [float(idx) for idx in range(1, frequencies + 1)]
        else:
            freqs = [float(freq) for freq in frequencies]
            if not freqs:
                raise ValueError("frequencies iterable must contain at least one value.")

        def _builder(inputs: np.ndarray) -> np.ndarray:
            arr = np.asarray(inputs, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            flat = arr.reshape(arr.shape[0], -1)
            basis = flat
            if normalise_input:
                norms = np.linalg.norm(flat, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-6)
                basis = flat / norms
            features: list[np.ndarray] = []
            for freq in freqs:
                scaled = basis * float(freq)
                if include_sin:
                    features.append(np.sin(scaled))
                if include_cos:
                    features.append(np.cos(scaled))
            if not features:
                raise RuntimeError("cosine context builder produced no features.")
            return np.concatenate(features, axis=1).astype(np.float32, copy=False)

        return _builder

    def _auto_context(self, features_2d: np.ndarray) -> Optional[np.ndarray]:
        builder = self._get_context_builder()
        if builder is None:
            return None
        context = builder(features_2d)
        context_arr = np.asarray(context, dtype=np.float32)
        if context_arr.ndim == 1:
            context_arr = context_arr.reshape(-1, 1)
        if context_arr.shape[0] != features_2d.shape[0]:
            raise ValueError(
                "Context builder must preserve the number of samples along the first dimension."
            )
        return context_arr

    # ------------------------- Scaling helpers -------------------------
    def _make_internal_scaler(self) -> Optional[Dict[str, Any]]:
        kind = self.scaler
        if kind is None:
            return None
        if isinstance(kind, str):
            key = kind.lower()
            if key not in {"standard", "minmax"}:
                raise ValueError(
                    "Unsupported scaler string. Use 'standard', 'minmax', or provide an object with fit/transform."
                )
            return {"type": key, "state": {}}
        # Custom object: must implement fit/transform; inverse_transform optional
        if not hasattr(kind, "fit") or not hasattr(kind, "transform"):
            raise ValueError(
                "Custom scaler must implement fit(X) and transform(X). Optional inverse_transform(X)."
            )
        return {"type": "custom", "obj": kind}

    def _scaler_fit_update(self, X2d: np.ndarray) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Fit or update scaler on 2D array and return a transform function.

        - For built-in scalers, supports incremental update when warm_start=True.
        - For custom object, calls .fit on first time, else attempts partial_fit if available, else refit on concat.
        """
        if self.scaler is None:
            self._scaler_kind_ = None
            self._scaler_state_ = None
            return None
        spec = getattr(self, "_scaler_spec_", None)
        if spec is None:
            spec = self._make_internal_scaler()
            self._scaler_spec_ = spec
        if spec is None:
            self._scaler_kind_ = None
            self._scaler_state_ = None
            return None

        if spec.get("type") == "standard":
            self._scaler_kind_ = "standard"
            st = self._scaler_state_ or {"n": 0, "mean": None, "M2": None}
            n0 = int(st["n"])
            X = np.asarray(X2d, dtype=np.float32)
            # Welford online update per feature
            if n0 == 0:
                mean = X.mean(axis=0)
                diff = X - mean
                M2 = (diff * diff).sum(axis=0)
                n = X.shape[0]
            else:
                mean0 = st["mean"]
                M20 = st["M2"]
                n = n0 + X.shape[0]
                delta = X.mean(axis=0) - mean0
                mean = (mean0 * n0 + X.sum(axis=0)) / n
                # Update M2 across batches: combine variances
                # M2_total = M2_a + M2_b + delta^2 * n_a * n_b / n_total
                M2a = M20
                xa = n0
                xb = X.shape[0]
                X_centered = X - X.mean(axis=0)
                M2b = (X_centered * X_centered).sum(axis=0)
                M2 = M2a + M2b + (delta * delta) * (xa * xb) / max(n, 1)
            self._scaler_state_ = {"n": int(n), "mean": mean, "M2": M2}

            def _xfm(Z: np.ndarray) -> np.ndarray:
                st2 = self._scaler_state_
                assert st2 is not None
                mean2 = st2["mean"]
                var = st2["M2"] / max(st2["n"], 1)
                std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
                return (Z - mean2) / std

            return _xfm

        if spec.get("type") == "minmax":
            self._scaler_kind_ = "minmax"
            st = self._scaler_state_ or {"min": None, "max": None}
            X = np.asarray(X2d, dtype=np.float32)
            mn = X.min(axis=0) if st["min"] is None else np.minimum(st["min"], X.min(axis=0))
            mx = X.max(axis=0) if st["max"] is None else np.maximum(st["max"], X.max(axis=0))
            self._scaler_state_ = {"min": mn, "max": mx}

            def _xfm(Z: np.ndarray) -> np.ndarray:
                st2 = self._scaler_state_
                assert st2 is not None
                mn2 = st2["min"]
                mx2 = st2["max"]
                scale = np.where((mx2 - mn2) > 1e-8, (mx2 - mn2), 1.0)
                return (Z - mn2) / scale

            return _xfm

        # Custom object
        obj = spec.get("obj")
        self._scaler_kind_ = "custom"
        if not hasattr(self, "_scaler_fitted_") or not getattr(self, "_scaler_fitted_", False):
            # Fit once
            try:
                obj.fit(X2d, **(self.scaler_params or {}))
            except TypeError:
                obj.fit(X2d)
            self._scaler_fitted_ = True
        else:
            if hasattr(obj, "partial_fit"):
                obj.partial_fit(X2d)
            else:
                # Fallback: refit on concatenation of small cache if available
                pass

        def _xfm(Z: np.ndarray) -> np.ndarray:
            return obj.transform(Z)

        return _xfm

    def _scaler_inverse_tensor(self, X_ep: torch.Tensor, *, feature_dim: int = -1) -> torch.Tensor:
        """Inverse-transform a torch tensor episode if scaler is active.

        Expects features along last dim by default (B,T,D) or (N,D).
        """
        kind = getattr(self, "_scaler_kind_", None)
        st = getattr(self, "_scaler_state_", None)
        if kind is None:
            return X_ep
        if kind == "standard" and st is not None:
            mean = torch.as_tensor(st["mean"], device=X_ep.device, dtype=X_ep.dtype)
            var = torch.as_tensor(st["M2"] / max(st["n"], 1), device=X_ep.device, dtype=X_ep.dtype)
            std = torch.sqrt(torch.clamp(var, min=1e-8))
            return X_ep * std + mean
        if kind == "minmax" and st is not None:
            mn = torch.as_tensor(st["min"], device=X_ep.device, dtype=X_ep.dtype)
            mx = torch.as_tensor(st["max"], device=X_ep.device, dtype=X_ep.dtype)
            scale = torch.where((mx - mn) > 1e-8, (mx - mn), torch.ones_like(mx))
            return X_ep * scale + mn
        if kind == "custom" and hasattr(self.scaler, "inverse_transform"):
            # Fallback via CPU numpy; small overhead acceptable for context extraction
            X_np = X_ep.detach().cpu().numpy()
            X_inv = self.scaler.inverse_transform(X_np)
            return torch.as_tensor(X_inv, device=X_ep.device, dtype=X_ep.dtype)
        return X_ep

    def _apply_fitted_scaler(self, X2d: np.ndarray) -> np.ndarray:
        kind = getattr(self, "_scaler_kind_", None)
        if kind is None:
            return X2d.astype(np.float32, copy=False)
        st = getattr(self, "_scaler_state_", None)
        if kind == "standard" and st is not None:
            n = max(int(st.get("n", 0)), 1)
            mean = np.asarray(st.get("mean"), dtype=np.float32)
            var = np.asarray(st.get("M2"), dtype=np.float32) / float(n)
            std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32, copy=False)
            return ((X2d - mean) / std).astype(np.float32, copy=False)
        if kind == "minmax" and st is not None:
            mn = np.asarray(st.get("min"), dtype=np.float32)
            mx = np.asarray(st.get("max"), dtype=np.float32)
            scale = np.where((mx - mn) > 1e-8, (mx - mn), 1.0).astype(np.float32, copy=False)
            return ((X2d - mn) / scale).astype(np.float32, copy=False)
        if kind == "custom" and hasattr(self, "scaler") and hasattr(self.scaler, "transform"):
            transformed = self.scaler.transform(X2d)
            return np.asarray(transformed, dtype=np.float32)
        return X2d.astype(np.float32, copy=False)

    def _ensure_fitted(self) -> None:
        model = getattr(self, "model_", None)
        if model is None:
            raise RuntimeError("Estimator is not fitted yet; call 'fit' before inference.")
        if getattr(self, "input_shape_", None) is None:
            raise RuntimeError("Estimator is missing fitted input shape; inference is unavailable.")

    def _prepare_inference_inputs(
        self,
        X: np.ndarray,
        context: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
        self._ensure_fitted()
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == len(self.input_shape_):
            X_arr = X_arr.reshape((1,) + tuple(self.input_shape_))
        if X_arr.ndim != len(self.input_shape_) + 1:
            raise ValueError(
                f"Expected input with {len(self.input_shape_) + 1} dimensions; received shape {X_arr.shape}."
            )
        expected = tuple(self.input_shape_)
        if tuple(X_arr.shape[1:]) != expected:
            raise ValueError(
                f"Input shape {X_arr.shape[1:]} does not match fitted shape {expected}."
            )

        meta: Dict[str, Any] = {
            "n_samples": int(X_arr.shape[0]),
            "data_format": self.data_format,
            "per_element": bool(self.per_element),
        }
        context_np: Optional[np.ndarray] = None
        if context is not None:
            context_np = np.asarray(context, dtype=np.float32)
            if context_np.ndim == 1:
                context_np = context_np.reshape(-1, 1)
            if context_np.shape[0] != meta["n_samples"]:
                raise ValueError(
                    f"context has {context_np.shape[0]} samples but X has {meta['n_samples']}."
                )
        flat_for_context: Optional[np.ndarray] = None

        if not self.preserve_shape:
            X2d = self._flatten(X_arr)
            X2d = self._apply_fitted_scaler(X2d)
            meta["layout"] = "flat"
            flat_for_context = X2d
            inputs_np = X2d
        else:
            X_cf = np.moveaxis(X_arr, -1, 1) if self.data_format == "channels_last" else X_arr.copy()
            meta["cf_shape"] = tuple(X_cf.shape[1:])
            N, C = X_cf.shape[0], int(X_cf.shape[1])
            X2d = X_cf.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
            X2d_scaled = self._apply_fitted_scaler(X2d)
            flat_for_context = X2d_scaled if X2d_scaled is not X2d else X2d
            if X2d_scaled is not X2d:
                X_cf = X2d_scaled.reshape(N, -1, C).transpose(0, 2, 1).reshape(X_cf.shape)
            use_cf_inputs = bool(
                self.per_element or getattr(self, "_use_channel_first_train_inputs_", False)
            )
            meta["layout"] = "cf" if use_cf_inputs else "flat"
            if use_cf_inputs:
                inputs_np = X_cf.astype(np.float32, copy=False)
            else:
                inputs_np = flat_for_context.reshape(N, -1).astype(np.float32, copy=False)

        if context_np is None and flat_for_context is not None:
            auto_ctx = self._auto_context(flat_for_context.astype(np.float32, copy=False))
            if auto_ctx is not None:
                context_np = auto_ctx

        if context_np is not None:
            if context_np.shape[0] != meta["n_samples"]:
                raise ValueError(
                    f"Context builder returned {context_np.shape[0]} samples but inputs have {meta['n_samples']}."
                )
            if self._context_dim_ is None:
                self._context_dim_ = int(context_np.shape[1])
                if hasattr(self, "context_dim"):
                    try:
                        setattr(self, "context_dim", int(self._context_dim_))
                    except Exception:
                        pass
            if self._context_dim_ is not None and self._context_dim_ not in (0, context_np.shape[1]):
                raise ValueError(
                    f"Expected context feature dimension {self._context_dim_}; received {context_np.shape[1]}."
                )
        elif self._context_dim_ not in (None, 0):
            raise ValueError(
                f"This estimator was fit expecting context_dim={self._context_dim_}; provide a matching context array."
            )

        return inputs_np, meta, context_np

    def _reshape_predictions(self, preds: np.ndarray, meta: Dict[str, Any]) -> np.ndarray:
        n_samples = int(meta["n_samples"])
        preds = preds.astype(np.float32, copy=False)

        if self.preserve_shape and self.per_element:
            if self._target_cf_shape_ is not None:
                cf_shape = (n_samples,) + tuple(self._target_cf_shape_)
            elif getattr(self, "_internal_input_shape_cf_", None) is not None:
                spatial = tuple(self._internal_input_shape_cf_[1:])
                channels = preds.shape[1]
                cf_shape = (n_samples, channels) + spatial
            else:
                cf_shape = (n_samples, preds.shape[1])
            preds_cf = preds.reshape(cf_shape)
            if self.data_format == "channels_last" and preds_cf.ndim >= 3:
                preds_cf = np.moveaxis(preds_cf, 1, -1)
            return preds_cf

        if self._output_shape_tuple_ is not None:
            return preds.reshape((n_samples,) + self._output_shape_tuple_)

        if not self._keep_column_output_ and preds.shape[1] == 1:
            return preds.reshape(n_samples)

        return preds

    def _run_model(
        self,
        inputs_np: np.ndarray,
        *,
        context_np: Optional[np.ndarray] = None,
        state_updates: bool = False,
    ) -> np.ndarray:
        self._ensure_fitted()
        state_updates = bool(state_updates and self.stateful)
        model = self.model_
        if model is None:
            raise RuntimeError("Estimator is not fitted yet; no model available.")

        device = self._device()
        prev_training = model.training
        try:
            if state_updates:
                model.train(True)
                if hasattr(model, "set_state_updates"):
                    model.set_state_updates(True)
            else:
                model.eval()
                if hasattr(model, "set_state_updates"):
                    model.set_state_updates(False)

            model.to(device)
            with torch.no_grad():
                tensor = torch.from_numpy(inputs_np.astype(np.float32, copy=False)).to(device)
                context_tensor = None
                if context_np is not None:
                    context_tensor = torch.from_numpy(
                        context_np.astype(np.float32, copy=False)
                    ).to(device)
                outputs = model(tensor, context_tensor) if context_tensor is not None else model(tensor)
                return outputs.detach().cpu().numpy()
        finally:
            model.train(prev_training)
            if hasattr(model, "set_state_updates"):
                model.set_state_updates(bool(prev_training))

    # Internal helpers
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

    def _build_conv_core(
        self,
        spatial_ndim: int,
        in_channels: int,
        output_dim: int,
        *,
        segmentation_head: bool,
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
        return conv_cls(
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
            core = self._build_dense_core(
                input_dim,
                int(prepared_local.output_dim),
                state_cfg=(self.state if self.stateful else None),
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
        self._model_rebuilt_ = bool(rebuild)

        device = self._device()
        self.model_.to(device)
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

    def predict(self, X: np.ndarray, *, context: Optional[np.ndarray] = None) -> np.ndarray:
        inputs_np, meta, context_np = self._prepare_inference_inputs(X, context)
        preds = self._run_model(inputs_np, context_np=context_np, state_updates=False)
        return self._reshape_predictions(preds, meta)

    def score(self, X: np.ndarray, y: np.ndarray, *, context: Optional[np.ndarray] = None) -> float:
        y_true = np.asarray(y, dtype=np.float32)
        y_pred = self.predict(X, context=context)
        if y_true.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)
        elif y_true.shape != y_pred.shape:
            y_pred = y_pred.reshape(y_true.shape)
        return float(_sk_r2_score(y_true, y_pred))

    def reset_state(self) -> None:
        self._ensure_fitted()
        if hasattr(self.model_, "reset_state"):
            self.model_.reset_state()

    def step(
        self,
        x: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
        target: Optional[np.ndarray] = None,
        update_params: bool = False,
        update_state: bool = True,
    ) -> Any:
        batch = np.asarray(x, dtype=np.float32)
        if batch.ndim == len(self.input_shape_):
            batch = batch.reshape((1,) + tuple(self.input_shape_))
        elif batch.ndim == len(self.input_shape_) + 1 and batch.shape[0] == 1:
            batch = batch.reshape((1,) + tuple(self.input_shape_))
        elif batch.ndim != len(self.input_shape_) + 1:
            raise ValueError(
                f"Expected input with {len(self.input_shape_) + 1} dims; received shape {batch.shape}."
            )

        inputs_np, meta, context_np = self._prepare_inference_inputs(batch, context)
        preds = self._run_model(inputs_np, context_np=context_np, state_updates=bool(update_state))
        reshaped = self._reshape_predictions(preds, meta)
        if update_params:
            if target is None:
                raise ValueError("step(..., update_params=True) requires a target array.")
            self._apply_stream_update(inputs_np, context_np=context_np, target=target)
        if isinstance(reshaped, np.ndarray):
            if reshaped.shape[0] == 1:
                return reshaped[0]
            if reshaped.ndim == 0:
                return float(reshaped)
        return reshaped

    def predict_sequence(
        self,
        X: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
        reset_state: bool = False,
        return_sequence: bool = False,
        update_state: bool = True,
    ) -> Any:
        return self._sequence_rollout(
            X,
            context_seq=context,
            targets=None,
            reset_state=reset_state,
            update_params=False,
            update_state=update_state,
            return_sequence=return_sequence,
        )

    def predict_sequence_online(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        context: Optional[np.ndarray] = None,
        reset_state: bool = True,
        return_sequence: bool = True,
        update_state: bool = True,
    ) -> np.ndarray:
        """Teacher-forced rollout with per-step streaming updates."""

        return self._sequence_rollout(
            X,
            context_seq=context,
            targets=y,
            reset_state=reset_state,
            update_params=True,
            update_state=update_state,
            return_sequence=return_sequence,
        )

    # ------------------------------------------------------------------
    # Internal helpers for stateful rollouts
    # ------------------------------------------------------------------

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
    ) -> Any:
        self._ensure_fitted()
        sequence = self._coerce_sequence_inputs(X_seq)
        steps = int(sequence.shape[0])
        if steps == 0:
            raise ValueError("predict_sequence requires at least one timestep.")

        context_arr: Optional[np.ndarray] = None
        expects_context = self._context_dim_ not in (None, 0)
        if context_seq is not None:
            context_arr = self._coerce_sequence_context(context_seq, steps)
        elif expects_context:
            raise ValueError(
                f"This estimator was fit expecting context_dim={self._context_dim_}; provide a context sequence."
            )

        targets_arr: Optional[np.ndarray] = None
        if targets is not None:
            targets_arr = self._coerce_sequence_targets(targets, steps)
        if update_params and targets_arr is None:
            raise ValueError("Streaming rollouts require targets when update_params=True.")

        if reset_state:
            self.reset_state()

        outputs: list[Any] = []
        for idx in range(steps):
            tgt_step = None if targets_arr is None else targets_arr[idx]
            ctx_step = None if context_arr is None else context_arr[idx : idx + 1]
            outputs.append(
                self.step(
                    sequence[idx],
                    context=ctx_step,
                    target=tgt_step,
                    update_params=bool(update_params and targets_arr is not None),
                    update_state=update_state,
                )
            )

        if not return_sequence:
            return outputs[-1]

        stacked_inputs = [np.asarray(out, dtype=np.float32) for out in outputs]
        try:
            return np.stack(stacked_inputs, axis=0)
        except ValueError as exc:
            raise RuntimeError(
                "Sequence outputs have inconsistent shapes; cannot stack results."
            ) from exc

    def _coerce_sequence_inputs(self, sequence: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=np.float32)
        expected_shape = tuple(self.input_shape_)

        if seq.ndim == len(expected_shape):
            seq = seq.reshape((1,) + expected_shape)
        elif seq.ndim == len(expected_shape) + 2 and seq.shape[0] == 1:
            seq = seq.reshape((-1,) + expected_shape)
        elif seq.ndim != len(expected_shape) + 1:
            raise ValueError(
                "Expected sequence shaped (T, ...) optionally preceded by a singleton batch; "
                f"received array with shape {seq.shape}."
            )

        if seq.shape[1:] != expected_shape:
            raise ValueError(
                f"Sequence feature layout {seq.shape[1:]} does not match fitted shape {expected_shape}."
            )

        return seq

    def _coerce_sequence_context(self, context: np.ndarray, steps: int) -> np.ndarray:
        arr = np.asarray(context, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[0] != steps:
            if arr.shape[0] == 1 and steps == 1:
                arr = arr.reshape(1, arr.shape[1])
            else:
                raise ValueError(
                    f"Context sequence length {arr.shape[0]} does not match sequence length {steps}."
                )
        if self._context_dim_ not in (None, 0, arr.shape[1]):
            raise ValueError(
                f"Context feature dimension {arr.shape[1]} does not match expected {self._context_dim_}."
            )
        return arr.astype(np.float32, copy=False)

    def _coerce_sequence_targets(self, targets: np.ndarray, steps: int) -> np.ndarray:
        arr = np.asarray(targets, dtype=np.float32)
        if arr.ndim == 0:
            if steps != 1:
                raise ValueError("Scalar targets are only valid for single-step rollouts.")
            return arr.reshape(1, 1)
        if arr.ndim >= 2 and arr.shape[0] == 1 and arr.shape[1] == steps:
            arr = arr.reshape(steps, *arr.shape[2:])
        elif arr.shape[0] != steps:
            raise ValueError(
                f"Targets length {arr.shape[0]} does not match sequence length {steps}."
            )
        return arr

    def _ensure_streaming_ready(self) -> None:
        if self.stream_lr is None or float(self.stream_lr) <= 0.0:
            raise RuntimeError(
                "Streaming updates require 'stream_lr' > 0. Configure the estimator accordingly."
            )
        self._ensure_fitted()
        model = self.model_
        if model is None:
            raise RuntimeError(
                "Estimator is not fitted yet; cannot initialise streaming optimiser."
            )

        needs_rebuild = self._stream_opt_ is None or self._stream_model_token_ != id(model)
        if needs_rebuild:
            self._stream_opt_ = self._make_optimizer(model, lr=float(self.stream_lr))
            self._stream_loss_ = self._make_loss()
            self._stream_model_token_ = id(model)
            self._stream_last_lr_ = float(self.stream_lr)
        elif self._stream_last_lr_ is None or self._stream_last_lr_ != float(self.stream_lr):
            assert self._stream_opt_ is not None
            for group in self._stream_opt_.param_groups:
                group["lr"] = float(self.stream_lr)
            self._stream_last_lr_ = float(self.stream_lr)

        if self._stream_loss_ is None:
            self._stream_loss_ = self._make_loss()

    def _coerce_stream_target(
        self,
        target: np.ndarray,
        reference: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        arr = np.asarray(target, dtype=np.float32)
        expected_shape = tuple(reference.shape)

        if arr.shape != expected_shape:
            if reference.ndim == 2 and reference.shape[0] == 1:
                feat_dim = int(reference.shape[1])
                if arr.ndim == 1 and arr.shape[0] == feat_dim:
                    arr = arr.reshape(1, feat_dim)
                elif arr.ndim == 0 and feat_dim == 1:
                    arr = np.asarray([arr], dtype=np.float32).reshape(1, 1)
                elif arr.size == reference.numel():
                    arr = arr.reshape(expected_shape)
            elif arr.size == reference.numel():
                arr = arr.reshape(expected_shape)

        if arr.shape != expected_shape:
            raise ValueError(
                f"Streaming target shape {arr.shape} does not match prediction shape {expected_shape}."
            )

        return torch.from_numpy(arr.astype(np.float32, copy=False)).to(device)

    def _apply_stream_update(
        self,
        inputs_np: np.ndarray,
        *,
        context_np: Optional[np.ndarray],
        target: np.ndarray,
    ) -> None:
        self._ensure_streaming_ready()
        model = self.model_
        if model is None:
            raise RuntimeError("Estimator is not fitted yet; cannot apply streaming update.")

        device = self._device()
        model.to(device)
        optimizer = self._stream_opt_
        loss_fn = self._stream_loss_
        if optimizer is None or loss_fn is None:
            raise RuntimeError(
                "Streaming optimiser state is missing; call _ensure_streaming_ready first."
            )

        prev_mode = model.training
        prev_state_updates = None
        if hasattr(model, "set_state_updates"):
            prev_state_updates = getattr(model, "enable_state_updates", None)
            model.set_state_updates(False)

        try:
            model.train(True)
            xb = torch.from_numpy(inputs_np.astype(np.float32, copy=False)).to(device)
            context_t = None
            if context_np is not None:
                context_t = torch.from_numpy(context_np.astype(np.float32, copy=False)).to(device)
            pred = model(xb, context_t) if context_t is not None else model(xb)
            target_t = self._coerce_stream_target(target, pred, device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(pred, target_t)
            loss.backward()
            optimizer.step()

            if hasattr(model, "commit_state_updates"):
                model.commit_state_updates()
        finally:
            if hasattr(model, "set_state_updates"):
                if prev_state_updates is None:
                    model.set_state_updates(True)
                else:
                    model.set_state_updates(bool(prev_state_updates))
            model.train(prev_mode)

    def save(self, path: str) -> None:
        self._ensure_fitted()
        model = self.model_
        orig_device = torch.device("cpu")
        for param in model.parameters():
            orig_device = param.device
            break
        model_cpu = copy.deepcopy(model).cpu()
        payload = {
            "class": self.__class__.__name__,
            "params": self.get_params(deep=True),
            "model": model_cpu,
            "scaler_kind": getattr(self, "_scaler_kind_", None),
            "scaler_state": getattr(self, "_scaler_state_", None),
            "scaler_spec": getattr(self, "_scaler_spec_", None),
            "scaler_obj": self.scaler if getattr(self, "_scaler_kind_", None) == "custom" else None,
            "input_shape": (
                tuple(self.input_shape_)
                if getattr(self, "input_shape_", None) is not None
                else None
            ),
            "internal_shape_cf": (
                tuple(self._internal_input_shape_cf_)
                if getattr(self, "_internal_input_shape_cf_", None) is not None
                else None
            ),
            "primary_dim": self._primary_dim_,
            "output_dim": self._output_dim_,
            "keep_column_output": bool(getattr(self, "_keep_column_output_", False)),
            "train_layout": self._train_inputs_layout_,
            "target_cf_shape": self._target_cf_shape_,
            "target_vector_dim": self._target_vector_dim_,
            "output_shape_tuple": self._output_shape_tuple_,
            "context_dim": self._context_dim_,
        }
        torch.save(payload, path)
        model.to(orig_device)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        map_location: Optional[Union[str, torch.device]] = "cpu",
    ) -> "PSANNRegressor":
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        class_name = payload.get("class")
        if class_name is not None and class_name != cls.__name__:
            raise ValueError(
                f"Checkpoint was created for '{class_name}', cannot load into '{cls.__name__}'."
            )
        params = payload.get("params", {})
        estimator = cls(**params)
        if "model" not in payload:
            raise RuntimeError("Checkpoint is missing model weights.")
        estimator.model_ = payload["model"]
        estimator.model_.to(estimator._device())
        estimator.model_.eval()

        estimator._scaler_kind_ = payload.get("scaler_kind")
        estimator._scaler_state_ = payload.get("scaler_state")
        estimator._scaler_spec_ = payload.get("scaler_spec")
        scaler_obj = payload.get("scaler_obj")
        if scaler_obj is not None:
            estimator.scaler = scaler_obj
            estimator._scaler_fitted_ = True

        input_shape = payload.get("input_shape")
        estimator.input_shape_ = tuple(input_shape) if input_shape is not None else None
        internal_cf = payload.get("internal_shape_cf")
        estimator._internal_input_shape_cf_ = (
            tuple(internal_cf) if internal_cf is not None else None
        )
        estimator._primary_dim_ = payload.get("primary_dim")
        estimator._output_dim_ = payload.get("output_dim")
        estimator._keep_column_output_ = bool(payload.get("keep_column_output", False))
        estimator._train_inputs_layout_ = payload.get("train_layout", "flat")
        target_cf = payload.get("target_cf_shape")
        estimator._target_cf_shape_ = tuple(target_cf) if target_cf is not None else None
        estimator._target_vector_dim_ = payload.get("target_vector_dim")
        output_shape_tuple = payload.get("output_shape_tuple")
        estimator._output_shape_tuple_ = (
            tuple(output_shape_tuple) if output_shape_tuple is not None else None
        )
        estimator._context_dim_ = payload.get("context_dim")

        estimator._hisso_trainer_ = None
        return estimator


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
    ) -> None:
        if preserve_shape:
            raise ValueError("WaveResNetRegressor currently supports preserve_shape=False.")
        if per_element:
            raise ValueError("WaveResNetRegressor does not support per_element=True.")
        if conv_channels is not None:
            warnings.warn(
                "conv_channels has no effect for WaveResNetRegressor; ignoring value.",
                RuntimeWarning,
                stacklevel=2,
            )
        if conv_kernel_size != 1:
            warnings.warn(
                "conv_kernel_size has no effect for WaveResNetRegressor; ignoring value.",
                RuntimeWarning,
                stacklevel=2,
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
            preserve_shape=False,
            data_format=data_format,
            conv_kernel_size=1,
            conv_channels=None,
            per_element=False,
            activation_type=activation_type,
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
        if isinstance(model, WithPreprocessor):
            core = model.core
            if isinstance(core, WaveResNet):
                return core
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
        )
        self.w0_first = float(w0_first)
        self.w0_hidden = float(w0_hidden)
        self.norm = str(norm)
        self.drop_path_max = float(drop_path_max)
        self.residual_alpha_init = float(residual_alpha_init)
        if self.preserve_shape:
            self._use_channel_first_train_inputs_ = True

    def _build_dense_core(
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
    ) -> nn.Module:
        if spatial_ndim != 2:
            raise ValueError("ResConvPSANNRegressor currently supports 2D inputs only")
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
        self.model_.to(device)

        result = maybe_run_hisso(hooks, request, fit_args=fit_args)
        if result is None:
            raise RuntimeError("HISSO requested but no variant hook was provided.")
        return self
