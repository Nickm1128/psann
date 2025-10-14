from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Tuple, Callable, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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

from .nn import PSANNNet, WithPreprocessor, ResidualPSANNNet
from .training import TrainingLoopConfig, run_training_loop
from .conv import PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet, ResidualPSANNConv2dNet
from .utils import choose_device, seed_all
from .types import ActivationConfig
from .preproc import PreprocessorSpec, build_preprocessor
from .extras import (
    SupervisedExtrasConfig,
    ensure_supervised_extras_config,
    rollout_supervised_extras,
    ExtrasGrowthConfig,
    ensure_extras_growth_config,
    extras_growth_to_metadata,
    expand_extras_head,
)


from .hisso import (
    HISSOTrainerConfig,
    coerce_warmstart_config,
    ensure_hisso_trainer_config,
    hisso_evaluate_reward,
    hisso_infer_series,
    run_hisso_supervised_warmstart,
    run_hisso_training,
)



class _ConvHISSOAdapter(nn.Module):
    """Reshape flattened HISSO inputs into conv layouts and append extras channels."""

    def __init__(self, base_shape_cf: Tuple[int, ...], extras_dim: int, base_module: Optional[nn.Module] = None) -> None:
        super().__init__()
        if len(base_shape_cf) < 2:
            raise ValueError("base_shape_cf must include channel and spatial dims")
        self.base_shape = tuple(int(v) for v in base_shape_cf)
        self.base_dim = int(np.prod(self.base_shape))
        self.extras_dim = max(0, int(extras_dim))
        self.base = base_module if base_module is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extras = None
        if x.ndim == len(self.base_shape) + 1:
            base = x
        elif x.ndim == 2:
            if x.shape[1] < self.base_dim:
                raise ValueError("Flattened input dimension is smaller than base conv features")
            base = x[:, : self.base_dim].view(x.shape[0], *self.base_shape)
            if self.extras_dim > 0:
                extras = x[:, self.base_dim : self.base_dim + self.extras_dim]
        else:
            raise ValueError("Input must be (N, F) or (N, C, ...) for conv HISSO adapter")
        if self.base is not None:
            base = self.base(base)
        if self.extras_dim > 0:
            spatial = base.shape[2:]
            if extras is None:
                extras = torch.zeros((base.shape[0], self.extras_dim), device=base.device, dtype=base.dtype)
            extras_map = extras.view(base.shape[0], self.extras_dim, *([1] * len(spatial)))
            extras_map = extras_map.expand(-1, -1, *spatial)
            base = torch.cat([base, extras_map], dim=1)
        return base
class _ExtrasWarmStartController:
    """Manage staged extras warm-start behaviour during training."""

    def __init__(
        self,
        model: nn.Module,
        *,
        extras_dim: int,
        warm_start_epochs: int,
        freeze_until_plateau: bool,
        plateau_patience: int,
        verbose: int = 0,
    ) -> None:
        self.model = model
        self.extras_dim = max(0, int(extras_dim))
        self.warm_start_epochs = max(0, int(warm_start_epochs))
        self.freeze_until_plateau = bool(freeze_until_plateau)
        self.plateau_patience = max(1, int(plateau_patience))
        self.verbose = int(verbose)
        self._head_weight: Optional[torch.Tensor] = None
        self._head_bias: Optional[torch.Tensor] = None
        self._epochs = 0
        self._no_improve = 0
        self._active = self.extras_dim > 0 and (self.warm_start_epochs > 0 or self.freeze_until_plateau)
        if self._active:
            self._capture_head_parameters()
        if self._head_weight is None:
            self._active = False

    def _capture_head_parameters(self) -> None:
        module = self.model
        if isinstance(module, WithPreprocessor):
            module = module.core
        head = getattr(module, 'head', None)
        if isinstance(head, nn.Linear) and self.extras_dim <= int(head.out_features):
            self._head_weight = head.weight
            self._head_bias = head.bias
        else:
            self._head_weight = None
            self._head_bias = None

    def extras_enabled(self) -> bool:
        return not self._active

    def gradient_hook(self, _: nn.Module) -> None:
        if not self._active:
            return
        if self._head_weight is not None and self._head_weight.grad is not None:
            self._head_weight.grad[-self.extras_dim:, :] = 0
        if self._head_bias is not None and self._head_bias.grad is not None:
            self._head_bias.grad[-self.extras_dim:] = 0

    def epoch_callback(
        self,
        epoch_idx: int,
        train_loss: float,
        val_loss: Optional[float],
        improved: bool,
        _patience_left: Optional[int],
    ) -> None:
        if not self._active:
            return
        self._epochs += 1
        if self._epochs < self.warm_start_epochs:
            return
        if self.freeze_until_plateau:
            if improved:
                self._no_improve = 0
            else:
                self._no_improve += 1
            if self._no_improve < self.plateau_patience:
                return
        self._unfreeze(epoch_idx, train_loss, val_loss)

    def _unfreeze(self, epoch_idx: int, train_loss: float, val_loss: Optional[float]) -> None:
        if not self._active:
            return
        self._active = False
        self._no_improve = 0
        message = f"Unfreezing extras head at epoch {epoch_idx + 1} (train_loss={train_loss:.6f}"
        if val_loss is not None:
            message += f", val_loss={val_loss:.6f}"
        message += ")"
        if self.verbose:
            print(message)
        else:
            warnings.warn(message, RuntimeWarning)

    def is_frozen(self) -> bool:
        return self._active

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
        loss: Any = "mse",
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
        state: Optional[Dict[str, Any]] = None,
        state_reset: str = "batch",  # 'batch' | 'epoch' | 'none'
        stream_lr: Optional[float] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        lsm: Optional[Any] = None,
        lsm_train: bool = False,
        lsm_pretrain_epochs: int = 0,
        lsm_lr: Optional[float] = None,
        extras: int = 0,
        extras_growth: Optional[Any] = None,
        extras_warm_start_epochs: Optional[int] = None,
        extras_freeze_until_plateau: Optional[bool] = None,
        warm_start: bool = False,
        scaler: Optional[Union[str, Any]] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.hidden_layers = int(hidden_layers)

        user_set_hidden_units = hidden_units is not None
        user_set_hidden_width = hidden_width is not None
        if user_set_hidden_width and not user_set_hidden_units:
            warnings.warn('`hidden_width` is deprecated; use `hidden_units` instead.', DeprecationWarning, stacklevel=2)
        if user_set_hidden_units and user_set_hidden_width and int(hidden_units) != int(hidden_width):
            warnings.warn('`hidden_units` overrides `hidden_width` because the values differ.', UserWarning, stacklevel=2)
        units_val = hidden_units if user_set_hidden_units else hidden_width
        if units_val is None:
            units_val = 64
        units = int(units_val)
        self.hidden_units = units
        self.hidden_width = units

        user_set_conv = conv_channels is not None
        if user_set_conv and not preserve_shape:
            warnings.warn('`conv_channels` has no effect when preserve_shape=False; ignoring value.', UserWarning, stacklevel=2)
        conv_val = conv_channels if user_set_conv else units
        if conv_val is None:
            conv_val = units
        conv_val = int(conv_val)
        if user_set_conv and user_set_hidden_units and conv_val != units:
            warnings.warn('`conv_channels` differs from `hidden_units`; using `conv_channels` for convolutional paths.', UserWarning, stacklevel=2)
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
        self.state = state or None
        self.state_reset = state_reset
        self.stream_lr = stream_lr
        self.output_shape = output_shape
        self.lsm = lsm
        self.lsm_train = bool(lsm_train)
        self.lsm_pretrain_epochs = int(lsm_pretrain_epochs)
        self.lsm_lr = lsm_lr
        base_extras_dim = max(0, int(extras))
        self._extras_growth_cfg: ExtrasGrowthConfig = ensure_extras_growth_config(extras_growth, default_dim=base_extras_dim)
        if extras_warm_start_epochs is not None or extras_freeze_until_plateau is not None:
            cfg_init = replace(
                self._extras_growth_cfg,
                warm_start_epochs=(
                    int(extras_warm_start_epochs)
                    if extras_warm_start_epochs is not None
                    else self._extras_growth_cfg.warm_start_epochs
                ),
                freeze_until_plateau=(
                    self._extras_growth_cfg.freeze_until_plateau
                    if extras_freeze_until_plateau is None
                    else bool(extras_freeze_until_plateau)
                ),
            )
            self._extras_growth_cfg = cfg_init
        self.extras_growth = self._extras_growth_cfg
        self.extras = int(self._extras_growth_cfg.extras_dim)
        self.warm_start = bool(warm_start)
        # Optional input scaler (minmax/standard or custom object with fit/transform)
        self.scaler = scaler
        self.scaler_params = scaler_params or None
        self._preproc_cfg_ = {'lsm': lsm, 'train': bool(lsm_train), 'pretrain_epochs': int(lsm_pretrain_epochs)}
        self._lsm_module_ = None
        self._extras_cache_: Optional[np.ndarray] = None
        self._hisso_extras_cache_: Optional[np.ndarray] = None
        self._hisso_trainer_: Optional[Any] = None
        self._supervised_extras_meta_: Optional[SupervisedExtrasConfig] = None

        # Training state caches
        self._optimizer_: Optional[torch.optim.Optimizer] = None
        self._lr_scheduler_: Optional[Any] = None
        self._amp_scaler_: Optional[Any] = None
        self._training_state_token_: int = 0

        # Fitted scaler state (set during fit)
        self._scaler_kind_: Optional[str] = None
        self._scaler_state_: Optional[Dict[str, Any]] = None

    @staticmethod
    def _normalize_param_aliases(params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        hidden_units = out.pop('hidden_units', None)
        hidden_width = out.get('hidden_width')
        if hidden_units is None and hidden_width is not None:
            hidden_units = hidden_width
        if hidden_units is not None:
            try:
                hidden_units = int(hidden_units)
            except Exception:
                pass
            out['hidden_units'] = hidden_units
            out.setdefault('hidden_width', hidden_units)
        conv_channels = out.pop('conv_channels', None)
        hidden_channels = out.pop('hidden_channels', None)
        if conv_channels is None:
            conv_channels = hidden_channels
        if conv_channels is None and hidden_units is not None:
            conv_channels = hidden_units
        if conv_channels is not None:
            try:
                conv_channels = int(conv_channels)
            except Exception:
                pass
            out['conv_channels'] = conv_channels
        else:
            out.pop('conv_channels', None)
        return out


    def get_extras_growth(self) -> ExtrasGrowthConfig:
        """Return the canonical extras growth configuration."""

        return ensure_extras_growth_config(self._extras_growth_cfg, default_dim=self.extras)

    def set_extras_growth(self, config: Any) -> "PSANNRegressor":
        """Update extras growth configuration and synchronize derived fields."""

        old_extras = int(self.extras) if hasattr(self, 'extras') else 0
        cfg = ensure_extras_growth_config(config, default_dim=self.extras)
        self._extras_growth_cfg = cfg
        self.extras_growth = cfg
        self.extras = int(cfg.extras_dim)
        if int(self.extras) != int(old_extras):
            self._invalidate_training_state()
        return self

    def _invalidate_training_state(self) -> None:
        """Drop cached optimiser/scheduler/scaler state after structural changes."""

        for attr in ("_optimizer_", "_lr_scheduler_", "_amp_scaler_"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if hasattr(self, "_stream_opt"):
            self._stream_opt = None
        if hasattr(self, "_hisso_trainer_"):
            self._hisso_trainer_ = None
        self._training_state_token_ = getattr(self, "_training_state_token_", 0) + 1

    def set_extras_warm_start_epochs(
        self,
        epochs: Optional[int],
        *,
        freeze_until_plateau: Optional[bool] = None,
    ) -> "PSANNRegressor":
        """Configure staged extras warm-start epochs and plateau behaviour."""

        cfg = self.get_extras_growth()
        cfg_next = replace(
            cfg,
            warm_start_epochs=(None if epochs is None else max(0, int(epochs))),
            freeze_until_plateau=(
                cfg.freeze_until_plateau
                if freeze_until_plateau is None
                else bool(freeze_until_plateau)
            ),
        )
        return self.set_extras_growth(cfg_next)

    def set_params(self, **params):
        """Extend sklearn's set_params to normalise extras-growth shorthands."""

        extras_growth_param = params.pop('extras_growth', None)
        extras_warm_epochs = params.pop('extras_warm_start_epochs', None)
        extras_freeze_flag = params.pop('extras_freeze_until_plateau', None)
        result = super().set_params(**params)
        if extras_growth_param is not None:
            result.set_extras_growth(extras_growth_param)
        if extras_warm_epochs is not None or extras_freeze_flag is not None:
            result.set_extras_warm_start_epochs(extras_warm_epochs, freeze_until_plateau=extras_freeze_flag)
        return result


    def _set_extras_cache(self, cache: Optional[Any]) -> None:
        if cache is None:
            self._extras_cache_ = None
            self._hisso_extras_cache_ = None
            return
        extras_dim = max(0, int(self.get_extras_growth().extras_dim))
        if extras_dim <= 0:
            raise ValueError("extras cache provided but estimator extras_dim=0")
        if isinstance(cache, torch.Tensor):
            arr = cache.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            arr = np.asarray(cache, dtype=np.float32)
        if arr.ndim == 1:
            if arr.shape[0] != extras_dim:
                raise ValueError("extras cache vector length does not match extras_dim")
            arr = arr.reshape(1, extras_dim)
        if arr.ndim != 2 or arr.shape[1] != extras_dim:
            raise ValueError("extras cache must have shape (N, extras_dim)")
        self._extras_cache_ = arr.copy()
        self._hisso_extras_cache_ = self._extras_cache_

    def _get_supervised_extras_config(self) -> SupervisedExtrasConfig:
        cfg = getattr(self, "_supervised_extras_meta_", None)
        if cfg is None:
            raise RuntimeError("No supervised extras configuration available; fit with extras targets to enable extras rollout.")
        cfg = ensure_supervised_extras_config(cfg)
        self._supervised_extras_meta_ = cfg
        return cfg


    # ------------------------- Scaling helpers -------------------------
    def _make_internal_scaler(self) -> Optional[Dict[str, Any]]:
        kind = self.scaler
        if kind is None:
            return None
        if isinstance(kind, str):
            key = kind.lower()
            if key not in {"standard", "minmax"}:
                raise ValueError("Unsupported scaler string. Use 'standard', 'minmax', or provide an object with fit/transform.")
            return {"type": key, "state": {}}
        # Custom object: must implement fit/transform; inverse_transform optional
        if not hasattr(kind, "fit") or not hasattr(kind, "transform"):
            raise ValueError("Custom scaler must implement fit(X) and transform(X). Optional inverse_transform(X).")
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
        lsm_module = module if module is not None else (preproc if isinstance(preproc, nn.Module) else None)
        if lsm_module is None or not hasattr(lsm_module, 'forward'):
            raise RuntimeError(
                "Provided lsm preprocessor must expose a torch.nn.Module. Fit the expander or pass an nn.Module."
            )

        self._lsm_module_ = lsm_module
        attr = 'out_channels' if preserve_shape else 'output_dim'
        dim = getattr(lsm_module, attr, getattr(preproc, attr, None))
        return lsm_module, int(dim) if dim is not None else None


    def _make_optimizer(self, model: torch.nn.Module, lr: Optional[float] = None):
        lr = float(self.lr if lr is None else lr)
        if self.optimizer.lower() == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        if self.optimizer.lower() == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)

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
            raise ValueError(f"Unknown loss '{self.loss}'. Supported: mse, l1/mae, smooth_l1, huber, or a callable.")

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

    def _prepare_supervised_extras_targets(
        self,
        y_primary: np.ndarray,
        *,
        extras_dim: int,
        extras_targets: Optional[np.ndarray],
        extras_loss_weight: Optional[float],
        extras_loss_mode: Optional[str],
        extras_loss_cycle: Optional[int],
    ) -> Optional[Dict[str, object]]:
        extras_requested = (
            extras_targets is not None
            or extras_loss_weight is not None
            or extras_loss_mode is not None
            or extras_loss_cycle is not None
        )
        if not extras_requested:
            return None
        if extras_dim <= 0:
            raise ValueError("extras supervision requested but estimator was initialised with extras=0")
        if y_primary.ndim != 2:
            raise ValueError("y_primary must be 2D with shape (N, primary_dim)")
        if extras_targets is None:
            raise ValueError("extras_targets must be provided when configuring extras supervision")
        extras_arr = np.asarray(extras_targets, dtype=np.float32)
        if extras_arr.ndim == 1:
            extras_arr = extras_arr.reshape(-1, 1)
        if extras_arr.shape[0] != y_primary.shape[0]:
            raise ValueError("extras_targets must have the same number of samples as y")
        if extras_arr.shape[1] != extras_dim:
            raise ValueError(
                f"extras_targets must have shape (N, {extras_dim}); received {extras_arr.shape}"
            )
        extras_arr = extras_arr.reshape(y_primary.shape[0], extras_dim)
        weight = float(extras_loss_weight if extras_loss_weight is not None else 1.0)
        mode = (extras_loss_mode or ("alternate" if weight > 0.0 else "joint")).strip().lower()
        if mode not in {"joint", "alternate"}:
            raise ValueError("extras_loss_mode must be 'joint' or 'alternate'")
        cycle = max(1, int(extras_loss_cycle) if extras_loss_cycle is not None else 2)
        targets_full = np.concatenate(
            [y_primary.astype(np.float32, copy=False), extras_arr.astype(np.float32, copy=False)],
            axis=1,
        )
        return {
            "targets": targets_full,
            "primary_dim": int(y_primary.shape[1]),
            "extras_dim": int(extras_dim),
            "weight": float(weight),
            "mode": mode,
            "cycle": int(cycle),
        }

    def _build_extras_loss(
        self,
        base_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        primary_dim: int,
        extras_dim: int,
        weight: float,
        mode: str,
        cycle: int,
        extras_enabled: Optional[Callable[[], bool]] = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        state = {"step": 0}

        def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            state["step"] += 1
            step = state["step"]
            primary_loss = base_loss_fn(pred[:, :primary_dim], target[:, :primary_dim])
            extras_active = extras_dim > 0 and weight > 0.0
            if extras_active and extras_enabled is not None and not extras_enabled():
                extras_active = False
            if not extras_active:
                return primary_loss
            extras_pred = pred[:, primary_dim : primary_dim + extras_dim]
            extras_target = target[:, primary_dim : primary_dim + extras_dim]
            extras_loss = F.mse_loss(extras_pred, extras_target)
            if mode == "joint":
                return primary_loss + weight * extras_loss
            if mode == "alternate":
                if cycle <= 1:
                    return weight * extras_loss
                if (step - 1) % cycle == 0:
                    return primary_loss
                return weight * extras_loss
            return primary_loss

        return _loss

    # Estimator API
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
        validation_data: Optional[tuple] = None,
        verbose: int = 0,
        noisy: Optional[float | np.ndarray] = None,
        extras_targets: Optional[np.ndarray] = None,
        extras_loss_weight: Optional[float] = None,
        extras_loss_mode: Optional[str] = None,
        extras_loss_cycle: Optional[int] = None,
        hisso: bool = False,
        hisso_window: Optional[int] = None,
        hisso_reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        hisso_context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        hisso_trans_cost: Optional[float] = None,
        hisso_supervised: Optional[Dict[str, Any]] = None,
        lr_max: Optional[float] = None,
        lr_min: Optional[float] = None,
        hisso_extras_weight: Optional[float] = None,
        hisso_extras_mode: Optional[str] = None,
        hisso_extras_cycle: Optional[int] = None,
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
        - validation_data: optional (X_val, y_val) or (X_val, y_val, extras_val) for early stopping/logging
        - verbose: 0/1 to control epoch logging
        - noisy: optional Gaussian input noise std; scalar, per-feature vector, or tensor matching input shape
        - extras_targets: optional (N, extras) array to supervise extras outputs; if omitted and extras columns are appended to y, they will be auto-detected when extras>0
        - extras_loss_weight/mode/cycle: tune scheduling between primary and extras losses (defaults to alternate updates when weight>0 or when extras columns are auto-detected)
        - hisso_supervised: optional bool or dict to run a supervised warm start before HISSO (requires providing 'y')
        - hisso: if True, train via Horizon-Informed Sampling Strategy Optimization (episodic reward)
        - hisso_window: episode/window length for HISSO (default 64)
        - hisso_extras_weight: optional float weight for extras MSE guidance (default off)
        - hisso_extras_mode: optional mode for extras loss ('joint'|'alternate'; default alternate when weight>0)
        - hisso_extras_cycle: optional alternate-cycle length (default 2)
        """
        seed_all(self.random_state)

        val_extras = None
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)):
                raise ValueError("validation_data must be a tuple (X, y) or (X, y, extras)")
            val_tuple = tuple(validation_data)
            if len(val_tuple) == 3:
                validation_data = (val_tuple[0], val_tuple[1])
                val_extras = val_tuple[2]
            elif len(val_tuple) == 2:
                validation_data = (val_tuple[0], val_tuple[1])
            else:
                raise ValueError("validation_data must be length 2 or 3")

        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y, dtype=np.float32)
        if not hisso and y is None:
            raise ValueError("y must be provided when hisso=False")
        if hisso and (extras_targets is not None or extras_loss_weight is not None or extras_loss_mode is not None or extras_loss_cycle is not None):
            raise ValueError("extras_targets/extras loss parameters are only supported when hisso=False")
        extras_cfg = self.get_extras_growth()
        extras_dim = max(0, int(extras_cfg.extras_dim))
        if extras_loss_weight is None and extras_cfg.loss_weight is not None:
            extras_loss_weight = extras_cfg.loss_weight
        if extras_loss_mode is None and extras_cfg.loss_mode is not None:
            extras_loss_mode = extras_cfg.loss_mode
        if extras_loss_cycle is None and extras_cfg.loss_cycle is not None:
            extras_loss_cycle = extras_cfg.loss_cycle
        extras_supervision_requested = (
            extras_targets is not None
            or extras_loss_weight is not None
            or extras_loss_mode is not None
            or extras_loss_cycle is not None
        )
        extras_info: Optional[Dict[str, object]] = None
        self._supervised_extras_meta_ = None
        if extras_supervision_requested and (self.preserve_shape or self.per_element):
            raise NotImplementedError("extras_targets currently supports preserve_shape=False and per_element=False")
        # Handle input shape
        self.input_shape_ = self._infer_input_shape(X)

        # Fit/Update scaler on training data and transform X for model input
        X_for_scaler = X
        # Flatten to (N,D) for scaler in non-preserve shape; else treat channel-wise after moveaxis
        if not self.preserve_shape:
            X2d = self._flatten(X_for_scaler)
            xfm = self._scaler_fit_update(X2d)
            if xfm is not None:
                X2d_scaled = xfm(X2d)
                X_scaled = X2d_scaled.reshape(X.shape[0], *self.input_shape_)
            # Flatten input for episode sampling
            X_flat = self._flatten(X)
            X_train_arr = X_flat
            if y is not None:
                y_total_dim = y.reshape(y.shape[0], -1).shape[1]
            else:
                y_total_dim = int(np.prod(self.input_shape_))
            primary_dim = max(1, int(y_total_dim - extras_dim))
            extras_dim_model = extras_dim
            out_dim = primary_dim + extras_dim_model
            y_vec = None
            if y is not None:
                if y.ndim > 1:
                    y_vec = y.reshape(y.shape[0], -1)
                elif y.ndim == 1:
                    y_vec = y[:, None]
                else:
                    y_vec = y

            if extras_supervision_requested and y_vec is not None:
                extras_info = self._prepare_supervised_extras_targets(
                    y_vec,
                    extras_dim=extras_dim,
                    extras_targets=extras_targets,
                    extras_loss_weight=extras_loss_weight,
                    extras_loss_mode=extras_loss_mode,
                    extras_loss_cycle=extras_loss_cycle,
                )

            if extras_info is None and extras_dim > 0 and y_vec is not None and y_vec.shape[1] >= extras_dim:
                primary_columns = max(1, y_vec.shape[1] - extras_dim)
                extras_arr = y_vec[:, primary_columns: primary_columns + extras_dim]
                if extras_arr.shape[1] == extras_dim:
                    primary_y = y_vec[:, :primary_columns]
                    weight = float(extras_loss_weight if extras_loss_weight is not None else 1.0)
                    mode = (extras_loss_mode or ("alternate" if weight > 0.0 else "joint")).strip().lower()
                    if mode not in {"joint", "alternate"}:
                        raise ValueError("extras_loss_mode must be 'joint' or 'alternate'")
                    cycle = max(1, int(extras_loss_cycle) if extras_loss_cycle is not None else 2)
                    targets_full = np.concatenate(
                        [primary_y.astype(np.float32, copy=False), extras_arr.astype(np.float32, copy=False)],
                        axis=1,
                    )
                    extras_info = {
                        "targets": targets_full,
                        "primary_dim": int(primary_y.shape[1]),
                        "extras_dim": int(extras_dim),
                        "weight": float(weight),
                        "mode": mode,
                        "cycle": int(cycle),
                    }
                    primary_dim = int(primary_y.shape[1])
                    out_dim = primary_dim + extras_dim_model

            if extras_info is not None:
                extras_info = dict(extras_info)
                primary_meta = int(extras_info['primary_dim'])
                extras_meta = int(extras_info['extras_dim'])
                targets_full = np.asarray(extras_info['targets'], dtype=np.float32)
                if primary_meta > 0 and extras_meta > 0:
                    primary_targets = targets_full[:, :primary_meta]
                    extras_targets_arr = targets_full[:, primary_meta: primary_meta + extras_meta]
                    corr_pairs = []
                    for j in range(extras_meta):
                        extras_col = extras_targets_arr[:, j]
                        if float(np.var(extras_col)) <= 1e-12:
                            continue
                        for i in range(primary_meta):
                            primary_col = primary_targets[:, i]
                            if float(np.var(primary_col)) <= 1e-12:
                                continue
                            corr = float(np.corrcoef(primary_col, extras_col)[0, 1])
                            if np.isfinite(corr) and abs(corr) >= 0.98:
                                corr_pairs.append((i, j, corr))
                    if corr_pairs:
                        base_weight = float(extras_info.get('weight', 1.0))
                        weight_scaled = max(base_weight * 0.25, 1e-4)
                        extras_info['weight'] = weight_scaled
                        pairs_text = ", ".join(
                            f"primary[{i}] vs extras[{j}] (corr={corr:.3f})" for i, j, corr in corr_pairs
                        )
                        warnings.warn(
                            f"extras targets are highly correlated with primary outputs ({pairs_text}); scaling extras loss weight to {weight_scaled:.3g} to reduce drift.",
                            RuntimeWarning,
                        )
                feature_dim = int(X_train_arr.shape[1])
                append_flag = feature_dim == primary_meta + extras_meta
                self._supervised_extras_meta_ = SupervisedExtrasConfig(
                    primary_dim=primary_meta,
                    extras_dim=extras_meta,
                    feature_dim=feature_dim,
                    append_to_inputs=bool(append_flag),
                    weight=float(extras_info["weight"]),
                    mode=str(extras_info["mode"]),
                    cycle=int(extras_info["cycle"]),
                )
            else:
                self._supervised_extras_meta_ = None
            lsm_model, lsm_dim = self._resolve_lsm_module(X_train_arr, preserve_shape=False)

            if lsm_model is not None:
                base_out = int(lsm_dim if lsm_dim is not None else getattr(lsm_model, 'output_dim'))
                core_in = base_out + extras_dim_model
            else:
                core_in = primary_dim + extras_dim_model


            # Build core PSANN input dim (LSM(base) + extras passthrough)
            # Build model unless warm-starting with existing compatible model
            rebuild = True
            if self.warm_start and hasattr(self, "model_") and isinstance(self.model_, nn.Module):
                rebuild = False
            if rebuild:
                core_model = PSANNNet(
                    core_in,
                    out_dim,
                    hidden_layers=self.hidden_layers,
                    hidden_units=self.hidden_units,
                    hidden_width=self.hidden_width,
                    act_kw=self.activation,
                    state_cfg=(self.state if self.stateful else None),
                    activation_type=self.activation_type,
                    w0=self.w0,
                )
                preproc = None
                if lsm_model is not None:
                    if not self.lsm_train:
                        for p in lsm_model.parameters():
                            p.requires_grad = False

                    class _BasePlusExtras(torch.nn.Module):
                        def __init__(self, base: torch.nn.Module, base_dim: int, extras_dim: int):
                            super().__init__()
                            self.base = base
                            self.base_dim = int(base_dim)
                            self.extras_dim = int(extras_dim)
                        def forward(self, x: torch.Tensor) -> torch.Tensor:
                            if self.extras_dim <= 0:
                                return self.base(x)
                            xb = x[..., : self.base_dim]
                            xe = x[..., self.base_dim : self.base_dim + self.extras_dim]
                            zb = self.base(xb)
                            return torch.cat([zb, xe], dim=-1)

                    preproc = _BasePlusExtras(lsm_model, base_dim=primary_dim, extras_dim=extras_dim_model)
                self.model_ = WithPreprocessor(preproc, core_model)
            device = self._device()
            self.model_.to(device)

            freeze_controller: Optional[_ExtrasWarmStartController] = None
            warm_epochs_cfg = int(extras_cfg.warm_start_epochs) if extras_cfg.warm_start_epochs is not None else 0
            plateau_flag = bool(extras_cfg.freeze_until_plateau)
            if extras_dim_model > 0 and (warm_epochs_cfg > 0 or plateau_flag):
                plateau_patience = int(self.patience) if int(self.patience) > 0 else max(5, max(warm_epochs_cfg, 1))
                try:
                    freeze_controller = _ExtrasWarmStartController(
                        self.model_,
                        extras_dim=extras_dim_model,
                        warm_start_epochs=warm_epochs_cfg,
                        freeze_until_plateau=plateau_flag,
                        plateau_patience=plateau_patience,
                        verbose=int(verbose),
                    )
                    if not freeze_controller.is_frozen():
                        freeze_controller = None
                except Exception:
                    freeze_controller = None

            warm_cfg = coerce_warmstart_config(hisso_supervised, y)
            if warm_cfg is not None:
                run_hisso_supervised_warmstart(
                    self,
                    X_train_arr,
                    primary_dim=primary_dim,
                    extras_dim=extras_dim_model,
                    config=warm_cfg,
                    lsm_module=lsm_model,
                )

            if hisso:
                extras_weight = float(hisso_extras_weight) if hisso_extras_weight is not None else 0.0
                extras_mode = (
                    hisso_extras_mode.lower().strip()
                    if hisso_extras_mode is not None
                    else ("alternate" if extras_weight > 0.0 else "joint")
                )
                extras_cycle = max(1, int(hisso_extras_cycle) if hisso_extras_cycle is not None else 2)
                trainer_cfg = HISSOTrainerConfig(
                    episode_length=int(hisso_window if hisso_window is not None else 64),
                    episodes_per_batch=32,
                    primary_dim=primary_dim,
                    extras_dim=extras_dim_model,
                    primary_transform="softmax",
                    extras_transform="tanh",
                    random_state=self.random_state,
                    transition_cost=float(hisso_trans_cost) if hisso_trans_cost is not None else 0.0,
                    extras_supervision_weight=extras_weight,
                    extras_supervision_mode=extras_mode,
                    extras_supervision_cycle=int(extras_cycle),
                )

                noise_std = None
                if noisy is not None:
                    noise_std = float(noisy) if np.isscalar(noisy) else None

                observed_window = int(X_train_arr.shape[0])
                requested_episode = int(trainer_cfg.episode_length)
                if observed_window <= 0:
                    raise ValueError("HISSO requires at least one timestep in X")
                allow_full_window = observed_window >= requested_episode
                if not allow_full_window:
                    if extras_dim_model > 0:
                        warnings.warn(
                            f"Deferred HISSO extras cache until {requested_episode} timesteps are available (got {observed_window}).",
                            RuntimeWarning,
                        )
                    adjusted_length = max(1, min(requested_episode, observed_window))
                    if adjusted_length != trainer_cfg.episode_length:
                        trainer_cfg = replace(trainer_cfg, episode_length=adjusted_length)

                cache_in = self._extras_cache_ if allow_full_window else None
                trainer_in = getattr(self, "_hisso_trainer_", None) if allow_full_window else None

                trainer = run_hisso_training(
                    self,
                    X_train_arr,
                    trainer_cfg=trainer_cfg,
                    lr=float(self.lr),
                    device=device,
                    reward_fn=hisso_reward_fn,
                    context_extractor=hisso_context_extractor,
                    extras_cache=cache_in,
                    lr_max=lr_max,
                    lr_min=lr_min,
                    input_noise_std=noise_std,
                    verbose=int(verbose),
                    trainer=trainer_in,
                )
                self._hisso_trainer_ = trainer if allow_full_window else None
                if allow_full_window:
                    self._set_extras_cache(getattr(trainer, "extras_cache", None))
                else:
                    self._set_extras_cache(None)
                self._hisso_cfg_ = trainer_cfg
                self._hisso_trained_ = True
                self.history_ = getattr(trainer, "history", [])
                return self


        else:
            if X.ndim < 3:
                raise ValueError("preserve_shape=True requires X with at least 3 dims (N, C, ...).")
            if self.data_format not in {"channels_first", "channels_last"}:
                raise ValueError("data_format must be 'channels_first' or 'channels_last'")
            X_cf = np.moveaxis(X, -1, 1) if self.data_format == "channels_last" else X
            self._internal_input_shape_cf_ = tuple(X_cf.shape[1:])
            in_channels = int(X_cf.shape[1])
            nd = X_cf.ndim - 2
            # Targets
            if self.per_element:
                # Determine desired output channels
                if self.output_shape is not None:
                    n_targets = int(self.output_shape[-1] if self.data_format == "channels_last" else self.output_shape[0])
                else:
                    # Infer from targets
                    if self.data_format == "channels_last":
                        if y.ndim == X.ndim:
                            n_targets = int(y.shape[-1])
                        elif y.ndim == X.ndim - 1:
                            n_targets = 1
                        else:
                            raise ValueError("y must match X spatial dims, with optional channel last.")
                    else:
                        if y.ndim == X_cf.ndim:
                            n_targets = int(y.shape[1])
                        elif y.ndim == X_cf.ndim - 1:
                            n_targets = 1
                        else:
                            raise ValueError("y must match X spatial dims, with optional channel first.")
                # Prepare y in channels-first layout
                if self.data_format == "channels_last":
                    if y.ndim == X.ndim:
                        y_cf = np.moveaxis(y, -1, 1)
                    elif y.ndim == X.ndim - 1:
                        if n_targets != 1:
                            raise ValueError(f"Provided output_shape implies {n_targets} channels but y has no channel dimension")
                        y_cf = y[:, None, ...]
                else:
                    if y.ndim == X_cf.ndim:
                        y_cf = y
                    elif y.ndim == X_cf.ndim - 1:
                        if n_targets != 1:
                            raise ValueError(f"Provided output_shape implies {n_targets} channels but y has no channel dimension")
                        y_cf = y[:, None, ...]
            else:
                # pooled/vector targets
                y_vec = y.reshape(y.shape[0], -1) if y.ndim > 2 else (y[:, None] if y.ndim == 1 else y)
                if self.output_shape is not None:
                    n_targets = int(np.prod(self.output_shape))
                    if y_vec.shape[1] != n_targets:
                        raise ValueError(f"y has {y_vec.shape[1]} targets, expected {n_targets} from output_shape")
                else:
                    n_targets = int(y_vec.shape[1])
                y_cf = y_vec

            # Optional Conv LSM integration
            lsm_model = None
            if self.lsm is not None:
                if nd != 2:
                    raise ValueError("Conv LSM is currently supported for 2D inputs only.")
                lsm_model, lsm_channels = self._resolve_lsm_module(X_cf, preserve_shape=True)
                if lsm_model is not None:
                    if lsm_channels is not None:
                        in_channels = int(lsm_channels)
                    elif hasattr(lsm_model, 'out_channels'):
                        in_channels = int(getattr(lsm_model, 'out_channels'))
                    elif hasattr(self.lsm, 'out_channels'):
                        in_channels = int(getattr(self.lsm, 'out_channels'))

            # Model (rebuild unless warm-starting with existing compatible model)
            rebuild = True
            if self.warm_start and hasattr(self, "model_") and isinstance(self.model_, nn.Module):
                rebuild = False
            if not rebuild:
                pass
            elif nd == 1:
                self.model_ = PSANNConv1dNet(
                    in_channels,
                    n_targets,
                    hidden_layers=self.hidden_layers,
                    conv_channels=self.conv_channels,
                    hidden_channels=self.conv_channels,
                    kernel_size=self.conv_kernel_size,
                    act_kw=self.activation,
                    activation_type=self.activation_type,
                    w0=self.w0,
                    segmentation_head=self.per_element,
                )
            elif nd == 2:
                self.model_ = PSANNConv2dNet(
                    in_channels,
                    n_targets,
                    hidden_layers=self.hidden_layers,
                    conv_channels=self.conv_channels,
                    hidden_channels=self.conv_channels,
                    kernel_size=self.conv_kernel_size,
                    act_kw=self.activation,
                    activation_type=self.activation_type,
                    w0=self.w0,
                    segmentation_head=self.per_element,
                )
            elif nd == 3:
                self.model_ = PSANNConv3dNet(
                    in_channels,
                    n_targets,
                    hidden_layers=self.hidden_layers,
                    conv_channels=self.conv_channels,
                    hidden_channels=self.conv_channels,
                    kernel_size=self.conv_kernel_size,
                    act_kw=self.activation,
                    activation_type=self.activation_type,
                    w0=self.w0,
                    segmentation_head=self.per_element,
                )
            else:
                raise ValueError(f"Unsupported number of spatial dims: {nd}. Supported: 1, 2, 3.")

            # Compose full model with optional preprocessor
            if rebuild:
                core_model = self.model_
                if lsm_model is not None:
                    if not self.lsm_train:
                        for p in lsm_model.parameters():
                            p.requires_grad = False
                    self.model_ = WithPreprocessor(lsm_model, core_model)
                else:
                    self.model_ = WithPreprocessor(None, core_model)
            X_train_arr = X_cf.astype(np.float32, copy=False)
            n_features = int(np.prod(self.input_shape_))
            X_flat = self._flatten(X)
            y_vec = y.reshape(y.shape[0], -1) if y.ndim > 1 else y[:, None]
            primary_expected = int(np.prod(self.output_shape)) if self.output_shape is not None else None

            if extras_supervision_requested:
                extras_info = self._prepare_supervised_extras_targets(
                    y_vec,
                    extras_dim=extras_dim,
                    extras_targets=extras_targets,
                    extras_loss_weight=extras_loss_weight,
                    extras_loss_mode=extras_loss_mode,
                    extras_loss_cycle=extras_loss_cycle,
                )

            if extras_info is None and extras_dim > 0:
                primary_y = None
                extras_arr = None
                if primary_expected is not None:
                    if y_vec.shape[1] == primary_expected + extras_dim:
                        primary_y = y_vec[:, :primary_expected]
                        extras_arr = y_vec[:, primary_expected:]
                else:
                    if y_vec.shape[1] > extras_dim:
                        primary_y = y_vec[:, :-extras_dim]
                        extras_arr = y_vec[:, -extras_dim:]
                if primary_y is not None and extras_arr is not None:
                    weight = float(extras_loss_weight if extras_loss_weight is not None else 1.0)
                    mode = (extras_loss_mode or ("alternate" if weight > 0.0 else "joint")).strip().lower()
                    if mode not in {"joint", "alternate"}:
                        raise ValueError("extras_loss_mode must be 'joint' or 'alternate'")
                    cycle = max(1, int(extras_loss_cycle) if extras_loss_cycle is not None else 2)
                    targets_full = np.concatenate(
                        [primary_y.astype(np.float32, copy=False), extras_arr.astype(np.float32, copy=False)],
                        axis=1,
                    )
                    extras_info = {
                        "targets": targets_full,
                        "primary_dim": int(primary_y.shape[1]),
                        "extras_dim": int(extras_dim),
                        "weight": float(weight),
                        "mode": mode,
                        "cycle": int(cycle),
                    }

            primary_dim_val: Optional[int] = None
            extras_dim_val = 0
            if extras_info is not None:
                primary_dim_val = int(extras_info['primary_dim'])
                extras_dim_val = int(extras_info['extras_dim'])
                y_cf = extras_info['targets']
                n_targets = int(primary_dim_val + extras_dim_val)
            else:
                y_cf = y_vec
                if primary_expected is not None:
                    n_targets = int(primary_expected)
                    if y_vec.shape[1] != n_targets:
                        raise ValueError(f"y has {y_vec.shape[1]} targets, expected {n_targets} from output_shape")
                else:
                    n_targets = int(y_vec.shape[1])
                primary_dim_val = int(n_targets)
                extras_dim_val = 0

            primary_dim = int(primary_dim_val if primary_dim_val is not None else 0)
            extras_dim_model = int(extras_dim_val) if extras_info is not None else extras_dim
            out_dim = int(primary_dim + extras_dim_model)

            # Optional LSM integration (flattened path only)
            X_train_arr = X_flat
            lsm_model, lsm_dim = self._resolve_lsm_module(X_train_arr, preserve_shape=False)

            if lsm_model is not None:
                lsm_out = lsm_dim if lsm_dim is not None else getattr(lsm_model, 'output_dim', None)
                if lsm_out is None and hasattr(self.lsm, 'output_dim'):
                    lsm_out = getattr(self.lsm, 'output_dim')
                if lsm_out is None:
                    lsm_out = X_train_arr.shape[1] if isinstance(X_train_arr, np.ndarray) else int(n_features)
                in_dim_psann = int(lsm_out)
            else:
                in_dim_psann = int(X_train_arr.shape[1]) if isinstance(X_train_arr, np.ndarray) else int(n_features)

            rebuild = True
            if self.warm_start and hasattr(self, "model_") and isinstance(self.model_, nn.Module):
                rebuild = False
            if rebuild:
                core_model = PSANNNet(
                    in_dim_psann,
                    n_targets,
                    hidden_layers=self.hidden_layers,
                    hidden_units=self.hidden_units,
                    hidden_width=self.hidden_width,
                    act_kw=self.activation,
                    state_cfg=(self.state if self.stateful else None),
                    activation_type=self.activation_type,
                    w0=self.w0,
                )
                if lsm_model is not None:
                    if not self.lsm_train:
                        for p in lsm_model.parameters():
                            p.requires_grad = False
                    self.model_ = WithPreprocessor(lsm_model, core_model)
                else:
                    self.model_ = WithPreprocessor(None, core_model)
        device = self._device()
        self.model_.to(device)

        # Optimizer: include preproc params if training; else only core (requires_grad governs)
        if self.lsm_train and lsm_model is not None:
            # Two param groups to allow separate LR
            params = [
                {"params": self.model_.core.parameters(), "lr": self.lr},
                {"params": self.model_.preproc.parameters(), "lr": float(self.lsm_lr) if self.lsm_lr is not None else self.lr},
            ]
            if self.optimizer.lower() == "adamw":
                opt = torch.optim.AdamW(params, weight_decay=self.weight_decay)
            elif self.optimizer.lower() == "sgd":
                opt = torch.optim.SGD(params, momentum=0.9)
            else:
                opt = torch.optim.Adam(params, weight_decay=self.weight_decay)
        else:
            opt = self._make_optimizer(self.model_)
        self._optimizer_ = opt
        self._lr_scheduler_ = None
        loss_fn = self._make_loss()
        extras_enabled_fn = None
        if freeze_controller is not None:
            extras_enabled_fn = freeze_controller.extras_enabled
        if extras_info is not None:
            loss_fn = self._build_extras_loss(
                loss_fn,
                primary_dim=int(extras_info["primary_dim"]),
                extras_dim=int(extras_info["extras_dim"]),
                weight=float(extras_info["weight"]),
                mode=str(extras_info["mode"]),
                cycle=int(extras_info["cycle"]),
                extras_enabled=extras_enabled_fn,
            )
            y_cf = extras_info["targets"]

        if 'y_cf' not in locals():
            y_cf = y.reshape(y.shape[0], -1) if y is not None and y.ndim > 1 else (y[:, None] if y is not None and y.ndim == 1 else y)
        if 'X_train_arr' not in locals():
            X_train_arr = self._flatten(X) if not self.preserve_shape else X

        # Always feed original inputs to the model (wrapper handles preprocessing)
        targets_np = np.asarray(y_cf, dtype=np.float32)
        inputs_np = np.asarray(X_train_arr, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(inputs_np), torch.from_numpy(targets_np))
        # If state should persist across batches/epoch, disable shuffling to preserve temporal order
        shuffle_batches = True
        if self.stateful and self.state_reset in ("epoch", "none"):
            shuffle_batches = False
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle_batches, num_workers=self.num_workers)

        # Prepare validation tensors if provided
        X_val_t = y_val_t = None
        if validation_data is not None:
            Xv, yv = validation_data
            Xv = np.asarray(Xv, dtype=np.float32)
            yv = np.asarray(yv, dtype=np.float32)
            if self.preserve_shape:
                Xv_cf = np.moveaxis(Xv, -1, 1) if self.data_format == "channels_last" else Xv
                if Xv_cf.shape[1] != self._internal_input_shape_cf_[0]:
                    raise ValueError("validation_data channels mismatch.")
                X_val_t = torch.from_numpy(Xv_cf).to(device)
                if self.per_element:
                    if self.data_format == "channels_last":
                        if yv.ndim == Xv.ndim:
                            yv_cf = np.moveaxis(yv, -1, 1)
                        elif yv.ndim == Xv.ndim - 1:
                            yv_cf = yv[:, None, ...]
                        else:
                            raise ValueError("validation y must match X spatial dims, optional channel last.")
                    else:
                        if yv.ndim == Xv_cf.ndim:
                            yv_cf = yv
                        elif yv.ndim == Xv_cf.ndim - 1:
                            yv_cf = yv[:, None, ...]
                        else:
                            raise ValueError("validation y must match X spatial dims, optional channel first.")
                    y_val_t = torch.from_numpy(yv_cf.astype(np.float32, copy=False)).to(device)
                else:
                    y_val_t = torch.from_numpy(yv.reshape(yv.shape[0], -1).astype(np.float32, copy=False)).to(device)
            else:
                n_features = int(np.prod(self.input_shape_))
                if tuple(Xv.shape[1:]) != self.input_shape_:
                    if int(np.prod(Xv.shape[1:])) != n_features:
                        raise ValueError(
                            f"validation_data X has shape {Xv.shape[1:]}, expected {self.input_shape_} (prod must match {n_features})."
                        )
                X_val_flat = self._flatten(Xv)
                X_val_t = torch.from_numpy(X_val_flat).to(device)
                y_val_flat = yv.reshape(yv.shape[0], -1).astype(np.float32, copy=False)
                if extras_info is not None:
                    extras_dim_val = int(extras_info["extras_dim"])
                    primary_dim_val = int(extras_info["primary_dim"])
                    extras_val_arr = None
                    if val_extras is not None:
                        extras_val_arr = np.asarray(val_extras, dtype=np.float32)
                        if extras_val_arr.ndim == 1:
                            extras_val_arr = extras_val_arr.reshape(-1, 1)
                        extras_val_arr = extras_val_arr.reshape(y_val_flat.shape[0], extras_dim_val)
                        if y_val_flat.shape[1] == primary_dim_val + extras_dim_val:
                            primary_val = y_val_flat[:, :primary_dim_val]
                        elif y_val_flat.shape[1] == primary_dim_val:
                            primary_val = y_val_flat
                        else:
                            raise ValueError(
                                f"validation y has {y_val_flat.shape[1]} columns; expected {primary_dim_val} or {primary_dim_val + extras_dim_val} when extras are supervised."
                            )
                    else:
                        if y_val_flat.shape[1] == primary_dim_val + extras_dim_val:
                            primary_val = y_val_flat[:, :primary_dim_val]
                            extras_val_arr = y_val_flat[:, primary_dim_val:]
                        else:
                            raise ValueError("validation_data must include extras targets (appended to y or provided as a third element) when extras are supervised.")
                    y_val_full = np.concatenate(
                        [primary_val.astype(np.float32, copy=False), extras_val_arr.astype(np.float32, copy=False)],
                        axis=1,
                    )
                else:
                    y_val_full = y_val_flat
                y_val_t = torch.from_numpy(y_val_full.astype(np.float32, copy=False)).to(device)

        # Prepare per-feature noise std (broadcast over batch) if requested
        noise_std_t: Optional[torch.Tensor] = None
        if noisy is not None:
            if self.preserve_shape:
                internal_shape = self._internal_input_shape_cf_
                if np.isscalar(noisy):
                    std = np.full((1, *internal_shape), float(noisy), dtype=np.float32)
                else:
                    arr = np.asarray(noisy, dtype=np.float32)
                    if tuple(arr.shape) == internal_shape:
                        std = arr.reshape(1, *internal_shape)
                    elif tuple(arr.shape) == self.input_shape_ and self.data_format == "channels_last":
                        std = np.moveaxis(arr, -1, 0).reshape(1, *internal_shape)
                    elif arr.ndim == 1 and arr.size == int(np.prod(internal_shape)):
                        std = arr.reshape(1, *internal_shape)
                    else:
                        raise ValueError(
                            f"noisy shape {arr.shape} not compatible with input shape {self.input_shape_}"
                        )
                noise_std_t = torch.from_numpy(std).to(device)
            else:
                n_features = int(np.prod(self.input_shape_))
                if np.isscalar(noisy):
                    std = np.full((1, n_features), float(noisy), dtype=np.float32)
                else:
                    arr = np.asarray(noisy, dtype=np.float32)
                    if arr.ndim == 1 and arr.shape[0] == n_features:
                        std = arr.reshape(1, -1)
                    elif tuple(arr.shape) == self.input_shape_:
                        std = arr.reshape(1, -1)
                    else:
                        raise ValueError(
                            f"noisy shape {arr.shape} not compatible with input shape {self.input_shape_} or flattened size {n_features}"
                        )
                noise_std_t = torch.from_numpy(std).to(device)

        if (lr_max is None) ^ (lr_min is None):
            raise ValueError("Provide both lr_max and lr_min, or neither.")
        if lr_max is not None and lr_min is not None and float(lr_max) < float(lr_min):
            warnings.warn("lr_max < lr_min; swapping to ensure non-increasing schedule.")
            lr_max, lr_min = lr_min, lr_max

        val_inputs = None
        val_targets = None
        if X_val_t is not None and y_val_t is not None:
            if self.lsm is not None and not self.preserve_shape:
                with torch.no_grad():
                    if self.lsm_train and lsm_model is not None:
                        val_inputs = lsm_model(X_val_t)
                    elif hasattr(self.lsm, 'transform'):
                        arr = self.lsm.transform(X_val_t.cpu().numpy())
                        val_inputs = torch.from_numpy(arr.astype(np.float32, copy=False)).to(device)
                    elif hasattr(self.lsm, 'forward'):
                        val_inputs = self.lsm(X_val_t)
                    else:
                        val_inputs = X_val_t
            else:
                val_inputs = X_val_t
            val_targets = y_val_t

        cfg_loop = TrainingLoopConfig(
            epochs=int(self.epochs),
            patience=int(self.patience),
            early_stopping=bool(self.early_stopping),
            stateful=bool(self.stateful),
            state_reset=str(self.state_reset),
            verbose=int(verbose),
            lr_max=None if lr_max is None else float(lr_max),
            lr_min=None if lr_min is None else float(lr_min),
        )

        gradient_hook = None if freeze_controller is None else freeze_controller.gradient_hook
        epoch_cb = None if freeze_controller is None else freeze_controller.epoch_callback

        _, best_state = run_training_loop(
            self.model_,
            optimizer=opt,
            loss_fn=loss_fn,
            train_loader=dl,
            device=device,
            cfg=cfg_loop,
            noise_std=noise_std_t,
            val_inputs=val_inputs,
            val_targets=val_targets,
            gradient_hook=gradient_hook,
            epoch_callback=epoch_cb,
        )

        if best_state is not None and self.early_stopping:
            self.model_.load_state_dict(best_state)
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for X.

        Returns
        - MLP/pooled head: (N, T) if T>1, else 1D shape (N,)
        - per_element=True: channels-first if data_format='channels_first',
          or channels-last if 'channels_last'. Spatial dims mirror input.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        if not hasattr(self, "input_shape_"):
            # Fallback to observed shape
            self.input_shape_ = tuple(X.shape[1:])
        # Validate and prepare
        if self.preserve_shape:
            X_arr = np.moveaxis(X, -1, 1) if self.data_format == "channels_last" else X
            if X_arr.shape[1] != self._internal_input_shape_cf_[0]:
                raise ValueError("X channels mismatch for predict().")
        else:
            if tuple(X.shape[1:]) != self.input_shape_:
                if int(np.prod(X.shape[1:])) != int(np.prod(self.input_shape_)):
                    raise ValueError(
                        f"X has shape {X.shape[1:]}, expected {self.input_shape_} (prod must match)."
                    )
            X_arr = self._flatten(X)
        # Apply scaler if active
        if getattr(self, "_scaler_kind_", None) is not None:
            if not self.preserve_shape:
                if self._scaler_kind_ == "standard":
                    st = self._scaler_state_
                    mean = st["mean"]; var = st["M2"]/max(st["n"],1); std = np.sqrt(np.maximum(var,1e-8)).astype(np.float32)
                    X_arr = (X_arr - mean) / std
                elif self._scaler_kind_ == "minmax":
                    st = self._scaler_state_
                    mn = st["min"]; mx = st["max"]; scale = np.where((mx-mn)>1e-8, (mx-mn), 1.0)
                    X_arr = (X_arr - mn) / scale
                elif self._scaler_kind_ == "custom" and hasattr(self.scaler, "transform"):
                    X_arr = self.scaler.transform(X_arr)
            else:
                # Scale per-channel in channels-first layout
                N, C = X_arr.shape[0], int(X_arr.shape[1])
                X2d = X_arr.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
                if self._scaler_kind_ == "standard":
                    st = self._scaler_state_
                    mean = st["mean"]; var = st["M2"]/max(st["n"],1); std = np.sqrt(np.maximum(var,1e-8)).astype(np.float32)
                    X2d = (X2d - mean) / std
                elif self._scaler_kind_ == "minmax":
                    st = self._scaler_state_
                    mn = st["min"]; mx = st["max"]; scale = np.where((mx-mn)>1e-8, (mx-mn), 1.0)
                    X2d = (X2d - mn) / scale
                elif self._scaler_kind_ == "custom" and hasattr(self.scaler, "transform"):
                    X2d = self.scaler.transform(X2d)
                X_arr = X2d.reshape(N, -1, C).transpose(0, 2, 1).reshape(X_arr.shape)
        device = self._device()
        self.model_.eval()
        with torch.no_grad():
            # Apply scaler if active (non-preserve shape path); preserve_shape handled before model input formatting
            Xin = torch.from_numpy(X_arr).to(device)
            out = self.model_(Xin).cpu().numpy()
        if self.preserve_shape and self.per_element:
            # Return in input's data_format
            if self.data_format == "channels_last":
                out = np.moveaxis(out, 1, -1)
            return out
        else:
            if out.shape[1] == 1:
                out = out[:, 0]
            return out

    # Stateful inference helpers
    def reset_state(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if hasattr(self.model_, "reset_state"):
            self.model_.reset_state()

    def step(self, x_t: np.ndarray, y_t: Optional[np.ndarray] = None, update: bool = False) -> np.ndarray | float:
        """Single-step inference; optionally apply an immediate parameter update.

        Returns a scalar (float) for single-target models or a 1D array for multi-target.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if not self.stateful:
            raise RuntimeError("step() requires stateful=True on the estimator.")
        # Prepare single input respecting preserve_shape/flatten
        xt = np.asarray(x_t, dtype=np.float32)
        if xt.ndim == 1:
            xt = xt[None, :]
        if self.preserve_shape:
            xt = np.moveaxis(xt, -1, 1) if self.data_format == "channels_last" else xt
        else:
            xt = xt.reshape(xt.shape[0], -1)
        # Apply scaler if active
        if getattr(self, "_scaler_kind_", None) is not None:
            if not self.preserve_shape:
                if self._scaler_kind_ == "standard":
                    st = self._scaler_state_
                    mean = st["mean"]; var = st["M2"]/max(st["n"],1); std = np.sqrt(np.maximum(var,1e-8)).astype(np.float32)
                    xt = (xt - mean) / std
                elif self._scaler_kind_ == "minmax":
                    st = self._scaler_state_
                    mn = st["min"]; mx = st["max"]; scale = np.where((mx-mn)>1e-8, (mx-mn), 1.0)
                    xt = (xt - mn) / scale
                elif self._scaler_kind_ == "custom" and hasattr(self.scaler, "transform"):
                    xt = self.scaler.transform(xt)
            else:
                N, C = xt.shape[0], int(xt.shape[1])
                X2d = xt.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
                if self._scaler_kind_ == "standard":
                    st = self._scaler_state_
                    mean = st["mean"]; var = st["M2"]/max(st["n"],1); std = np.sqrt(np.maximum(var,1e-8)).astype(np.float32)
                    X2d = (X2d - mean) / std
                elif self._scaler_kind_ == "minmax":
                    st = self._scaler_state_
                    mn = st["min"]; mx = st["max"]; scale = np.where((mx-mn)>1e-8, (mx-mn), 1.0)
                    X2d = (X2d - mn) / scale
                elif self._scaler_kind_ == "custom" and hasattr(self.scaler, "transform"):
                    X2d = self.scaler.transform(X2d)
                xt = X2d.reshape(N, -1, C).transpose(0, 2, 1)
        device = self._device()
        # Temporarily set to train mode to allow state updates
        prev_mode = self.model_.training
        self.model_.train()
        with torch.no_grad():
            out = self.model_(torch.from_numpy(xt).to(device)).cpu().numpy()
        if hasattr(self.model_, "commit_state_updates"):
            self.model_.commit_state_updates()
        # Optional online update with target, without additional state update
        if update and y_t is not None:
            # Ensure streaming optimizer
            if not hasattr(self, "_stream_opt") or self._stream_opt is None:
                self._stream_opt = self._make_optimizer(self.model_, lr=self.stream_lr)
                self._stream_loss = self._make_loss()
            # Disable state updates during gradient pass
            if hasattr(self.model_, "set_state_updates"):
                self.model_.set_state_updates(False)
            self.model_.train()
            opt = self._stream_opt
            loss_fn = self._stream_loss
            opt.zero_grad()
            xb = torch.from_numpy(xt).to(device)
            pred = self.model_(xb)
            yt = np.asarray(y_t, dtype=np.float32)
            if yt.ndim == 0:
                yt = yt[None]
            if yt.ndim == 1:
                yt = yt[:, None]
            yb = torch.from_numpy(yt).to(device)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            if hasattr(self.model_, "commit_state_updates"):
                self.model_.commit_state_updates()
            if hasattr(self.model_, "set_state_updates"):
                self.model_.set_state_updates(True)
        # Restore mode
        self.model_.train(prev_mode)
        if out.shape[1] == 1:
            return out[0, 0]
        return out[0]

    def predict_sequence(self, X_seq: np.ndarray, *, reset_state: bool = True, return_sequence: bool = False) -> np.ndarray:
        """Free-run over a sequence preserving internal state across steps.

        If return_sequence=False, returns last prediction; else returns the full sequence.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if not self.stateful:
            raise RuntimeError("predict_sequence() requires stateful=True on the estimator.")
        Xs = np.asarray(X_seq, dtype=np.float32)
        if Xs.ndim == 3:
            if Xs.shape[0] != 1:
                raise ValueError("Only batch size 1 supported for predict_sequence (got N != 1).")
            Xs = Xs[0]
        if Xs.ndim != 2:
            raise ValueError("X_seq must be (T, D) or (1, T, D)")
        if reset_state:
            self.reset_state()
        outs = []
        for t in range(Xs.shape[0]):
            outs.append(self.step(Xs[t]))
        outs = np.asarray(outs)
        return outs if return_sequence else outs[-1]

    def predict_sequence_online(self, X_seq: np.ndarray, y_seq: np.ndarray, *, reset_state: bool = True) -> np.ndarray:
        """Online prediction with per-step target updates.

        - Preserves internal state across steps (no resets mid-sequence).
        - After each prediction, immediately updates model params with the true target.
        - Returns the sequence of predictions.
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if not self.stateful:
            raise RuntimeError("predict_sequence_online() requires stateful=True")
        Xs = np.asarray(X_seq, dtype=np.float32)
        ys = np.asarray(y_seq, dtype=np.float32)
        if Xs.ndim == 3:
            if Xs.shape[0] != 1:
                raise ValueError("Only batch size 1 supported (got N != 1)")
            Xs = Xs[0]
        if Xs.ndim != 2:
            raise ValueError("X_seq must be (T, D) or (1, T, D)")
        if ys.ndim == 1:
            ys = ys[:, None]
        if ys.shape[0] != Xs.shape[0]:
            raise ValueError("y_seq must match X_seq length")
        if reset_state:
            self.reset_state()
        outs = []
        for t in range(Xs.shape[0]):
            yhat_t = self.step(Xs[t], y_t=ys[t], update=True)
            outs.append(yhat_t)
        return np.asarray(outs)

    def supervised_extras_rollout(
        self,
        X_obs: np.ndarray,
        *,
        E0: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Roll out supervised extras over a sequence."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if int(self.get_extras_growth().extras_dim) <= 0:
            raise RuntimeError("Estimator was not configured with extras; set extras>0 during fit.")
        cfg = self._get_supervised_extras_config()
        primary, extras_seq, cache = rollout_supervised_extras(
            self,
            X_obs,
            config=cfg,
            extras_cache=self._extras_cache_,
            initial_extras=E0,
        )
        self._set_extras_cache(cache)
        return primary, extras_seq


    # ------------------------- HISSO convenience methods -------------------------
    @torch.no_grad()
    def hisso_infer_series(self, X_obs: np.ndarray, *, E0: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        """If trained with hisso=True, roll out allocations and extras over a series.

        Returns (primary_allocations, extras_seq) with shapes (N, M) and (N+1, K).
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if not getattr(self, "_hisso_trained_", False):
            raise RuntimeError("hisso_infer_series() requires fit(..., hisso=True)")
        cfg = getattr(self, "_hisso_cfg_", None)
        if cfg is None:
            raise RuntimeError("Missing HISSO config on estimator.")
        if not isinstance(cfg, HISSOTrainerConfig):
            cfg = HISSOTrainerConfig.from_predictive_extras(cfg)
            self._hisso_cfg_ = cfg
        X_arr = np.asarray(X_obs, dtype=np.float32)
        episode_len = int(cfg.episode_length)
        extras_dim = max(0, int(self.get_extras_growth().extras_dim))
        cache_in = self._extras_cache_ if extras_dim > 0 else None
        if cache_in is not None:
            try:
                length = int(cache_in.shape[0]) if hasattr(cache_in, "shape") else int(len(cache_in))
            except Exception:
                length = 0
            if length < episode_len + 1:
                cache_in = None
        allow_cache_update = extras_dim > 0 and episode_len > 0 and X_arr.shape[0] >= episode_len
        prim, extras, cache = hisso_infer_series(
            self,
            X_arr,
            trainer_cfg=cfg,
            extras_cache=cache_in,
            initial_extras=E0,
        )
        if allow_cache_update:
            self._set_extras_cache(cache)
        return prim, extras

    @torch.no_grad()
    def hisso_evaluate_reward(self, X_obs: np.ndarray, *, n_batches: int = 8) -> float:
        """Evaluate average episode reward after HISSO training."""
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if not getattr(self, "_hisso_trained_", False):
            raise RuntimeError("hisso_evaluate_reward() requires fit(..., hisso=True)")
        cfg = getattr(self, "_hisso_cfg_", None)
        if cfg is None:
            raise RuntimeError("Missing HISSO config on estimator.")
        if not isinstance(cfg, HISSOTrainerConfig):
            cfg = HISSOTrainerConfig.from_predictive_extras(cfg)
            self._hisso_cfg_ = cfg
        X_arr = np.asarray(X_obs, dtype=np.float32)
        episode_len = int(cfg.episode_length)
        extras_dim = max(0, int(self.get_extras_growth().extras_dim))
        cache_in = self._extras_cache_ if extras_dim > 0 else None
        if cache_in is not None:
            try:
                length = int(cache_in.shape[0]) if hasattr(cache_in, "shape") else int(len(cache_in))
            except Exception:
                length = 0
            if length < episode_len + 1:
                cache_in = None
        allow_cache_update = extras_dim > 0 and episode_len > 0 and X_arr.shape[0] >= episode_len
        val, cache = hisso_evaluate_reward(
            self,
            X_arr,
            trainer_cfg=cfg,
            extras_cache=cache_in,
            n_batches=n_batches,
        )
        if allow_cache_update:
            self._set_extras_cache(cache)
        return val

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        try:
            return float(_sk_r2_score(y, y_pred))
        except Exception:
            # Minimal R^2 fallback
            y = np.asarray(y)
            y_pred = np.asarray(y_pred)
            u = ((y - y_pred) ** 2).sum()
            v = ((y - y.mean()) ** 2).sum()
            return float(1.0 - (u / v if v != 0 else np.nan))

    # Persistence
    def save(self, path: str) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("Model is not fitted. Call fit() before save().")
        params = self.get_params(deep=False) if hasattr(self, "get_params") else {}
        params = self._normalize_param_aliases(params)
        params['hidden_units'] = int(getattr(self, 'hidden_units', params.get('hidden_units', params.get('hidden_width', 64))))
        params['hidden_width'] = int(getattr(self, 'hidden_width', params['hidden_units']))
        conv_default = params['hidden_units']
        params['conv_channels'] = int(getattr(self, 'conv_channels', params.get('conv_channels', conv_default)))
        meta: Dict[str, Any] = {}
        meta["extras_growth"] = extras_growth_to_metadata(self.get_extras_growth())
        # Avoid trying to pickle a custom callable in params
        if callable(params.get("loss", None)):
            params["loss"] = "mse"
            meta["note"] = "Original loss was a custom callable and is not serialized; defaulted to 'mse'."
        # Record integrated preprocessor (LSM) spec if present
        preproc_meta: Dict[str, Any] | None = None
        if hasattr(self, "model_") and isinstance(self.model_, torch.nn.Module) and hasattr(self.model_, "preproc"):
            pre = getattr(self.model_, "preproc")
            if pre is not None:
                # Unwrap helper wrappers (e.g., base+extras)
                pre0 = getattr(pre, "base", pre)
                preproc_meta = {"present": True, "type": pre0.__class__.__name__.lower()}
                # Try to serialize key structural args
                try:
                    if pre0.__class__.__name__ == "LSM":
                        spec = {
                            "input_dim": int(getattr(pre0, "input_dim", 0)),
                            "output_dim": int(getattr(pre0, "output_dim", 0)),
                            "hidden_layers": int(getattr(pre0, "hidden_layers", 0)),
                            "hidden_units": int(getattr(pre0, "hidden_units", getattr(pre0, "hidden_width", 0))),
                            "sparsity": float(getattr(pre0, "sparsity", 0.8)) if hasattr(pre0, "sparsity") else 0.8,
                            "nonlinearity": str(getattr(pre0, "nonlinearity", "sine")) if hasattr(pre0, "nonlinearity") else "sine",
                        }
                        preproc_meta["spec"] = spec
                    elif pre0.__class__.__name__ == "LSMConv2d":
                        # Deduce parameters
                        nonlin = "sine"
                        try:
                            import torch as _t
                            if getattr(pre0, "_act", None) is _t.sin:
                                nonlin = "sine"
                            elif getattr(pre0, "_act", None) is _t.tanh:
                                nonlin = "tanh"
                            else:
                                from torch.nn.functional import relu as _relu
                                if getattr(pre0, "_act", None) is _relu:
                                    nonlin = "relu"
                        except Exception:
                            pass
                        ks = 1
                        try:
                            if len(getattr(pre0, "body", [])) > 0:
                                ks = int(getattr(pre0.body[0], "kernel_size", (1,))[0])
                        except Exception:
                            pass
                        hidden_channels = 128
                        try:
                            if len(getattr(pre0, "body", [])) > 0:
                                hidden_channels = int(pre0.body[0].out_channels)
                        except Exception:
                            pass
                        spec = {
                            "in_channels": int(getattr(pre0, "in_channels", 0)) if hasattr(pre0, "in_channels") else None,
                            "out_channels": int(getattr(pre0, "out_channels", 0)) if hasattr(pre0, "out_channels") else None,
                            "hidden_layers": int(len(getattr(pre0, "body", []))),
                            "conv_channels": hidden_channels,
                            "kernel_size": ks,
                            "sparsity": float(getattr(pre0, "sparsity", 0.8)) if hasattr(pre0, "sparsity") else 0.8,
                            "nonlinearity": nonlin,
                        }
                        preproc_meta["spec"] = spec
                except Exception:
                    pass
        # Do not pickle the original 'lsm' object in params; state captures integrated weights
        if "lsm" in params:
            params["lsm"] = None
        if preproc_meta is not None:
            meta["preproc_meta"] = preproc_meta
        if hasattr(self, "input_shape_"):
            meta["input_shape"] = tuple(self.input_shape_)
        meta["preserve_shape"] = bool(getattr(self, "preserve_shape", False))
        meta["data_format"] = getattr(self, "data_format", "channels_first")
        if hasattr(self, "_internal_input_shape_cf_"):
            meta["internal_input_shape_cf"] = tuple(self._internal_input_shape_cf_)
        meta["per_element"] = bool(getattr(self, "per_element", False))
        # Serialize scaler state if available (built-in scalers only)
        if getattr(self, "_scaler_kind_", None) in {"standard", "minmax"} and getattr(self, "_scaler_state_", None) is not None:
            meta["scaler"] = {
                "kind": self._scaler_kind_,
                "state": {
                    k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self._scaler_state_.items()
                },
            }
        elif getattr(self, "_scaler_kind_", None) == "custom":
            meta.setdefault("warnings", []).append("Custom scaler was used but is not serialized; set it manually after load if needed.")
        # Persist HISSO metadata if available
        if self._supervised_extras_meta_ is not None:
            cfg = ensure_supervised_extras_config(self._supervised_extras_meta_)
            self._supervised_extras_meta_ = cfg
            meta["supervised_extras"] = asdict(cfg)
        if self._extras_cache_ is not None:
            meta["extras_cache"] = np.asarray(self._extras_cache_, dtype=np.float32).tolist()

        if getattr(self, "_hisso_trained_", False):
            cfg = getattr(self, "_hisso_cfg_", None)
            cfg_dict: Optional[Dict[str, Any]] = None
            if cfg is not None:
                try:
                    cfg = ensure_hisso_trainer_config(cfg)
                    self._hisso_cfg_ = cfg
                except Exception:
                    pass
            try:
                cfg_dict = asdict(cfg) if isinstance(cfg, HISSOTrainerConfig) else None  # type: ignore[arg-type]
            except Exception:
                try:
                    cfg_dict = dict(cfg) if isinstance(cfg, dict) else None  # type: ignore[arg-type]
                except Exception:
                    cfg_dict = None
            meta["hisso"] = {"trained": True, "cfg": cfg_dict}
        payload = {
            "class": "PSANNRegressor",
            "params": params,
            "state_dict": self.model_.state_dict(),
            "meta": meta,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: Optional[str | torch.device] = None) -> "PSANNRegressor":
        payload = torch.load(path, map_location=map_location or "cpu")
        params = cls._normalize_param_aliases(payload.get("params", {}))
        saved_extras = int(params.get("extras", 0))
        obj = cls(**params)
        state = payload["state_dict"]
        meta = payload.get("meta", {})

        extras_growth_meta = meta.get("extras_growth")
        if extras_growth_meta is not None:
            try:
                obj.set_extras_growth(ensure_extras_growth_config(extras_growth_meta, default_dim=obj.extras))
            except Exception:
                try:
                    obj.set_extras_growth(extras_growth_meta)
                except Exception:
                    pass
        extras_meta = meta.get("supervised_extras")
        if extras_meta is not None:
            try:
                obj._supervised_extras_meta_ = ensure_supervised_extras_config(extras_meta)
            except Exception:
                obj._supervised_extras_meta_ = extras_meta
        else:
            obj._supervised_extras_meta_ = None
        extras_cache = meta.get("extras_cache")
        if extras_cache is not None:
            obj._set_extras_cache(np.asarray(extras_cache, dtype=np.float32))
        else:
            obj._set_extras_cache(None)
        preproc = None
        pre_meta = meta.get("preproc_meta", None)
        if pre_meta and pre_meta.get("present"):
            ptype = pre_meta.get("type")
            spec = pre_meta.get("spec", {})
            try:
                value: Any
                if ptype:
                    value = PreprocessorSpec(name=str(ptype).lower(), params=spec)
                else:
                    value = spec
                resolved, underlying = build_preprocessor(value, allow_train=False)
                candidate = underlying if underlying is not None else resolved
                preproc = candidate if isinstance(candidate, nn.Module) else None
            except Exception:
                preproc = None

        obj._lsm_module_ = preproc


        # Determine core architecture from namespaced keys
        # Prefer MLP if 'core.body.0.linear.weight' exists; else check conv
        out_dim = None
        if "core.head.weight" in state:
            out_dim = state["core.head.weight"].shape[0]
        if out_dim is None and "core.fc.weight" in state:
            out_dim = state["core.fc.weight"].shape[0]
        if "core.body.0.linear.weight" in state:
            in_dim = state["core.body.0.linear.weight"].shape[1] if obj.hidden_layers > 0 else state["core.head.weight"].shape[1]
            core = PSANNNet(
                int(in_dim),
                int(out_dim),
                hidden_layers=obj.hidden_layers,
                hidden_width=obj.hidden_width,
                act_kw=obj.activation,
                state_cfg=(obj.state if getattr(obj, "stateful", False) else None),
                w0=obj.w0,
            )
        else:
            # Convolutional
            if "core.body.0.conv.weight" not in state:
                raise RuntimeError("Unrecognized state dict: cannot determine MLP or Conv architecture.")
            w = state["core.body.0.conv.weight"]
            in_channels = int(w.shape[1])
            nd = w.ndim - 2
            seg = "core.head.weight" in state and state["core.head.weight"].ndim >= 3 and "core.fc.weight" not in state
            if nd == 1:
                core = PSANNConv1dNet(
                    in_channels,
                    int(out_dim),
                    hidden_layers=obj.hidden_layers,
                    hidden_channels=obj.hidden_width,
                    kernel_size=getattr(obj, "conv_kernel_size", 1),
                    act_kw=obj.activation,
                    w0=obj.w0,
                    segmentation_head=seg,
                )
            elif nd == 2:
                core = PSANNConv2dNet(
                    in_channels,
                    int(out_dim),
                    hidden_layers=obj.hidden_layers,
                    hidden_channels=obj.hidden_width,
                    kernel_size=getattr(obj, "conv_kernel_size", 1),
                    act_kw=obj.activation,
                    w0=obj.w0,
                    segmentation_head=seg,
                )
            elif nd == 3:
                core = PSANNConv3dNet(
                    in_channels,
                    int(out_dim),
                    hidden_layers=obj.hidden_layers,
                    hidden_channels=obj.hidden_width,
                    kernel_size=getattr(obj, "conv_kernel_size", 1),
                    act_kw=obj.activation,
                    w0=obj.w0,
                    segmentation_head=seg,
                )
            else:
                raise RuntimeError("Unsupported convolutional kernel dimensionality in saved state.")
        obj.model_ = WithPreprocessor(preproc, core)
        obj.model_.load_state_dict(state)
        obj.model_.to(choose_device(obj.device))
        # Restore input shape if available
        if "input_shape" in meta:
            obj.input_shape_ = tuple(meta["input_shape"])  # type: ignore[assignment]
        if "preserve_shape" in meta:
            obj.preserve_shape = bool(meta["preserve_shape"])  # type: ignore[assignment]
        if "data_format" in meta:
            obj.data_format = str(meta["data_format"])  # type: ignore[assignment]
        if "internal_input_shape_cf" in meta:
            obj._internal_input_shape_cf_ = tuple(meta["internal_input_shape_cf"])  # type: ignore[assignment]
        if "per_element" in meta:
            obj.per_element = bool(meta["per_element"])  # type: ignore[assignment]
        # Restore scaler state if present
        sc = meta.get("scaler")
        if isinstance(sc, dict) and sc.get("kind") in {"standard", "minmax"}:
            obj._scaler_kind_ = sc["kind"]
            st = sc.get("state", {})
            # Convert lists back to numpy arrays where appropriate
            conv = {}
            for k, v in st.items():
                conv[k] = np.asarray(v, dtype=np.float32) if isinstance(v, (list, tuple)) else v
            obj._scaler_state_ = conv
            obj._scaler_spec_ = {"type": obj._scaler_kind_}
        # Restore HISSO metadata if present
        hisso_meta = meta.get("hisso")
        if isinstance(hisso_meta, dict) and bool(hisso_meta.get("trained")):
            obj._hisso_trained_ = True
            cfg_raw = hisso_meta.get("cfg")
            if cfg_raw is not None:
                try:
                    obj._hisso_cfg_ = ensure_hisso_trainer_config(cfg_raw)
                except Exception:
                    obj._hisso_cfg_ = cfg_raw
            else:
                obj._hisso_cfg_ = None
        # Fallback: infer HISSO dims from state if metadata absent (MLP hisso only)
        if not hasattr(obj, "_hisso_trained_") or not getattr(obj, "_hisso_trained_", False):
            try:
                # Out dim already computed above
                out_dim_infer = None
                if "core.head.weight" in state:
                    out_dim_infer = int(state["core.head.weight"].shape[0])
                elif "core.fc.weight" in state:
                    out_dim_infer = int(state["core.fc.weight"].shape[0])
                in_dim_infer = None
                if "core.body.0.linear.weight" in state:
                    in_dim_infer = int(state["core.body.0.linear.weight"].shape[1])
                elif "core.head.weight" in state:
                    in_dim_infer = int(state["core.head.weight"].shape[1])
                if out_dim_infer is not None and in_dim_infer is not None and out_dim_infer >= in_dim_infer:
                    extras_dim = max(0, out_dim_infer - in_dim_infer)
                    obj._hisso_cfg_ = ensure_hisso_trainer_config({
                        "episode_length": 64,
                        "episodes_per_batch": 32,
                        "primary_dim": int(in_dim_infer),
                        "extras_dim": int(extras_dim),
                        "primary_transform": "softmax",
                        "extras_transform": "tanh",
                        "transition_cost": 0.0,
                        "random_state": obj.random_state,
                    })
                    obj._hisso_trained_ = True
            except Exception:
                pass
        target_cfg = obj.get_extras_growth()
        requested_extras = int(target_cfg.extras_dim)
        auto_expand_requested = (
            getattr(target_cfg, "auto_expand_on_load", True)
            and requested_extras > saved_extras
        )
        if auto_expand_requested:
            hisso_enabled = bool(getattr(obj, "_hisso_trained_", False))
            hisso_cfg = getattr(obj, "_hisso_cfg_", None)
            episode_len = 0
            extras_dim_hisso = 0
            if isinstance(hisso_cfg, HISSOTrainerConfig):
                episode_len = int(hisso_cfg.episode_length)
                extras_dim_hisso = int(hisso_cfg.extras_dim)
            elif hasattr(hisso_cfg, "episode_length"):
                try:
                    episode_len = int(hisso_cfg.episode_length)
                except Exception:
                    episode_len = 0
                try:
                    extras_dim_hisso = int(getattr(hisso_cfg, "extras_dim"))
                except Exception:
                    extras_dim_hisso = 0
            if extras_dim_hisso <= 0:
                hisso_enabled = False
            cache = getattr(obj, "_extras_cache_", None)
            observed_window = 0
            if cache is not None:
                try:
                    length = int(cache.shape[0]) if hasattr(cache, "shape") else int(len(cache))
                except Exception:
                    length = 0
                observed_window = max(0, length - 1)
            allow_expand = hisso_enabled and episode_len > 0 and observed_window >= episode_len
            if allow_expand or not hisso_enabled:
                obj.set_extras_growth(saved_extras)
                obj = expand_extras_head(obj, requested_extras, extras_growth=target_cfg)
                obj._invalidate_training_state()
            else:
                if episode_len <= 0:
                    warnings.warn("Skipped extras auto-expansion on load because HISSO episode length could not be resolved; keeping persisted extras width.", RuntimeWarning)
                else:
                    warnings.warn(
                        f"Skipped extras auto-expansion on load until at least {episode_len} timesteps are cached (found {observed_window}).",
                        RuntimeWarning,
                    )

        return obj


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
        loss: Any = "mse",
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
        state: Optional[Dict[str, Any]] = None,
        state_reset: str = "batch",
        stream_lr: Optional[float] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        lsm: Optional[Any] = None,
        lsm_train: bool = False,
        lsm_pretrain_epochs: int = 0,
        lsm_lr: Optional[float] = None,
        extras: int = 0,
        extras_growth: Optional[Any] = None,
        warm_start: bool = False,
        scaler: Optional[Union[str, Any]] = None,
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
            extras=extras,
            extras_growth=extras_growth,
            warm_start=warm_start,
            scaler=scaler,
            scaler_params=scaler_params,
        )
        self.w0_first = float(w0_first)
        self.w0_hidden = float(w0_hidden)
        self.norm = str(norm)
        self.drop_path_max = float(drop_path_max)
        self.residual_alpha_init = float(residual_alpha_init)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
        validation_data: Optional[tuple] = None,
        verbose: int = 0,
        noisy: Optional[float | np.ndarray] = None,
        extras_targets: Optional[np.ndarray] = None,
        extras_loss_weight: Optional[float] = None,
        extras_loss_mode: Optional[str] = None,
        extras_loss_cycle: Optional[int] = None,
        hisso: bool = False,
        hisso_window: Optional[int] = None,
        hisso_reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        hisso_context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        hisso_trans_cost: Optional[float] = None,
        hisso_supervised: Optional[Dict[str, Any]] = None,
        lr_max: Optional[float] = None,
        lr_min: Optional[float] = None,
        hisso_extras_weight: Optional[float] = None,
        hisso_extras_mode: Optional[str] = None,
        hisso_extras_cycle: Optional[int] = None,
    ):
        # This mirrors PSANNRegressor.fit but swaps PSANNNet -> ResidualPSANNNet
        seed_all(self.random_state)

        val_extras = None
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)):
                raise ValueError("validation_data must be a tuple (X, y) or (X, y, extras)")
            val_tuple = tuple(validation_data)
            if len(val_tuple) == 3:
                validation_data = (val_tuple[0], val_tuple[1])
                val_extras = val_tuple[2]
            elif len(val_tuple) == 2:
                validation_data = (val_tuple[0], val_tuple[1])
            else:
                raise ValueError("validation_data must be length 2 or 3")

        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y, dtype=np.float32)
        if not hisso and y is None:
            raise ValueError("y must be provided when hisso=False")
        if hisso and (extras_targets is not None or extras_loss_weight is not None or extras_loss_mode is not None or extras_loss_cycle is not None):
            raise ValueError("extras_targets/extras loss parameters are only supported when hisso=False")
        extras_cfg = self.get_extras_growth()
        extras_dim = max(0, int(extras_cfg.extras_dim))
        if extras_loss_weight is None and extras_cfg.loss_weight is not None:
            extras_loss_weight = extras_cfg.loss_weight
        if extras_loss_mode is None and extras_cfg.loss_mode is not None:
            extras_loss_mode = extras_cfg.loss_mode
        if extras_loss_cycle is None and extras_cfg.loss_cycle is not None:
            extras_loss_cycle = extras_cfg.loss_cycle
        extras_supervision_requested = (
            extras_targets is not None
            or extras_loss_weight is not None
            or extras_loss_mode is not None
            or extras_loss_cycle is not None
        )
        extras_info: Optional[Dict[str, object]] = None
        self._supervised_extras_meta_ = None
        if extras_supervision_requested and (self.preserve_shape or self.per_element):
            raise NotImplementedError("extras_targets currently supports preserve_shape=False and per_element=False")
        self.input_shape_ = self._infer_input_shape(X)

        # Scale inputs
        if not self.preserve_shape:
            X2d = self._flatten(X)
            xfm = self._scaler_fit_update(X2d)
            X = xfm(X2d).reshape(X.shape[0], *self.input_shape_) if xfm is not None else X
        else:
            X_cf = np.moveaxis(X, -1, 1) if self.data_format == "channels_last" else X
            self._internal_input_shape_cf_ = tuple(X_cf.shape[1:])
            N, C = X_cf.shape[0], int(X_cf.shape[1])
            X2d = X_cf.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
            xfm = self._scaler_fit_update(X2d)
            if xfm is not None:
                X2d_scaled = xfm(X2d)
                X_cf_scaled = X2d_scaled.reshape(N, -1, C).transpose(0, 2, 1).reshape(X_cf.shape)
                X = np.moveaxis(X_cf_scaled, 1, -1) if self.data_format == "channels_last" else X_cf_scaled

        if self.stateful and not self.state:
            warnings.warn(
                "stateful=True but no state config provided; stateful mechanics will be disabled. "
                "Pass state={init,rho,beta,max_abs,detach} to enable persistent state.",
                RuntimeWarning,
            )

        # HISSO branch
        if hisso:
            if self.preserve_shape:
                raise ValueError("hisso=True currently supports flattened vector inputs only (preserve_shape=False)")
            primary_dim = int(np.prod(self.input_shape_))
            extras_dim = max(0, int(extras_cfg.extras_dim))
            extras_dim_model = extras_dim
            out_dim = primary_dim + extras_dim_model
            X_train_arr = X.reshape(X.shape[0], -1)
            if extras_info is not None:
                primary_dim_val = int(extras_info['primary_dim'])
                extras_dim_val = int(extras_info['extras_dim'])
                feature_dim = int(X_train_arr.shape[1])
                append_flag = feature_dim == primary_dim_val + extras_dim_val
                self._supervised_extras_meta_ = SupervisedExtrasConfig(
                    primary_dim=primary_dim_val,
                    extras_dim=extras_dim_val,
                    feature_dim=feature_dim,
                    append_to_inputs=bool(append_flag),
                    weight=float(extras_info['weight']),
                    mode=str(extras_info['mode']),
                    cycle=int(extras_info['cycle']),
                )
                primary_dim = primary_dim_val
                extras_dim_model = extras_dim_val
                out_dim = int(primary_dim + extras_dim_model)
            else:
                self._supervised_extras_meta_ = None
            lsm_model, lsm_dim = self._resolve_lsm_module(X_train_arr, preserve_shape=False)

            if lsm_model is not None:
                base_out = int(lsm_dim if lsm_dim is not None else getattr(lsm_model, 'output_dim'))
                core_in = base_out + extras_dim_model
            else:
                core_in = primary_dim + extras_dim_model

            rebuild = True
            if self.warm_start and hasattr(self, "model_") and isinstance(self.model_, nn.Module):
                rebuild = False
            if rebuild:
                core_model = ResidualPSANNNet(
                    core_in,
                    out_dim,
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
                preproc = None
                if lsm_model is not None:
                    if not self.lsm_train:
                        for p in lsm_model.parameters():
                            p.requires_grad = False

                    class _BasePlusExtras(torch.nn.Module):
                        def __init__(self, base: torch.nn.Module, base_dim: int, extras_dim: int):
                            super().__init__()
                            self.base = base
                            self.base_dim = int(base_dim)
                            self.extras_dim = int(extras_dim)
                        def forward(self, x: torch.Tensor) -> torch.Tensor:
                            if self.extras_dim <= 0:
                                return self.base(x)
                            xb = x[..., : self.base_dim]
                            xe = x[..., self.base_dim : self.base_dim + self.extras_dim]
                            zb = self.base(xb)
                            return torch.cat([zb, xe], dim=-1)
                    preproc = _BasePlusExtras(lsm_model, base_dim=primary_dim, extras_dim=extras_dim_model)
                self.model_ = WithPreprocessor(preproc, core_model)
            device = self._device()
            self.model_.to(device)

            warm_cfg = coerce_warmstart_config(hisso_supervised, y)
            if warm_cfg is not None:
                run_hisso_supervised_warmstart(
                    self,
                    X_train_arr,
                    primary_dim=primary_dim,
                    extras_dim=extras_dim_model,
                    config=warm_cfg,
                    lsm_module=lsm_model,
                )

        if hisso:
                        extras_weight = float(hisso_extras_weight) if hisso_extras_weight is not None else 0.0
                        extras_mode = (
                            hisso_extras_mode.lower().strip()
                            if hisso_extras_mode is not None
                            else ("alternate" if extras_weight > 0.0 else "joint")
                        )
                        extras_cycle = max(1, int(hisso_extras_cycle) if hisso_extras_cycle is not None else 2)
                        trainer_cfg = HISSOTrainerConfig(
                            episode_length=int(hisso_window if hisso_window is not None else 64),
                            episodes_per_batch=32,
                            primary_dim=primary_dim,
                            extras_dim=extras_dim_model,
                            primary_transform="softmax",
                            extras_transform="tanh",
                            random_state=self.random_state,
                            transition_cost=float(hisso_trans_cost) if hisso_trans_cost is not None else 0.0,
                            extras_supervision_weight=extras_weight,
                            extras_supervision_mode=extras_mode,
                            extras_supervision_cycle=int(extras_cycle),
                        )

                        noise_std = None
                        if noisy is not None:
                            noise_std = float(noisy) if np.isscalar(noisy) else None

                        trainer = run_hisso_training(
                            self,
                            X_train_arr,
                            trainer_cfg=trainer_cfg,
                            lr=float(self.lr),
                            device=device,
                            reward_fn=hisso_reward_fn,
                            context_extractor=hisso_context_extractor,
                            extras_cache=self._extras_cache_,
                            lr_max=lr_max,
                            lr_min=lr_min,
                            input_noise_std=noise_std,
                            verbose=int(verbose),
                            trainer=getattr(self, "_hisso_trainer_", None),
                        )
                        self._hisso_trainer_ = trainer
                        self._set_extras_cache(getattr(trainer, "extras_cache", None))
                        self._hisso_cfg_ = trainer_cfg
                        self._hisso_trained_ = True
                        self.history_ = getattr(trainer, "history", [])
                        return self



        # Supervised branch
        if self.preserve_shape:
            if X.ndim < 3:
                raise ValueError("preserve_shape=True requires X with at least 3 dims (N, C, ...).")
            if self.data_format not in {"channels_first", "channels_last"}:
                raise ValueError("data_format must be 'channels_first' or 'channels_last'")
            X_cf = np.moveaxis(X, -1, 1) if self.data_format == "channels_last" else X
            self._internal_input_shape_cf_ = tuple(X_cf.shape[1:])
            in_channels = int(X_cf.shape[1])
            nd = X_cf.ndim - 2

            # Targets
            if self.per_element:
                if self.output_shape is not None:
                    n_targets = int(self.output_shape[-1] if self.data_format == "channels_last" else self.output_shape[0])
                else:
                    if self.data_format == "channels_last":
                        if y.ndim == X.ndim:
                            n_targets = int(y.shape[-1])
                        elif y.ndim == X.ndim - 1:
                            n_targets = 1
                        else:
                            raise ValueError("y must match X spatial dims, with optional channel last.")
                    else:
                        if y.ndim == X_cf.ndim:
                            n_targets = int(y.shape[1])
                        elif y.ndim == X_cf.ndim - 1:
                            n_targets = 1
                        else:
                            raise ValueError("y must match X spatial dims, with optional channel first.")
                if self.data_format == "channels_last":
                    if y.ndim == X.ndim:
                        y_cf = np.moveaxis(y, -1, 1)
                    elif y.ndim == X.ndim - 1:
                        if n_targets != 1:
                            raise ValueError(f"Provided output_shape implies {n_targets} channels but y has no channel dimension")
                        y_cf = y[:, None, ...]
                else:
                    if y.ndim == X_cf.ndim:
                        y_cf = y
                    elif y.ndim == X_cf.ndim - 1:
                        if n_targets != 1:
                            raise ValueError(f"Provided output_shape implies {n_targets} channels but y has no channel dimension")
                        y_cf = y[:, None, ...]
            else:
                y_vec = y.reshape(y.shape[0], -1) if y.ndim > 2 else (y[:, None] if y.ndim == 1 else y)
                if self.output_shape is not None:
                    n_targets = int(np.prod(self.output_shape))
                    if y_vec.shape[1] != n_targets:
                        raise ValueError(f"y has {y_vec.shape[1]} targets, expected {n_targets} from output_shape")
                else:
                    n_targets = int(y_vec.shape[1])
                y_cf = y_vec

            # Optional conv LSM
            lsm_model = None
            if self.lsm is not None:
                if nd != 2:
                    warnings.warn("preserve_shape=True currently supports 2D conv LSM only.")
                else:
                    lsm_model, lsm_channels = self._resolve_lsm_module(X_cf, preserve_shape=True)
                    if lsm_model is not None:
                        if lsm_channels is not None:
                            in_channels = int(lsm_channels)
                        elif hasattr(lsm_model, 'out_channels'):
                            in_channels = int(getattr(lsm_model, 'out_channels'))
                        elif hasattr(self.lsm, 'out_channels'):
                            in_channels = int(getattr(self.lsm, 'out_channels'))

            # Determine in-dim for MLP head (flattened)
            in_dim_psann = int(X_cf.reshape(X_cf.shape[0], -1).shape[1])

            rebuild = True
            if self.warm_start and hasattr(self, "model_") and isinstance(self.model_, nn.Module):
                rebuild = False
            if rebuild:
                core_model = ResidualPSANNNet(
                    in_dim_psann,
                    n_targets,
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
                if lsm_model is not None:
                    if not self.lsm_train:
                        for p in lsm_model.parameters():
                            p.requires_grad = False
                    self.model_ = WithPreprocessor(lsm_model, core_model)
                else:
                    self.model_ = WithPreprocessor(None, core_model)
            device = self._device()
            self.model_.to(device)

            # Optimizer and loaders
            if self.lsm_train and lsm_model is not None:
                params = [
                    {"params": self.model_.core.parameters(), "lr": self.lr},
                    {"params": self.model_.preproc.parameters(), "lr": float(self.lsm_lr) if self.lsm_lr is not None else self.lr},
                ]
                if self.optimizer.lower() == "adamw":
                    opt = torch.optim.AdamW(params, weight_decay=self.weight_decay)
                elif self.optimizer.lower() == "sgd":
                    opt = torch.optim.SGD(params, momentum=0.9)
                else:
                    opt = torch.optim.Adam(params, weight_decay=self.weight_decay)
            else:
                opt = self._make_optimizer(self.model_)
            self._optimizer_ = opt
            self._lr_scheduler_ = None
            loss_fn = self._make_loss()

            ds = TensorDataset(torch.from_numpy(X_cf.astype(np.float32, copy=False)), torch.from_numpy(y_cf.astype(np.float32, copy=False)))
            shuffle_batches = True
            if self.stateful and self.state_reset in ("epoch", "none"):
                shuffle_batches = False
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle_batches, num_workers=self.num_workers)

            X_val_t = y_val_t = None
            if validation_data is not None:
                Xv, yv = validation_data
                Xv = np.asarray(Xv, dtype=np.float32)
                yv = np.asarray(yv, dtype=np.float32)
                if Xv.ndim != X.ndim:
                    raise ValueError("validation X must match dimensionality of training X")
                Xv_cf = np.moveaxis(Xv, -1, 1) if self.data_format == "channels_last" else Xv
                device = self._device()
                X_val_t = torch.from_numpy(Xv_cf.astype(np.float32, copy=False)).to(device)
                if self.per_element:
                    if self.data_format == "channels_last":
                        if yv.ndim == X.ndim:
                            yv_cf = np.moveaxis(yv, -1, 1)
                        elif yv.ndim == X.ndim - 1:
                            yv_cf = yv[:, None, ...]
                        else:
                            raise ValueError("validation y must match X spatial dims, optional channel last.")
                    else:
                        if yv.ndim == Xv_cf.ndim:
                            yv_cf = yv
                        elif yv.ndim == Xv_cf.ndim - 1:
                            yv_cf = yv[:, None, ...]
                        else:
                            raise ValueError("validation y must match X spatial dims, optional channel first.")
                    y_val_t = torch.from_numpy(yv_cf.astype(np.float32, copy=False)).to(device)
                else:
                    y_val_t = torch.from_numpy(yv.reshape(yv.shape[0], -1).astype(np.float32, copy=False)).to(device)
        else:
            # Flattened vector MLP (no LSM)
            X_train_arr = X.reshape(X.shape[0], -1)
            y_vec = y.reshape(y.shape[0], -1) if y.ndim > 2 else (y[:, None] if y.ndim == 1 else y)
            primary_expected = int(np.prod(self.output_shape)) if self.output_shape is not None else None

            if extras_supervision_requested:
                extras_info = self._prepare_supervised_extras_targets(
                    y_vec,
                    extras_dim=extras_dim,
                    extras_targets=extras_targets,
                    extras_loss_weight=extras_loss_weight,
                    extras_loss_mode=extras_loss_mode,
                    extras_loss_cycle=extras_loss_cycle,
                )

            if extras_info is None and extras_dim > 0:
                primary_y = None
                extras_arr = None
                if primary_expected is not None:
                    if y_vec.shape[1] == primary_expected + extras_dim:
                        primary_y = y_vec[:, :primary_expected]
                        extras_arr = y_vec[:, primary_expected:]
                else:
                    if y_vec.shape[1] > extras_dim:
                        primary_y = y_vec[:, :-extras_dim]
                        extras_arr = y_vec[:, -extras_dim:]
                if primary_y is not None and extras_arr is not None:
                    weight = float(extras_loss_weight if extras_loss_weight is not None else 1.0)
                    mode = (extras_loss_mode or ("alternate" if weight > 0.0 else "joint")).strip().lower()
                    if mode not in {"joint", "alternate"}:
                        raise ValueError("extras_loss_mode must be 'joint' or 'alternate'")
                    cycle = max(1, int(extras_loss_cycle) if extras_loss_cycle is not None else 2)
                    targets_full = np.concatenate(
                        [primary_y.astype(np.float32, copy=False), extras_arr.astype(np.float32, copy=False)],
                        axis=1,
                    )
                    extras_info = {
                        "targets": targets_full,
                        "primary_dim": int(primary_y.shape[1]),
                        "extras_dim": int(extras_dim),
                        "weight": float(weight),
                        "mode": mode,
                        "cycle": int(cycle),
                    }

            primary_dim_val: Optional[int] = None
            extras_dim_val = 0
            if extras_info is not None:
                primary_dim_val = int(extras_info["primary_dim"])
                extras_dim_val = int(extras_info["extras_dim"])
                targets_np = extras_info["targets"]
                n_targets = int(primary_dim_val + extras_dim_val)
            else:
                targets_np = y_vec.astype(np.float32, copy=False)
                if primary_expected is not None:
                    n_targets = int(primary_expected)
                    if y_vec.shape[1] != n_targets:
                        raise ValueError(
                            f"y has {y_vec.shape[1]} targets, expected {n_targets} from output_shape"
                        )
                else:
                    n_targets = int(y_vec.shape[1])
                primary_dim_val = int(n_targets)
                extras_dim_val = 0

            rebuild = True
            if self.warm_start and hasattr(self, "model_") and isinstance(self.model_, nn.Module):
                rebuild = False
            if rebuild:
                core_model = ResidualPSANNNet(
                    int(X_train_arr.shape[1]),
                    n_targets,
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
                self.model_ = WithPreprocessor(None, core_model)
            device = self._device()
            self.model_.to(device)
            opt = self._make_optimizer(self.model_)
            loss_fn = self._make_loss()
            if extras_info is not None:
                loss_fn = self._build_extras_loss(
                    loss_fn,
                    primary_dim=int(extras_info["primary_dim"]),
                    extras_dim=int(extras_info["extras_dim"]),
                    weight=float(extras_info["weight"]),
                    mode=str(extras_info["mode"]),
                    cycle=int(extras_info["cycle"]),
                )

            X_train_np = X_train_arr.astype(np.float32, copy=False)
            ds = TensorDataset(torch.from_numpy(X_train_np), torch.from_numpy(targets_np.astype(np.float32, copy=False)))
            shuffle_batches = True
            if self.stateful and self.state_reset in ("epoch", "none"):
                shuffle_batches = False
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle_batches, num_workers=self.num_workers)

            X_val_t = y_val_t = None
            if validation_data is not None:
                Xv, yv = validation_data
                Xv = np.asarray(Xv, dtype=np.float32)
                yv = np.asarray(yv, dtype=np.float32)
                n_features = int(np.prod(self.input_shape_))
                if tuple(Xv.shape[1:]) != self.input_shape_:
                    if int(np.prod(Xv.shape[1:])) != n_features:
                        raise ValueError(
                            f"validation_data X has shape {Xv.shape[1:]}, expected {self.input_shape_} (prod must match {n_features})."
                        )
                X_val_flat = self._flatten(Xv)
                X_val_t = torch.from_numpy(X_val_flat).to(device)
                y_val_flat = yv.reshape(yv.shape[0], -1).astype(np.float32, copy=False)
                if extras_info is not None:
                    extras_dim_val = int(extras_info["extras_dim"])
                    primary_dim_val = int(extras_info["primary_dim"])
                    extras_val_arr = None
                    if val_extras is not None:
                        extras_val_arr = np.asarray(val_extras, dtype=np.float32)
                        if extras_val_arr.ndim == 1:
                            extras_val_arr = extras_val_arr.reshape(-1, 1)
                        extras_val_arr = extras_val_arr.reshape(y_val_flat.shape[0], extras_dim_val)
                        if y_val_flat.shape[1] == primary_dim_val + extras_dim_val:
                            primary_val = y_val_flat[:, :primary_dim_val]
                        elif y_val_flat.shape[1] == primary_dim_val:
                            primary_val = y_val_flat
                        else:
                            raise ValueError(
                                f"validation y has {y_val_flat.shape[1]} columns; expected {primary_dim_val} or {primary_dim_val + extras_dim_val} when extras are supervised."
                            )
                    else:
                        if y_val_flat.shape[1] == primary_dim_val + extras_dim_val:
                            primary_val = y_val_flat[:, :primary_dim_val]
                            extras_val_arr = y_val_flat[:, primary_dim_val:]
                        else:
                            raise ValueError("validation_data must include extras targets (appended to y or provided as a third element) when extras are supervised.")
                    y_val_full = np.concatenate(
                        [primary_val.astype(np.float32, copy=False), extras_val_arr.astype(np.float32, copy=False)],
                        axis=1,
                    )
                else:
                    y_val_full = y_val_flat
                y_val_t = torch.from_numpy(y_val_full.astype(np.float32, copy=False)).to(device)
        # Optional input noise
        noise_std_t: Optional[torch.Tensor] = None
        device = self._device()
        if noisy is not None:
            if self.preserve_shape:
                internal_shape = self._internal_input_shape_cf_
                if np.isscalar(noisy):
                    std = np.full((1, *internal_shape), float(noisy), dtype=np.float32)
                else:
                    arr = np.asarray(noisy, dtype=np.float32)
                    if tuple(arr.shape) == internal_shape:
                        std = arr.reshape(1, *internal_shape)
                    elif tuple(arr.shape) == self.input_shape_ and self.data_format == "channels_last":
                        std = np.moveaxis(arr, -1, 0).reshape(1, *internal_shape)
                    elif arr.ndim == 1 and arr.size == int(np.prod(internal_shape)):
                        std = arr.reshape(1, *internal_shape)
                    else:
                        raise ValueError(
                            f"noisy shape {arr.shape} not compatible with input shape {self.input_shape_}"
                        )
                noise_std_t = torch.from_numpy(std).to(device)
            else:
                n_features = int(np.prod(self.input_shape_))
                if np.isscalar(noisy):
                    std = np.full((1, n_features), float(noisy), dtype=np.float32)
                else:
                    arr = np.asarray(noisy, dtype=np.float32)
                    if arr.ndim == 1 and arr.shape[0] == n_features:
                        std = arr.reshape(1, -1)
                    elif tuple(arr.shape) == self.input_shape_:
                        std = arr.reshape(1, -1)
                    else:
                        raise ValueError(
                            f"noisy shape {arr.shape} not compatible with input shape {self.input_shape_} or flattened size {n_features}"
                        )
                noise_std_t = torch.from_numpy(std).to(device)

        if (lr_max is None) ^ (lr_min is None):
            raise ValueError("Provide both lr_max and lr_min, or neither.")
        if lr_max is not None and lr_min is not None and float(lr_max) < float(lr_min):
            warnings.warn("lr_max < lr_min; swapping to ensure non-increasing schedule.")
            lr_max, lr_min = lr_min, lr_max

        val_inputs = None
        val_targets = None
        if X_val_t is not None and y_val_t is not None:
            if self.lsm is not None and not self.preserve_shape:
                with torch.no_grad():
                    if self.lsm_train and lsm_model is not None:
                        val_inputs = lsm_model(X_val_t)
                    elif hasattr(self.lsm, 'transform'):
                        arr = self.lsm.transform(X_val_t.cpu().numpy())
                        val_inputs = torch.from_numpy(arr.astype(np.float32, copy=False)).to(device)
                    elif hasattr(self.lsm, 'forward'):
                        val_inputs = self.lsm(X_val_t)
                    else:
                        val_inputs = X_val_t
            else:
                val_inputs = X_val_t
            val_targets = y_val_t

        cfg_loop = TrainingLoopConfig(
            epochs=int(self.epochs),
            patience=int(self.patience),
            early_stopping=bool(self.early_stopping),
            stateful=bool(self.stateful),
            state_reset=str(self.state_reset),
            verbose=int(verbose),
            lr_max=None if lr_max is None else float(lr_max),
            lr_min=None if lr_min is None else float(lr_min),
        )

        _, best_state = run_training_loop(
            self.model_,
            optimizer=opt,
            loss_fn=loss_fn,
            train_loader=dl,
            device=device,
            cfg=cfg_loop,
            noise_std=noise_std_t,
            val_inputs=val_inputs,
            val_targets=val_targets,
        )

        if best_state is not None and self.early_stopping:
            self.model_.load_state_dict(best_state)
        return self












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
        loss: Any = "mse",
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
        state: Optional[Dict[str, Any]] = None,
        state_reset: str = "batch",
        stream_lr: Optional[float] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        lsm: Optional[Any] = None,
        lsm_train: bool = False,
        lsm_pretrain_epochs: int = 0,
        lsm_lr: Optional[float] = None,
        extras: int = 0,
        extras_growth: Optional[Any] = None,
        warm_start: bool = False,
        scaler: Optional[Union[str, Any]] = None,
        scaler_params: Optional[Dict[str, Any]] = None,
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
            extras=extras,
            extras_growth=extras_growth,
            warm_start=warm_start,
            scaler=scaler,
            scaler_params=scaler_params,
            w0_first=w0_first,
            w0_hidden=w0_hidden,
            norm=norm,
            drop_path_max=drop_path_max,
            residual_alpha_init=residual_alpha_init,
        )
        if not self.preserve_shape:
            warnings.warn(
                "ResConvPSANNRegressor is intended for preserve_shape=True; falling back to residual MLP core.",
                UserWarning,
                stacklevel=2,
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        *,
        validation_data: Optional[tuple] = None,
        verbose: int = 0,
        noisy: Optional[float | np.ndarray] = None,
        extras_targets: Optional[np.ndarray] = None,
        extras_loss_weight: Optional[float] = None,
        extras_loss_mode: Optional[str] = None,
        extras_loss_cycle: Optional[int] = None,
        hisso: bool = False,
        hisso_window: Optional[int] = None,
        hisso_reward_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        hisso_context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        hisso_trans_cost: Optional[float] = None,
        hisso_supervised: Optional[Dict[str, Any]] = None,
        lr_max: Optional[float] = None,
        lr_min: Optional[float] = None,
        hisso_extras_weight: Optional[float] = None,
        hisso_extras_mode: Optional[str] = None,
        hisso_extras_cycle: Optional[int] = None,
    ):
        if not self.preserve_shape:
            return super().fit(
                X,
                y,
                validation_data=validation_data,
                verbose=verbose,
                noisy=noisy,
                extras_targets=extras_targets,
                extras_loss_weight=extras_loss_weight,
                extras_loss_mode=extras_loss_mode,
                extras_loss_cycle=extras_loss_cycle,
                hisso=hisso,
                hisso_window=hisso_window,
                hisso_reward_fn=hisso_reward_fn,
                hisso_context_extractor=hisso_context_extractor,
                hisso_trans_cost=hisso_trans_cost,
                hisso_supervised=hisso_supervised,
                lr_max=lr_max,
                lr_min=lr_min,
                hisso_extras_weight=hisso_extras_weight,
                hisso_extras_mode=hisso_extras_mode,
                hisso_extras_cycle=hisso_extras_cycle,
            )

        seed_all(self.random_state)

        val_extras = None
        if validation_data is not None:
            if not isinstance(validation_data, (tuple, list)):
                raise ValueError("validation_data must be a tuple (X, y) or (X, y, extras)")
            val_tuple = tuple(validation_data)
            if len(val_tuple) == 3:
                validation_data = (val_tuple[0], val_tuple[1])
                val_extras = val_tuple[2]
            elif len(val_tuple) == 2:
                validation_data = (val_tuple[0], val_tuple[1])
            else:
                raise ValueError("validation_data must be length 2 or 3")

        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y, dtype=np.float32)
        if not hisso and y is None:
            raise ValueError("y must be provided when hisso=False")
        if hisso and (
            extras_targets is not None
            or extras_loss_weight is not None
            or extras_loss_mode is not None
            or extras_loss_cycle is not None
        ):
            raise ValueError("extras_targets/extras loss parameters are only supported when hisso=False")

        extras_cfg = self.get_extras_growth()
        extras_dim = max(0, int(extras_cfg.extras_dim))
        if extras_loss_weight is None and extras_cfg.loss_weight is not None:
            extras_loss_weight = extras_cfg.loss_weight
        if extras_loss_mode is None and extras_cfg.loss_mode is not None:
            extras_loss_mode = extras_cfg.loss_mode
        if extras_loss_cycle is None and extras_cfg.loss_cycle is not None:
            extras_loss_cycle = extras_cfg.loss_cycle
        extras_supervision_requested = (
            extras_targets is not None
            or extras_loss_weight is not None
            or extras_loss_mode is not None
            or extras_loss_cycle is not None
        )
        extras_info: Optional[Dict[str, object]] = None
        self._supervised_extras_meta_ = None
        if extras_supervision_requested:
            raise NotImplementedError(
                "extras_targets currently supports preserve_shape=False and per_element=False"
            )
        self.input_shape_ = self._infer_input_shape(X)

        if X.ndim < 3:
            raise ValueError("preserve_shape=True requires X with at least 3 dims (N, C, H, W)")
        if self.data_format not in {"channels_first", "channels_last"}:
            raise ValueError("data_format must be 'channels_first' or 'channels_last'")
        X_cf = np.moveaxis(X, -1, 1) if self.data_format == "channels_last" else X
        self._internal_input_shape_cf_ = tuple(X_cf.shape[1:])
        nd = X_cf.ndim - 2
        if nd != 2:
            raise ValueError("ResConvPSANNRegressor currently supports 2D inputs only")

        N, C = X_cf.shape[0], int(X_cf.shape[1])
        X2d = X_cf.reshape(N, C, -1).transpose(0, 2, 1).reshape(-1, C)
        xfm = self._scaler_fit_update(X2d)
        if xfm is not None:
            X2d_scaled = xfm(X2d)
            X_cf = X2d_scaled.reshape(N, -1, C).transpose(0, 2, 1).reshape(X_cf.shape)

        if hisso:
            if self.per_element:
                raise ValueError("hisso=True is not compatible with per_element=True for ResConvPSANNRegressor")
            primary_dim = int(np.prod(self._internal_input_shape_cf_))
            extras_dim_model = extras_dim
            out_dim = primary_dim + extras_dim_model
            X_flat = X_cf.reshape(X_cf.shape[0], -1)
            self._supervised_extras_meta_ = None

            lsm_model, lsm_channels = self._resolve_lsm_module(X_cf, preserve_shape=True)
            if lsm_model is not None and not self.lsm_train:
                for p in lsm_model.parameters():
                    p.requires_grad = False
            if lsm_model is not None:
                if lsm_channels is not None:
                    base_channels = int(lsm_channels)
                elif hasattr(lsm_model, "out_channels"):
                    base_channels = int(getattr(lsm_model, "out_channels"))
                else:
                    with torch.no_grad():
                        sample = torch.from_numpy(X_cf[:1]).to(torch.float32)
                        base_channels = int(lsm_model(sample).shape[1])
            else:
                base_channels = int(self._internal_input_shape_cf_[0])

            core_in = base_channels + extras_dim_model
            core_model = ResidualPSANNConv2dNet(
                core_in,
                out_dim,
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
                segmentation_head=False,
            )
            preproc = _ConvHISSOAdapter(self._internal_input_shape_cf_, extras_dim_model, base_module=lsm_model)
            self.model_ = WithPreprocessor(preproc, core_model)
            device = self._device()
            self.model_.to(device)

            warm_cfg = coerce_warmstart_config(hisso_supervised, y)
            if warm_cfg is not None:
                run_hisso_supervised_warmstart(
                    self,
                    X_flat,
                    primary_dim=primary_dim,
                    extras_dim=extras_dim_model,
                    config=warm_cfg,
                    lsm_module=lsm_model,
                )

            extras_weight = float(hisso_extras_weight) if hisso_extras_weight is not None else 0.0
            extras_mode = (
                hisso_extras_mode.lower().strip()
                if hisso_extras_mode is not None
                else ("alternate" if extras_weight > 0.0 else "joint")
            )
            extras_cycle = max(1, int(hisso_extras_cycle) if hisso_extras_cycle is not None else 2)
            trainer_cfg = HISSOTrainerConfig(
                episode_length=int(hisso_window if hisso_window is not None else 64),
                episodes_per_batch=32,
                primary_dim=primary_dim,
                extras_dim=extras_dim_model,
                primary_transform="softmax",
                extras_transform="tanh",
                random_state=self.random_state,
                transition_cost=float(hisso_trans_cost) if hisso_trans_cost is not None else 0.0,
                extras_supervision_weight=extras_weight,
                extras_supervision_mode=extras_mode,
                extras_supervision_cycle=int(extras_cycle),
            )

            noise_std = None
            if noisy is not None:
                noise_std = float(noisy) if np.isscalar(noisy) else None

            trainer = run_hisso_training(
                self,
                X_flat,
                trainer_cfg=trainer_cfg,
                lr=float(self.lr),
                device=device,
                reward_fn=hisso_reward_fn,
                context_extractor=hisso_context_extractor,
                extras_cache=self._extras_cache_,
                lr_max=lr_max,
                lr_min=lr_min,
                input_noise_std=noise_std,
                verbose=int(verbose),
                trainer=getattr(self, "_hisso_trainer_", None),
            )
            self._hisso_trainer_ = trainer
            self._set_extras_cache(getattr(trainer, "extras_cache", None))
            self._hisso_cfg_ = trainer_cfg
            self._hisso_trained_ = True
            self.history_ = getattr(trainer, "history", [])
            return self

        if y is None:
            raise ValueError("y must be provided for supervised training")

        if self.per_element:
            if y.ndim < X.ndim - 1:
                raise ValueError("per_element=True expects y to include spatial dims")
            if self.data_format == "channels_last":
                if y.ndim == X.ndim:
                    y_cf = np.moveaxis(y, -1, 1)
                elif y.ndim == X.ndim - 1:
                    y_cf = y[:, None, ...]
                else:
                    raise ValueError("y must match X spatial dims for channels_last")
            else:
                if y.ndim == X_cf.ndim:
                    y_cf = y
                elif y.ndim == X_cf.ndim - 1:
                    y_cf = y[:, None, ...]
                else:
                    raise ValueError("y must match X spatial dims for channels_first")
            n_targets = int(y_cf.shape[1])
        else:
            y_vec = y.reshape(y.shape[0], -1) if y.ndim > 2 else (y[:, None] if y.ndim == 1 else y)
            if self.output_shape is not None:
                n_targets = int(np.prod(self.output_shape))
                if y_vec.shape[1] != n_targets:
                    raise ValueError(
                        f"y has {y_vec.shape[1]} targets, expected {n_targets} from output_shape"
                    )
            else:
                n_targets = int(y_vec.shape[1])
            y_cf = y_vec

        lsm_model, lsm_channels = self._resolve_lsm_module(X_cf, preserve_shape=True)
        in_channels = int(lsm_channels) if lsm_channels is not None else int(X_cf.shape[1])
        if lsm_model is not None and not self.lsm_train:
            for p in lsm_model.parameters():
                p.requires_grad = False

        rebuild = True
        if self.warm_start and hasattr(self, "model_") and isinstance(self.model_, nn.Module):
            rebuild = False
        if rebuild:
            core_model = ResidualPSANNConv2dNet(
                in_channels,
                n_targets,
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
                segmentation_head=self.per_element,
            )
            preproc = lsm_model if lsm_model is not None else None
            self.model_ = WithPreprocessor(preproc, core_model)
        X_train_arr = X_cf.astype(np.float32, copy=False)

        device = self._device()
        self.model_.to(device)

        if self.lsm_train and lsm_model is not None:
            params = [
                {"params": self.model_.core.parameters(), "lr": self.lr},
                {
                    "params": self.model_.preproc.parameters(),
                    "lr": float(self.lsm_lr) if self.lsm_lr is not None else self.lr,
                },
            ]
            if self.optimizer.lower() == "adamw":
                opt = torch.optim.AdamW(params, weight_decay=self.weight_decay)
            elif self.optimizer.lower() == "sgd":
                opt = torch.optim.SGD(params, momentum=0.9)
            else:
                opt = torch.optim.Adam(params, weight_decay=self.weight_decay)
        else:
            opt = self._make_optimizer(self.model_)

        loss_fn = self._make_loss()

        targets_np = np.asarray(y_cf, dtype=np.float32)
        inputs_np = np.asarray(X_train_arr, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(inputs_np), torch.from_numpy(targets_np))
        shuffle_batches = True
        if self.stateful and self.state_reset in ("epoch", "none"):
            shuffle_batches = False
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle_batches, num_workers=self.num_workers)

        X_val_t = y_val_t = None
        if validation_data is not None:
            Xv, yv = validation_data
            Xv = np.asarray(Xv, dtype=np.float32)
            yv = np.asarray(yv, dtype=np.float32)
            Xv_cf = np.moveaxis(Xv, -1, 1) if self.data_format == "channels_last" else Xv
            if Xv_cf.shape[1] != self._internal_input_shape_cf_[0]:
                raise ValueError("validation_data channels mismatch.")
            X_val_t = torch.from_numpy(Xv_cf.astype(np.float32, copy=False)).to(device)
            if self.per_element:
                yv_cf = np.moveaxis(yv, -1, 1) if self.data_format == "channels_last" else yv
            else:
                yv_cf = yv.reshape(yv.shape[0], -1) if yv.ndim > 2 else (yv[:, None] if yv.ndim == 1 else yv)
            y_val_t = torch.from_numpy(yv_cf.astype(np.float32, copy=False)).to(device)
            if val_extras is not None:
                self._set_extras_cache(val_extras)

        noise_std_t: Optional[torch.Tensor] = None
        if noisy is not None:
            internal_shape = self._internal_input_shape_cf_
            if np.isscalar(noisy):
                std = np.full((1, *internal_shape), float(noisy), dtype=np.float32)
            else:
                arr = np.asarray(noisy, dtype=np.float32)
                if tuple(arr.shape) == internal_shape:
                    std = arr.reshape(1, *internal_shape)
                elif tuple(arr.shape) == self.input_shape_ and self.data_format == "channels_last":
                    std = np.moveaxis(arr, -1, 0).reshape(1, *internal_shape)
                elif arr.ndim == 1 and arr.size == int(np.prod(internal_shape)):
                    std = arr.reshape(1, *internal_shape)
                else:
                    raise ValueError(
                        f"noisy shape {arr.shape} not compatible with input shape {self.input_shape_}"
                    )
            noise_std_t = torch.from_numpy(std).to(device)

        cfg_loop = TrainingLoopConfig(
            epochs=int(self.epochs),
            patience=int(self.patience),
            early_stopping=bool(self.early_stopping),
            stateful=bool(self.stateful),
            state_reset=str(self.state_reset),
            verbose=int(verbose),
            lr_max=None if lr_max is None else float(lr_max),
            lr_min=None if lr_min is None else float(lr_min),
        )

        history, best_state = run_training_loop(
            self.model_,
            optimizer=opt,
            loss_fn=loss_fn,
            train_loader=dl,
            device=device,
            cfg=cfg_loop,
            noise_std=noise_std_t,
            val_inputs=X_val_t,
            val_targets=y_val_t,
        )

        self.history_ = history
        if best_state is not None and self.early_stopping:
            self.model_.load_state_dict(best_state)
        return self
