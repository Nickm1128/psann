from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Union

import numpy as np
import torch

from .._aliases import resolve_int_alias


class _PSANNRegressorScalingMixin:
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

    def _ensure_model_device(self, device: torch.device) -> None:
        model = getattr(self, "model_", None)
        if model is None:
            return
        current = getattr(self, "_model_device_", None)
        if current == device:
            return
        model.to(device)
        self._model_device_ = device

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

    def _make_internal_target_scaler(self) -> Optional[Dict[str, Any]]:
        kind = self.target_scaler
        if kind is None:
            return None
        if isinstance(kind, str):
            key = kind.lower()
            if key not in {"standard", "minmax"}:
                raise ValueError(
                    "Unsupported target_scaler string. Use 'standard', 'minmax', or provide an object with fit/transform."
                )
            return {"type": key, "state": {}}
        # Custom object: must implement fit/transform; inverse_transform optional
        if not hasattr(kind, "fit") or not hasattr(kind, "transform"):
            raise ValueError(
                "Custom target_scaler must implement fit(X) and transform(X). Optional inverse_transform(X)."
            )
        return {"type": "custom", "obj": kind}

    def _target_scaler_fit_update(
        self, y2d: np.ndarray
    ) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        """Fit or update the target scaler on a 2D array and return a transform function."""
        if self.target_scaler is None:
            self._target_scaler_kind_ = None
            self._target_scaler_state_ = None
            return None
        spec = getattr(self, "_target_scaler_spec_", None)
        if spec is None:
            spec = self._make_internal_target_scaler()
            self._target_scaler_spec_ = spec
        if spec is None:
            self._target_scaler_kind_ = None
            self._target_scaler_state_ = None
            return None

        if spec.get("type") == "standard":
            self._target_scaler_kind_ = "standard"
            st = self._target_scaler_state_ or {"n": 0, "mean": None, "M2": None}
            n0 = int(st["n"])
            y_arr = np.asarray(y2d, dtype=np.float32)
            if n0 == 0:
                mean = y_arr.mean(axis=0)
                diff = y_arr - mean
                M2 = (diff * diff).sum(axis=0)
                n = y_arr.shape[0]
            else:
                mean0 = st["mean"]
                M20 = st["M2"]
                n = n0 + y_arr.shape[0]
                delta = y_arr.mean(axis=0) - mean0
                mean = (mean0 * n0 + y_arr.sum(axis=0)) / n
                M2a = M20
                xa = n0
                xb = y_arr.shape[0]
                y_centered = y_arr - y_arr.mean(axis=0)
                M2b = (y_centered * y_centered).sum(axis=0)
                M2 = M2a + M2b + (delta * delta) * (xa * xb) / max(n, 1)
            self._target_scaler_state_ = {"n": int(n), "mean": mean, "M2": M2}

            def _xfm(Z: np.ndarray) -> np.ndarray:
                st2 = self._target_scaler_state_
                assert st2 is not None
                mean2 = st2["mean"]
                var = st2["M2"] / max(st2["n"], 1)
                std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
                return (Z - mean2) / std

            return _xfm

        if spec.get("type") == "minmax":
            self._target_scaler_kind_ = "minmax"
            st = self._target_scaler_state_ or {"min": None, "max": None}
            y_arr = np.asarray(y2d, dtype=np.float32)
            mn = (
                y_arr.min(axis=0) if st["min"] is None else np.minimum(st["min"], y_arr.min(axis=0))
            )
            mx = (
                y_arr.max(axis=0) if st["max"] is None else np.maximum(st["max"], y_arr.max(axis=0))
            )
            self._target_scaler_state_ = {"min": mn, "max": mx}

            def _xfm(Z: np.ndarray) -> np.ndarray:
                st2 = self._target_scaler_state_
                assert st2 is not None
                mn2 = st2["min"]
                mx2 = st2["max"]
                scale = np.where((mx2 - mn2) > 1e-8, (mx2 - mn2), 1.0)
                return (Z - mn2) / scale

            return _xfm

        obj = spec.get("obj")
        self._target_scaler_kind_ = "custom"
        if not hasattr(self, "_target_scaler_fitted_") or not getattr(
            self, "_target_scaler_fitted_", False
        ):
            try:
                obj.fit(y2d, **(self.target_scaler_params or {}))
            except TypeError:
                obj.fit(y2d)
            self._target_scaler_fitted_ = True
        else:
            if hasattr(obj, "partial_fit"):
                obj.partial_fit(y2d)
            else:
                pass

        def _xfm(Z: np.ndarray) -> np.ndarray:
            return obj.transform(Z)

        return _xfm

    def _apply_fitted_target_scaler(self, y2d: np.ndarray) -> np.ndarray:
        kind = getattr(self, "_target_scaler_kind_", None)
        if kind is None:
            return y2d.astype(np.float32, copy=False)
        st = getattr(self, "_target_scaler_state_", None)
        if kind == "standard" and st is not None:
            n = max(int(st.get("n", 0)), 1)
            mean = np.asarray(st.get("mean"), dtype=np.float32)
            var = np.asarray(st.get("M2"), dtype=np.float32) / float(n)
            std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32, copy=False)
            return ((y2d - mean) / std).astype(np.float32, copy=False)
        if kind == "minmax" and st is not None:
            mn = np.asarray(st.get("min"), dtype=np.float32)
            mx = np.asarray(st.get("max"), dtype=np.float32)
            scale = np.where((mx - mn) > 1e-8, (mx - mn), 1.0).astype(np.float32, copy=False)
            return ((y2d - mn) / scale).astype(np.float32, copy=False)
        if (
            kind == "custom"
            and hasattr(self, "target_scaler")
            and hasattr(self.target_scaler, "transform")
        ):
            transformed = self.target_scaler.transform(y2d)
            return np.asarray(transformed, dtype=np.float32)
        return y2d.astype(np.float32, copy=False)

    def _inverse_fitted_target_scaler(self, y2d: np.ndarray) -> np.ndarray:
        kind = getattr(self, "_target_scaler_kind_", None)
        st = getattr(self, "_target_scaler_state_", None)
        if kind is None:
            return y2d.astype(np.float32, copy=False)
        if kind == "standard" and st is not None:
            mean = np.asarray(st.get("mean"), dtype=np.float32)
            n = max(int(st.get("n", 0)), 1)
            var = np.asarray(st.get("M2"), dtype=np.float32) / float(n)
            std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32, copy=False)
            return (y2d * std + mean).astype(np.float32, copy=False)
        if kind == "minmax" and st is not None:
            mn = np.asarray(st.get("min"), dtype=np.float32)
            mx = np.asarray(st.get("max"), dtype=np.float32)
            scale = np.where((mx - mn) > 1e-8, (mx - mn), 1.0).astype(np.float32, copy=False)
            return (y2d * scale + mn).astype(np.float32, copy=False)
        if kind == "custom" and hasattr(self.target_scaler, "inverse_transform"):
            inv = self.target_scaler.inverse_transform(y2d)
            return np.asarray(inv, dtype=np.float32)
        return y2d.astype(np.float32, copy=False)

    def _apply_fitted_target_scaler_like(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float32)
        if getattr(self, "_target_scaler_kind_", None) is None:
            return y_arr.astype(np.float32, copy=False)
        orig_shape = y_arr.shape

        if self.preserve_shape and self.per_element:
            target_cf = getattr(self, "_target_cf_shape_", None)
            if target_cf is not None and y_arr.size == int(y_arr.shape[0]) * int(
                np.prod(target_cf)
            ):
                y_cf = y_arr.reshape((y_arr.shape[0],) + tuple(target_cf))
                n, n_targets = int(y_cf.shape[0]), int(y_cf.shape[1])
                y2d = y_cf.reshape(n, n_targets, -1).transpose(0, 2, 1).reshape(-1, n_targets)
                y2d = self._apply_fitted_target_scaler(y2d)
                y_cf = y2d.reshape(n, -1, n_targets).transpose(0, 2, 1).reshape(y_cf.shape)
                return y_cf.reshape(orig_shape).astype(np.float32, copy=False)

        if y_arr.ndim == 0:
            y2d = y_arr.reshape(1, 1)
            y2d = self._apply_fitted_target_scaler(y2d)
            return y2d.reshape(orig_shape).astype(np.float32, copy=False)
        if y_arr.ndim == 1:
            y2d = y_arr.reshape(1, -1)
            y2d = self._apply_fitted_target_scaler(y2d)
            return y2d.reshape(orig_shape).astype(np.float32, copy=False)
        y2d = y_arr.reshape(int(y_arr.shape[0]), -1)
        y2d = self._apply_fitted_target_scaler(y2d)
        return y2d.reshape(orig_shape).astype(np.float32, copy=False)

    def _inverse_fitted_target_scaler_like(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float32)
        if getattr(self, "_target_scaler_kind_", None) is None:
            return y_arr.astype(np.float32, copy=False)
        orig_shape = y_arr.shape

        if self.preserve_shape and self.per_element:
            target_cf = getattr(self, "_target_cf_shape_", None)
            if target_cf is not None and y_arr.size == int(y_arr.shape[0]) * int(
                np.prod(target_cf)
            ):
                y_cf = y_arr.reshape((y_arr.shape[0],) + tuple(target_cf))
                n, n_targets = int(y_cf.shape[0]), int(y_cf.shape[1])
                y2d = y_cf.reshape(n, n_targets, -1).transpose(0, 2, 1).reshape(-1, n_targets)
                y2d = self._inverse_fitted_target_scaler(y2d)
                y_cf = y2d.reshape(n, -1, n_targets).transpose(0, 2, 1).reshape(y_cf.shape)
                return y_cf.reshape(orig_shape).astype(np.float32, copy=False)

        if y_arr.ndim == 0:
            y2d = y_arr.reshape(1, 1)
            y2d = self._inverse_fitted_target_scaler(y2d)
            return y2d.reshape(orig_shape).astype(np.float32, copy=False)
        if y_arr.ndim == 1:
            y2d = y_arr.reshape(1, -1)
            y2d = self._inverse_fitted_target_scaler(y2d)
            return y2d.reshape(orig_shape).astype(np.float32, copy=False)
        y2d = y_arr.reshape(int(y_arr.shape[0]), -1)
        y2d = self._inverse_fitted_target_scaler(y2d)
        return y2d.reshape(orig_shape).astype(np.float32, copy=False)

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
        if not hasattr(self, "model_") or self.model_ is None:
            raise RuntimeError(
                "Estimator is not fitted yet. Call fit(X, y) before predict/score/save."
            )


__all__ = ["_PSANNRegressorScalingMixin"]
