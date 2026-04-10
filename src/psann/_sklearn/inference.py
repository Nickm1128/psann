from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .shared import _sk_r2_score


class _PSANNRegressorInferenceMixin:
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
            X_cf = (
                np.moveaxis(X_arr, -1, 1) if self.data_format == "channels_last" else X_arr.copy()
            )
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
            if self._context_dim_ is not None and self._context_dim_ not in (
                0,
                context_np.shape[1],
            ):
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
        self._ensure_model_device(device)
        model = self.model_
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

            with torch.no_grad():
                inputs_arr = (
                    inputs_np
                    if inputs_np.dtype == np.float32
                    else inputs_np.astype(np.float32, copy=False)
                )
                tensor = torch.from_numpy(inputs_arr)
                if device.type != "cpu":
                    tensor = tensor.to(device=device, dtype=torch.float32)
                context_tensor = None
                if context_np is not None:
                    ctx_arr = (
                        context_np
                        if context_np.dtype == np.float32
                        else context_np.astype(np.float32, copy=False)
                    )
                    context_tensor = torch.from_numpy(ctx_arr)
                    if device.type != "cpu":
                        context_tensor = context_tensor.to(device=device, dtype=torch.float32)
                outputs = (
                    model(tensor, context_tensor) if context_tensor is not None else model(tensor)
                )
                return outputs.detach().cpu().numpy()
        finally:
            model.train(prev_training)
            if hasattr(model, "set_state_updates"):
                model.set_state_updates(bool(prev_training))

    def predict(self, X: np.ndarray, *, context: Optional[np.ndarray] = None) -> np.ndarray:
        inputs_np, meta, context_np = self._prepare_inference_inputs(X, context)
        preds = self._run_model(inputs_np, context_np=context_np, state_updates=False)
        preds = self._inverse_fitted_target_scaler_like(preds)
        return self._reshape_predictions(preds, meta)

    def score(self, X: np.ndarray, y: np.ndarray, *, context: Optional[np.ndarray] = None) -> float:
        y_true = np.asarray(y, dtype=np.float32)
        y_pred = self.predict(X, context=context)
        if y_true.ndim == 1 and y_pred.ndim == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)
        elif y_true.shape != y_pred.shape:
            y_pred = y_pred.reshape(y_true.shape)
        return float(_sk_r2_score(y_true, y_pred))


__all__ = ["_PSANNRegressorInferenceMixin"]
