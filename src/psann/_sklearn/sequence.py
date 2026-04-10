from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch


class _PSANNRegressorSequenceMixin:
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
        preds = self._inverse_fitted_target_scaler_like(preds)
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

        arr = self._apply_fitted_target_scaler_like(arr)
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


__all__ = ["_PSANNRegressorSequenceMixin"]
