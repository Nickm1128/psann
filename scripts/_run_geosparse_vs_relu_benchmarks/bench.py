# ruff: noqa: F403,F405
from __future__ import annotations

from .data import DatasetSplit, _compute_epochs, _maybe_cuda_sync, _subsample
from .shared import *


def _attach_progress(model: Any, steps_per_epoch: int, progress_every_steps: int) -> None:
    next_report = {"step": progress_every_steps}

    def _epoch_callback(
        self, epoch: int, train_loss: float, val_loss: Optional[float], *_: Any
    ) -> None:
        steps_done = (epoch + 1) * steps_per_epoch
        if steps_done >= next_report["step"]:
            msg = f"[step ~{steps_done}] train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            print(msg, flush=True)
            next_report["step"] += progress_every_steps

    model.epoch_callback = _epoch_callback.__get__(model, model.__class__)


def warmup_fit(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    warmup_samples: int,
    warmup_epochs: int,
) -> None:
    if not torch.cuda.is_available():
        return
    X_w, y_w = _subsample(X_train, y_train, warmup_samples, seed=123)
    model = model_factory(epochs=warmup_epochs)
    _maybe_cuda_sync()
    model.fit(X_w, y_w, validation_data=(X_val, y_val), verbose=0)
    _maybe_cuda_sync()


def fit_with_timing(
    model_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    target_steps: int,
    progress_every_steps: int,
    timing_warmup_epochs: int,
) -> Dict[str, Any]:
    epochs, steps_per_epoch = _compute_epochs(len(X_train), batch_size, target_steps)
    model = model_factory(epochs=epochs)

    warmup_epochs = int(max(0, timing_warmup_epochs))
    warmup_epochs = int(min(warmup_epochs, max(0, epochs - 1)))

    timing: Dict[str, Optional[float]] = {"start": None, "end": None}
    next_report = {"step": progress_every_steps}

    def _epoch_callback(
        self, epoch: int, train_loss: float, val_loss: Optional[float], *_: Any
    ) -> None:
        steps_done = (epoch + 1) * steps_per_epoch
        if steps_done >= next_report["step"]:
            msg = f"[step ~{steps_done}] train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f" val_loss={val_loss:.4f}"
            print(msg, flush=True)
            next_report["step"] += progress_every_steps

        if warmup_epochs > 0 and timing["start"] is None and epoch + 1 == warmup_epochs:
            _maybe_cuda_sync()
            timing["start"] = time.perf_counter()
        if epoch + 1 == epochs:
            _maybe_cuda_sync()
            timing["end"] = time.perf_counter()

    model.epoch_callback = _epoch_callback.__get__(model, model.__class__)
    _maybe_cuda_sync()
    start_total = time.perf_counter()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0)
    _maybe_cuda_sync()
    end_total = time.perf_counter()
    elapsed_total = end_total - start_total

    elapsed = elapsed_total
    if timing["start"] is not None and timing["end"] is not None:
        elapsed = max(0.0, float(timing["end"] - timing["start"]))
    total_steps = steps_per_epoch * epochs
    timed_epochs = epochs - warmup_epochs if elapsed != elapsed_total else epochs
    samples_seen = len(X_train) * timed_epochs
    timed_steps = steps_per_epoch * timed_epochs
    return {
        "model": model,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "timed_steps": timed_steps,
        "train_time_total_s": elapsed_total,
        "train_time_s": elapsed,
        "steps_per_sec": timed_steps / max(elapsed, 1e-9),
        "samples_per_sec": samples_seen / max(elapsed, 1e-9),
    }


def _unscale_y(y: np.ndarray, y_scaler: Optional[StandardScaler]) -> np.ndarray:
    if y_scaler is None:
        return y
    return y_scaler.inverse_transform(y.reshape(-1, 1)).ravel()


def evaluate_regression(model: Any, split: DatasetSplit) -> Dict[str, float]:
    y_true = _unscale_y(split.y_test, split.y_scaler)
    y_pred = np.asarray(model.predict(split.X_test), dtype=np.float32)
    y_pred = _unscale_y(y_pred, split.y_scaler)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
