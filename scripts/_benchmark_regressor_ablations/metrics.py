# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _coerce_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    u = ((y_true - y_pred) ** 2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - (u / v if v != 0 else np.nan))


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    rmse = math.sqrt(float(np.mean((y_pred - y_true) ** 2)))
    span = float(y_true.max() - y_true.min())
    if span == 0:
        return float("nan")
    return float(rmse / span)


def _mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    numerator = float(np.mean(np.abs(y_pred - y_true)))
    diffs = np.abs(np.diff(y_true, axis=0))
    denom = float(np.mean(diffs))
    if denom == 0:
        return float("nan")
    return float(numerator / denom)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = _coerce_2d(y_true)
    y_pred = _coerce_2d(y_pred)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = math.sqrt(mse)
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": _r2_score(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
        "nrmse": _nrmse(y_true, y_pred),
        "mase": _mase(y_true, y_pred),
    }


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        if tp + fp == 0 or tp + fn == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def _classification_metrics(
    y_true_labels: np.ndarray,
    y_pred_scores: np.ndarray,
    *,
    num_classes: int,
) -> Dict[str, float]:
    y_pred_scores = _coerce_2d(y_pred_scores)
    y_pred = y_pred_scores.argmax(axis=1)
    acc = float(np.mean(y_pred == y_true_labels))
    return {
        "accuracy": acc,
        "macro_f1": _macro_f1(y_true_labels, y_pred, num_classes=num_classes),
    }


def _split_train_val(
    dataset: DatasetBundle,
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    if dataset.X_val is not None and dataset.y_val is not None:
        return (
            dataset.X_train,
            dataset.y_train,
            dataset.X_val,
            dataset.y_val,
            dataset.y_train_labels,
            dataset.y_val_labels,
        )

    X = dataset.X_train
    y = dataset.y_train
    n = X.shape[0]
    if n < 2 or val_fraction <= 0.0:
        return X, y, X[:0], y[:0], dataset.y_train_labels, None

    rng = np.random.default_rng(seed)
    if dataset.task == "classification" and dataset.y_train_labels is not None:
        labels = dataset.y_train_labels
        train_idx = []
        val_idx = []
        for cls in np.unique(labels):
            cls_idx = np.where(labels == cls)[0]
            rng.shuffle(cls_idx)
            n_val = max(1, int(round(len(cls_idx) * val_fraction)))
            val_idx.extend(cls_idx[:n_val])
            train_idx.extend(cls_idx[n_val:])
        train_idx = np.array(train_idx, dtype=np.int64)
        val_idx = np.array(val_idx, dtype=np.int64)
    else:
        idx = rng.permutation(n)
        n_val = max(1, int(round(n * val_fraction)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    y_train_labels = (
        dataset.y_train_labels[train_idx] if dataset.y_train_labels is not None else None
    )
    y_val_labels = dataset.y_train_labels[val_idx] if dataset.y_train_labels is not None else None
    return X_train, y_train, X_val, y_val, y_train_labels, y_val_labels


def _history_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not history:
        return {}
    train_losses = [h.get("train_loss") for h in history if h.get("train_loss") is not None]
    val_losses = [h.get("val_loss") for h in history if h.get("val_loss") is not None]
    epoch_times = [h.get("epoch_time_s") for h in history if h.get("epoch_time_s") is not None]
    step_times = [
        h.get("step_time_s_mean") for h in history if h.get("step_time_s_mean") is not None
    ]
    grad_norms = [h.get("grad_norm_max") for h in history if h.get("grad_norm_max") is not None]
    loss_nonfinite = sum(int(h.get("loss_nonfinite_steps") or 0) for h in history)
    grad_nonfinite = sum(int(h.get("grad_nonfinite_steps") or 0) for h in history)
    if train_losses:
        mean = float(sum(train_losses) / len(train_losses))
        var = float(sum((val - mean) ** 2 for val in train_losses) / len(train_losses))
        loss_vol = math.sqrt(var)
        mad = float(
            sum(abs(train_losses[i] - train_losses[i - 1]) for i in range(1, len(train_losses)))
        ) / float(max(1, len(train_losses) - 1))
    else:
        loss_vol = None
        mad = None
    if val_losses:
        mean_val = float(sum(val_losses) / len(val_losses))
        var_val = float(sum((val - mean_val) ** 2 for val in val_losses) / len(val_losses))
        val_vol = math.sqrt(var_val)
    else:
        val_vol = None
    return {
        "history_length": int(len(history)),
        "loss_curve_volatility": loss_vol,
        "loss_curve_mean_abs_diff": mad,
        "val_curve_volatility": val_vol,
        "epoch_time_s_mean": float(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        "step_time_s_mean": float(sum(step_times) / len(step_times)) if step_times else None,
        "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
        "loss_nonfinite_steps": int(loss_nonfinite),
        "grad_nonfinite_steps": int(grad_nonfinite),
        "train_time_s_total": history[-1].get("train_time_s_total"),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _count_params(estimator: Any) -> int:
    model = getattr(estimator, "model_", None)
    if model is None:
        return 0
    return int(psann_count_params(model))
