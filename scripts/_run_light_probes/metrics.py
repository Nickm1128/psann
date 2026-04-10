# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + float(eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = math.sqrt(mse)
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "smape": _smape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_y_scaler(y_train: np.ndarray) -> StandardScaler:
    y_arr = np.asarray(y_train, dtype=np.float32)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    return StandardScaler().fit(y_arr)


def _apply_y_scaler(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim == 1:
        return scaler.transform(y_arr.reshape(-1, 1)).reshape(-1).astype(np.float32, copy=False)
    return scaler.transform(y_arr).astype(np.float32, copy=False)


def _inverse_y_scaler(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    y_arr = np.asarray(y, dtype=np.float32)
    if y_arr.ndim == 1:
        return (
            scaler.inverse_transform(y_arr.reshape(-1, 1))
            .reshape(-1)
            .astype(np.float32, copy=False)
        )
    return scaler.inverse_transform(y_arr).astype(np.float32, copy=False)


def _loss_curve_volatility(losses: List[float]) -> float:
    if len(losses) < 2:
        return 0.0
    mean = float(sum(losses) / len(losses))
    var = float(sum((val - mean) ** 2 for val in losses) / len(losses))
    return float(math.sqrt(var))


def _loss_curve_mean_abs_diff(losses: List[float]) -> float:
    if len(losses) < 2:
        return 0.0
    diffs = [abs(losses[i] - losses[i - 1]) for i in range(1, len(losses))]
    return float(sum(diffs) / len(diffs))


def _grad_norm(model: nn.Module) -> Optional[float]:
    total_sq = 0.0
    has_grad = False
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        total_sq += float((grad.float() ** 2).sum().item())
        has_grad = True
    if not has_grad:
        return None
    return math.sqrt(total_sq)


def _match_mlp_hidden(
    *,
    target_params: int,
    input_dim: int,
    output_dim: int,
    depth: int,
    max_width: int = 8192,
) -> Tuple[int, int]:
    width, mismatch = match_dense_width(
        target_params=int(target_params),
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        depth=int(depth),
        max_width=int(max_width),
    )
    return int(width), int(mismatch)


def evaluate_regressor(
    model: nn.Module,
    test_X: np.ndarray,
    test_y: np.ndarray,
    device: torch.device,
    y_scaler: Optional[StandardScaler] = None,
    y_unscaled: Optional[np.ndarray] = None,
):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(test_X).float().to(device)).cpu().numpy()
    preds = preds.reshape(test_y.shape)
    if y_scaler is None:
        return _regression_metrics(test_y, preds)
    if y_unscaled is None:
        y_unscaled = _inverse_y_scaler(test_y, y_scaler)
    preds_unscaled = _inverse_y_scaler(preds, y_scaler)
    return _regression_metrics(y_unscaled, preds_unscaled)


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _history_stats(history: List[dict]) -> dict:
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
    stats = {
        "history_length": int(len(history)),
        "loss_curve_volatility": (
            _loss_curve_volatility([float(x) for x in train_losses]) if train_losses else None
        ),
        "loss_curve_mean_abs_diff": (
            _loss_curve_mean_abs_diff([float(x) for x in train_losses]) if train_losses else None
        ),
        "val_curve_volatility": (
            _loss_curve_volatility([float(x) for x in val_losses]) if val_losses else None
        ),
        "epoch_time_s_mean": float(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        "step_time_s_mean": float(sum(step_times) / len(step_times)) if step_times else None,
        "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
        "loss_nonfinite_steps": int(loss_nonfinite),
        "grad_nonfinite_steps": int(grad_nonfinite),
        "train_time_s_total": history[-1].get("train_time_s_total"),
    }
    return stats


def jacobian_pr(model: nn.Module, X_sample: np.ndarray, device: torch.device):
    model.eval()
    x = torch.from_numpy(X_sample).float().to(device)
    x.requires_grad_(True)
    y = model(x)
    grads = torch.autograd.grad(y.sum(), x, create_graph=False, retain_graph=False)[0]
    J = grads.detach().cpu().numpy().reshape(x.size(0), -1)
    try:
        s = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        M = J @ J.T
        evals, _ = np.linalg.eigh(M)
        s = np.sqrt(np.clip(evals, 0, None))[::-1]
    top_sv = float(s[0]) if s.size > 0 else 0.0
    sum_sv = float(s.sum())
    pr = float((sum_sv**2) / (np.sum(s**2) + 1e-8))
    return top_sv, sum_sv, pr
