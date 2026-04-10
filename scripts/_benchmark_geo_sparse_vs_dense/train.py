# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


def _build_loader(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
    )


def _set_precision(tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    if tf32:
        torch.set_float32_matmul_precision("high")


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).lower()
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def _make_autocast_context(device: torch.device, amp: bool, amp_dtype: torch.dtype):
    if not amp:
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)


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


def _train_model(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    amp: bool,
    amp_dtype: torch.dtype,
    compile_model: bool,
    compile_backend: Optional[str],
    compile_mode: Optional[str],
    timing_warmup_steps: int,
    timing_epochs: int,
    curve_every: int = 1,
    curve_test: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    curve_test_unscaled: Optional[np.ndarray] = None,
    curve_test_y_scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    loss_fn = nn.MSELoss()

    compile_time_s = None
    if compile_model:
        model = torch.compile(model, backend=compile_backend, mode=compile_mode)
        first_batch = next(iter(train_loader))
        xb, yb = first_batch
        xb = xb.to(device=device)
        yb = yb.to(device=device)
        compile_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with _make_autocast_context(device, amp, amp_dtype):
            pred = model(xb)
            loss = loss_fn(pred, yb)
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        compile_time_s = time.perf_counter() - compile_start
        optimizer.zero_grad(set_to_none=True)

    use_scaler = bool(amp and amp_dtype == torch.float16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    curve: List[Dict[str, Any]] = []
    step_events = []
    total_samples = 0
    epoch_times: List[float] = []
    grad_norms: List[float] = []
    loss_nonfinite = 0
    grad_nonfinite = 0
    start = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_samples = 0
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device=device)
            yb = yb.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            record_step = epoch < timing_epochs and step >= timing_warmup_steps
            if record_step and device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            with _make_autocast_context(device, amp, amp_dtype):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                loss_nonfinite += 1
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
                grad_norm = _grad_norm(model)
                if grad_norm is not None:
                    grad_norms.append(grad_norm)
                    if not math.isfinite(grad_norm):
                        grad_nonfinite += 1
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = _grad_norm(model)
                if grad_norm is not None:
                    grad_norms.append(grad_norm)
                    if not math.isfinite(grad_norm):
                        grad_nonfinite += 1
                optimizer.step()
            if record_step and device.type == "cuda":
                end_event.record()
                step_events.append((start_event, end_event))
            bs = int(xb.shape[0])
            epoch_loss_sum += loss_value * bs
            epoch_samples += bs
            total_samples += int(xb.shape[0])
        if curve_every > 0 and ((epoch + 1) % int(curve_every) == 0):
            train_mse = epoch_loss_sum / float(max(1, epoch_samples))
            point: Dict[str, Any] = {"epoch": int(epoch + 1), "train_mse": float(train_mse)}
            if curve_test is not None:
                X_test, y_test = curve_test
                point["test_mse"] = _evaluate(
                    model,
                    X=X_test,
                    y=y_test,
                    device=device,
                    y_scaler=curve_test_y_scaler,
                    y_unscaled=curve_test_unscaled,
                )
            curve.append(point)
        epoch_times.append(time.perf_counter() - epoch_start)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - start

    step_time_ms = None
    if step_events:
        step_time_ms = float(np.mean([s.elapsed_time(e) for s, e in step_events]))

    return {
        "train_time_s": float(wall_time_s),
        "compile_time_s": float(compile_time_s) if compile_time_s is not None else None,
        "step_time_ms_mean": step_time_ms,
        "samples_per_sec": float(total_samples) / float(wall_time_s) if wall_time_s > 0 else None,
        "epoch_time_s_mean": float(sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        "grad_norm_max": float(max(grad_norms)) if grad_norms else None,
        "loss_nonfinite_steps": int(loss_nonfinite),
        "grad_nonfinite_steps": int(grad_nonfinite),
        "curve": curve,
    }


def _evaluate(
    model: nn.Module,
    *,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    y_scaler: Optional[StandardScaler] = None,
    y_unscaled: Optional[np.ndarray] = None,
) -> float:
    model.eval()
    X_t = torch.from_numpy(X).to(device=device)
    with torch.no_grad():
        pred = model(X_t)

    if y_scaler is None:
        y_t = torch.from_numpy(y).to(device=device)
        return float(torch.mean((pred - y_t) ** 2).item())

    if y_unscaled is None:
        y_unscaled = y_scaler.inverse_transform(y)

    y_t = torch.from_numpy(y_unscaled).to(device=device, dtype=torch.float32)
    mean_t = torch.from_numpy(y_scaler.mean).to(device=device, dtype=torch.float32)
    scale_t = torch.from_numpy(y_scaler.scale).to(device=device, dtype=torch.float32)
    pred_unscaled = pred.to(dtype=torch.float32) * scale_t + mean_t
    return float(torch.mean((pred_unscaled - y_t) ** 2).item())


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + float(eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.reshape(y_true.shape[0], -1)
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "smape": _smape(y_true, y_pred),
        "r2": _r2_score_np(y_true, y_pred),
    }


def _predict_unscaled(
    model: nn.Module,
    *,
    X: np.ndarray,
    device: torch.device,
    y_scaler: Optional[StandardScaler],
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device=device))
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    if y_scaler is None:
        return pred_np
    return (pred_np * y_scaler.scale + y_scaler.mean).astype(np.float32)


def _evaluate_metrics(
    model: nn.Module,
    *,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    y_scaler: Optional[StandardScaler],
    y_unscaled: Optional[np.ndarray],
) -> Dict[str, float]:
    y_true = y_unscaled if y_unscaled is not None else y
    pred = _predict_unscaled(model, X=X, device=device, y_scaler=y_scaler)
    return _regression_metrics(y_true, pred)


def _flatten_dict(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict):
            for sub_k, sub_v in value.items():
                flat[f"{prefix}{key}_{sub_k}"] = sub_v
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _write_summary(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def _get_env_info(device: torch.device) -> Dict[str, Any]:
    info = gather_env_info()
    info["selected_device"] = str(device)
    info["device_type"] = device.type
    return info
