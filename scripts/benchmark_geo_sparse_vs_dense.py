#!/usr/bin/env python
"""Benchmark GeoSparseNet vs dense baselines with matched parameters."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from psann.layers.sine_residual import RMSNorm
from psann.nn_geo_sparse import GeoSparseNet, _build_activation as _build_geo_activation
from psann.params import count_params, dense_mlp_params, geo_sparse_net_params, match_dense_width
from psann.utils import choose_device, seed_all
from gpu_env_report import gather_env_info


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]


@dataclass(frozen=True)
class StandardScaler:
    mean: np.ndarray
    scale: np.ndarray
    eps: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.scale + self.mean).astype(np.float32)


def _fit_standard_scaler(x: np.ndarray, *, eps: float = 1e-6) -> StandardScaler:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    scale = x.std(axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < float(eps), 1.0, scale).astype(np.float32)
    return StandardScaler(mean=mean, scale=scale, eps=float(eps))


def _parse_shape(text: str) -> Tuple[int, int]:
    for sep in ("x", "X", ","):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    raise ValueError("shape must be formatted as HxW or H,W")


def _print_header(args: argparse.Namespace, out_dir: Path, device: torch.device) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        "[bench] start",
        f"time={ts}",
        f"device={device}",
        f"seed={args.seed}",
        f"shape={args.shape}",
        f"depth={args.depth}",
        f"k={args.k}",
        f"task={args.task}",
        f"out={out_dir}",
        flush=True,
    )
    print(
        f"[bench] torch={torch.__version__} numpy={np.__version__} tf32={args.tf32} amp={args.amp}",
        flush=True,
    )


def _build_tabular_sine(
    *,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    X = rng.uniform(-2.0, 2.0, size=(total, features)).astype(np.float32)
    noise = 0.05 * rng.standard_normal(size=(total,)).astype(np.float32)
    y = np.zeros((total,), dtype=np.float32)
    if features >= 1:
        y += np.sin(3.0 * X[:, 0])
    if features >= 2:
        y += 0.5 * np.cos(2.0 * X[:, 1])
    if features >= 4:
        y += 0.2 * X[:, 2] * X[:, 3]
    if features >= 5:
        y += -0.1 * (X[:, 4] ** 2)
    if features >= 6:
        y += 0.05 * np.sin(X[:, 5] * X[:, 0])
    y = (y + noise).reshape(-1, 1)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "task": "sine",
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
        },
    )


def _build_tabular_mixed(
    *,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    X = rng.uniform(-2.0, 2.0, size=(total, features)).astype(np.float32)
    noise = 0.05 * rng.standard_normal(size=(total,)).astype(np.float32)
    y = np.zeros((total,), dtype=np.float32)
    if features >= 1:
        y += 0.35 * np.sin(2.0 * X[:, 0])
    if features >= 2:
        y += 0.25 * np.cos(1.5 * X[:, 1])
    if features >= 4:
        y += 0.25 * X[:, 2] * X[:, 3]
    if features >= 5:
        y += -0.10 * (X[:, 4] ** 2)
    if features >= 6:
        y += 0.20 * np.tanh(X[:, 5])
    if features >= 7:
        y += 0.15 * np.maximum(0.0, X[:, 6])
    if features >= 8:
        y += 0.10 * np.where(X[:, 7] > 0.0, X[:, 7] ** 2, -0.5 * X[:, 7])
    if features >= 10:
        y += 0.05 * (X[:, 8] ** 3) - 0.03 * X[:, 9]
    y = (y + noise).reshape(-1, 1)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "task": "mixed",
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
        },
    )


def _build_tabular_teacher_mlp(
    *,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
    activation: str,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    total = n_train + n_test
    X = rng.normal(0.0, 1.0, size=(total, features)).astype(np.float32)

    hidden1 = max(32, min(256, int(features * 4)))
    hidden2 = max(16, min(256, int(features * 2)))

    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    act_key = activation.lower()
    if act_key == "relu":
        act_fn = _relu
    elif act_key == "tanh":
        act_fn = np.tanh
    else:
        raise ValueError("activation must be one of: 'relu', 'tanh'")

    def _scaled(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
        return rng.normal(0.0, 1.0 / np.sqrt(max(1, fan_in)), size=shape).astype(np.float32)

    W1 = _scaled((features, hidden1), fan_in=features)
    b1 = _scaled((hidden1,), fan_in=features)
    W2 = _scaled((hidden1, hidden2), fan_in=hidden1)
    b2 = _scaled((hidden2,), fan_in=hidden1)
    W3 = _scaled((hidden2, 1), fan_in=hidden2)
    b3 = _scaled((1,), fan_in=hidden2)

    h1 = act_fn(X @ W1 + b1)
    h2 = act_fn(h1 @ W2 + b2)
    y = (h2 @ W3 + b3).reshape(-1).astype(np.float32)

    y_std = float(np.std(y))
    if y_std > 1e-6:
        y = y / y_std
    y = y + 0.05 * rng.standard_normal(size=(total,)).astype(np.float32)
    y = y.reshape(-1, 1)

    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            "task": f"teacher_mlp_{act_key}",
            "teacher_hidden1": hidden1,
            "teacher_hidden2": hidden2,
            "features": features,
            "train_size": n_train,
            "test_size": n_test,
        },
    )


def _build_dataset(
    *,
    task: str,
    seed: int,
    n_train: int,
    n_test: int,
    features: int,
) -> DatasetBundle:
    key = str(task).lower().strip()
    if key in {"sine", "tabular_sine"}:
        return _build_tabular_sine(seed=seed, n_train=n_train, n_test=n_test, features=features)
    if key in {"mixed", "tabular_mixed"}:
        return _build_tabular_mixed(seed=seed, n_train=n_train, n_test=n_test, features=features)
    if key in {"teacher_relu", "teacher_mlp_relu"}:
        return _build_tabular_teacher_mlp(
            seed=seed, n_train=n_train, n_test=n_test, features=features, activation="relu"
        )
    if key in {"teacher_tanh", "teacher_mlp_tanh"}:
        return _build_tabular_teacher_mlp(
            seed=seed, n_train=n_train, n_test=n_test, features=features, activation="tanh"
        )
    raise ValueError("task must be one of: sine, mixed, teacher_relu, teacher_tanh")


def _train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0.0:
        return X, y, X[:0], y[:0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    n_val = max(1, int(round(X.shape[0] * val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class DenseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        activation_type: str = "relu",
        activation_config: Optional[Dict[str, Any]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(_build_geo_activation(activation_type, hidden_dim, activation_config))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.size(0), -1))


class DenseResidualBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        activation_type: str,
        activation_config: Optional[Dict[str, Any]] = None,
        norm: str = "rms",
        residual_alpha_init: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        norm_key = str(norm).lower()
        if norm_key == "none":
            self.norm = nn.Identity()
        elif norm_key == "layer":
            self.norm = nn.LayerNorm(int(dim))
        elif norm_key == "rms":
            self.norm = RMSNorm(int(dim))
        else:
            raise ValueError("norm must be one of: 'none', 'layer', 'rms'")

        self.fc1 = nn.Linear(int(dim), int(dim), bias=bool(bias))
        self.act = _build_geo_activation(str(activation_type), int(dim), activation_config)
        self.fc2 = nn.Linear(int(dim), int(dim), bias=bool(bias))
        self.alpha = nn.Parameter(torch.full((1,), float(residual_alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.fc2(h)
        return x + self.alpha * h


class DenseResidualNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        activation_type: str,
        activation_config: Optional[Dict[str, Any]] = None,
        norm: str = "rms",
        residual_alpha_init: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.in_proj = nn.Linear(int(input_dim), int(hidden_dim), bias=bool(bias))
        self.blocks = nn.ModuleList(
            [
                DenseResidualBlock(
                    int(hidden_dim),
                    activation_type=str(activation_type),
                    activation_config=activation_config,
                    norm=norm,
                    residual_alpha_init=residual_alpha_init,
                    bias=bias,
                )
                for _ in range(int(depth))
            ]
        )
        self.head = nn.Linear(int(hidden_dim), int(output_dim), bias=bool(bias))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x.reshape(x.size(0), -1))
        for block in self.blocks:
            z = block(z)
        return self.head(z)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=str, default="4x4", help="Layer shape HxW.")
    p.add_argument("--depth", type=int, default=4, help="Sparse depth (blocks).")
    p.add_argument("--k", type=int, default=8, help="Sparse fan-in per output.")
    p.add_argument("--pattern", type=str, default="local", choices=["local", "random", "hash"])
    p.add_argument("--radius", type=int, default=1, help="Neighborhood radius for connectivity.")
    p.add_argument("--wrap-mode", type=str, default="clamp", choices=["clamp", "wrap"])
    p.add_argument(
        "--sparse-activation",
        type=str,
        default="psann",
        help=(
            "GeoSparse activation type (e.g. psann, relu, tanh, mixed, "
            "relu_sigmoid_psann)."
        ),
    )
    p.add_argument(
        "--activation-config",
        type=str,
        default=None,
        help="JSON string for activation config (optional).",
    )
    p.add_argument(
        "--task",
        type=str,
        default="mixed",
        choices=["sine", "mixed", "teacher_relu", "teacher_tanh"],
        help="Synthetic regression task used to generate training data.",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--amp", action="store_true", help="Enable autocast mixed precision.")
    p.add_argument(
        "--amp-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Autocast dtype.",
    )
    p.add_argument("--tf32", action="store_true", help="Enable TF32 matmul on CUDA.")
    p.add_argument("--compile", action="store_true", help="Use torch.compile.")
    p.add_argument("--compile-backend", type=str, default="inductor")
    p.add_argument("--compile-mode", type=str, default="default")
    p.add_argument("--timing-warmup-steps", type=int, default=2)
    p.add_argument("--timing-epochs", type=int, default=1)
    p.add_argument("--train-size", type=int, default=4096)
    p.add_argument("--test-size", type=int, default=1024)
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data held out for validation (0 disables).",
    )
    p.add_argument(
        "--scale-x",
        action="store_true",
        help="Standardize X using train mean/std (fit on train split, applied to train+test).",
    )
    p.add_argument(
        "--scale-y",
        action="store_true",
        help=(
            "Standardize y using train mean/std. During evaluation, predictions are inverse-"
            "transformed back to original scale before computing MSE."
        ),
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--dense-depth", type=int, default=None, help="Dense hidden depth (defaults to sparse depth).")
    p.add_argument("--dense-max-width", type=int, default=4096)
    p.add_argument(
        "--match-tolerance",
        type=float,
        default=0.01,
        help="Max relative parameter mismatch allowed (fraction).",
    )
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def _activation_param_multiplier(activation_type: str) -> int:
    key = str(activation_type).lower()
    if key in {"psann", "sine", "respsann"}:
        return 3
    if key == "phase_psann":
        return 4
    if key in {"relu_sigmoid_psann", "rspsann", "rsp", "clipped_psann"}:
        return 4
    return 0


def _dense_mlp_params_with_activation(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation_type: str,
    bias: bool = True,
) -> int:
    base = dense_mlp_params(
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        bias=bool(bias),
    )
    extra = _activation_param_multiplier(activation_type) * int(hidden_dim) * int(depth)
    return int(base + extra)


def _match_dense_width_with_activation(
    *,
    target_params: int,
    input_dim: int,
    output_dim: int,
    depth: int,
    activation_type: str,
    bias: bool = True,
    max_width: int = 8192,
) -> Tuple[int, int]:
    if target_params <= 0:
        raise ValueError("target_params must be positive.")
    lo, hi = 1, int(max_width)
    best_width = 1
    best_mismatch = abs(
        _dense_mlp_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=best_width,
            depth=depth,
            activation_type=activation_type,
            bias=bias,
        )
        - target_params
    )
    while lo <= hi:
        mid = (lo + hi) // 2
        params = _dense_mlp_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mid,
            depth=depth,
            activation_type=activation_type,
            bias=bias,
        )
        mismatch = abs(params - target_params)
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_width = mid
        if params < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return int(best_width), int(best_mismatch)


def _norm_param_count(norm: str, dim: int) -> int:
    key = str(norm).lower()
    if key == "none":
        return 0
    if key == "layer":
        return 2 * int(dim)  # weight + bias
    if key == "rms":
        return int(dim)  # RMSNorm scale
    raise ValueError("norm must be one of: 'none', 'layer', 'rms'")


def _dense_linear_param_count(in_features: int, out_features: int, *, bias: bool) -> int:
    return int(in_features) * int(out_features) + (int(out_features) if bias else 0)


def _dense_residual_params_with_activation(
    *,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    depth: int,
    activation_type: str,
    norm: str = "rms",
    bias: bool = True,
) -> int:
    if depth <= 0:
        raise ValueError("depth must be positive.")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    total = _dense_linear_param_count(input_dim, hidden_dim, bias=bias)
    total += _dense_linear_param_count(hidden_dim, output_dim, bias=bias)
    block = 0
    block += _norm_param_count(norm, hidden_dim)
    block += _dense_linear_param_count(hidden_dim, hidden_dim, bias=bias)
    block += _activation_param_multiplier(activation_type) * int(hidden_dim)
    block += _dense_linear_param_count(hidden_dim, hidden_dim, bias=bias)
    block += 1  # residual alpha
    total += int(depth) * int(block)
    return int(total)


def _match_dense_residual_width_with_activation(
    *,
    target_params: int,
    input_dim: int,
    output_dim: int,
    depth: int,
    activation_type: str,
    norm: str = "rms",
    bias: bool = True,
    max_width: int = 8192,
) -> Tuple[int, int]:
    if target_params <= 0:
        raise ValueError("target_params must be positive.")
    lo, hi = 1, int(max_width)
    best_width = 1
    best_mismatch = abs(
        _dense_residual_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=best_width,
            depth=depth,
            activation_type=activation_type,
            norm=norm,
            bias=bias,
        )
        - target_params
    )
    while lo <= hi:
        mid = (lo + hi) // 2
        params = _dense_residual_params_with_activation(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=mid,
            depth=depth,
            activation_type=activation_type,
            norm=norm,
            bias=bias,
        )
        mismatch = abs(params - target_params)
        if mismatch < best_mismatch:
            best_mismatch = mismatch
            best_width = mid
        if params < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    return int(best_width), int(best_mismatch)


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    shape = _parse_shape(args.shape)
    features = shape[0] * shape[1]
    if args.dense_depth is None:
        args.dense_depth = int(args.depth)

    out_dir = Path(args.out) if args.out else Path("reports") / "geo_sparse" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    activation_cfg = json.loads(args.activation_config) if args.activation_config else None
    if str(args.sparse_activation).lower() == "mixed":
        types = None if activation_cfg is None else activation_cfg.get("activation_types", activation_cfg.get("types"))
        if not isinstance(types, list) or not types:
            raise SystemExit(
                "--sparse-activation mixed requires --activation-config JSON with "
                "'activation_types' (a non-empty list)"
            )

    dataset = _build_dataset(
        task=str(args.task),
        seed=int(args.seed),
        n_train=int(args.train_size),
        n_test=int(args.test_size),
        features=features,
    )
    X_train_raw = dataset.X_train
    y_train_raw = dataset.y_train
    X_test_raw = dataset.X_test
    y_test_raw = dataset.y_test
    X_train_raw, y_train_raw, X_val_raw, y_val_raw = _train_val_split(
        X_train_raw,
        y_train_raw,
        val_fraction=float(args.val_fraction),
        seed=int(args.seed),
    )
    y_train_unscaled = y_train_raw
    y_val_unscaled = y_val_raw
    y_test_unscaled = y_test_raw
    x_scaler = None
    y_scaler = None

    X_train = X_train_raw
    X_val = X_val_raw
    X_test = X_test_raw
    y_train = y_train_raw
    y_val = y_val_raw
    y_test = y_test_raw
    if args.scale_x:
        x_scaler = _fit_standard_scaler(X_train_raw)
        X_train = x_scaler.transform(X_train_raw)
        X_val = x_scaler.transform(X_val_raw)
        X_test = x_scaler.transform(X_test_raw)
    if args.scale_y:
        y_scaler = _fit_standard_scaler(y_train_unscaled)
        y_train = y_scaler.transform(y_train_unscaled)
        y_val = y_scaler.transform(y_val_unscaled)
        y_test = y_scaler.transform(y_test_unscaled)

    dataset = DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        meta={
            **dataset.meta,
            "val_size": int(X_val.shape[0]),
            "scale_x": bool(args.scale_x),
            "scale_y": bool(args.scale_y),
            "x_scaler": "standard" if args.scale_x else None,
            "y_scaler": "standard" if args.scale_y else None,
            "y_scaler_mean": float(y_scaler.mean.reshape(-1)[0]) if y_scaler is not None else None,
            "y_scaler_std": float(y_scaler.scale.reshape(-1)[0]) if y_scaler is not None else None,
        },
    )
    train_loader = _build_loader(
        dataset.X_train,
        dataset.y_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    device = choose_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        raise SystemExit(
            "CUDA device requested but torch.cuda.is_available() is False.\n"
            f"- torch={torch.__version__}\n"
            f"- CUDA_VISIBLE_DEVICES={cuda_visible!r}\n\n"
            "If running in Docker, start the container with GPU access (e.g. `--gpus all`)."
        )
    _print_header(args, out_dir, device)
    _set_precision(args.tf32)
    dtype = _resolve_dtype(args.dtype)
    amp_dtype = _resolve_dtype(args.amp_dtype)
    if device.type == "cpu" and amp_dtype == torch.float16:
        amp_dtype = torch.bfloat16

    sparse_model = GeoSparseNet(
        input_dim=features,
        output_dim=1,
        shape=shape,
        depth=int(args.depth),
        k=int(args.k),
        pattern=str(args.pattern),
        radius=int(args.radius),
        wrap_mode=str(args.wrap_mode),
        activation_type=str(args.sparse_activation),
        activation_config=activation_cfg,
        seed=int(args.seed),
    ).to(dtype=dtype)
    sparse_params_emp = count_params(sparse_model)
    sparse_params_analytic = geo_sparse_net_params(
        shape=shape,
        depth=int(args.depth),
        k=int(args.k),
        output_dim=1,
        bias=True,
    )

    sparse_params_trainable = count_params(sparse_model, trainable_only=True)

    dense_resrelu_width, dense_resrelu_mismatch = _match_dense_residual_width_with_activation(
        target_params=int(sparse_params_emp),
        input_dim=features,
        output_dim=1,
        depth=int(args.dense_depth),
        activation_type="relu",
        norm="rms",
        max_width=int(args.dense_max_width),
    )
    dense_resrelu_model = DenseResidualNet(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_resrelu_width),
        depth=int(args.dense_depth),
        activation_type="relu",
        norm="rms",
        bias=True,
    ).to(dtype=dtype)
    dense_resrelu_params_emp = count_params(dense_resrelu_model)
    dense_resrelu_params_trainable = count_params(dense_resrelu_model, trainable_only=True)
    dense_resrelu_params_analytic = _dense_residual_params_with_activation(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_resrelu_width),
        depth=int(args.dense_depth),
        activation_type="relu",
        norm="rms",
        bias=True,
    )
    dense_resrelu_mismatch_ratio = (
        abs(dense_resrelu_params_emp - sparse_params_emp) / float(sparse_params_emp)
        if sparse_params_emp > 0
        else None
    )
    if dense_resrelu_mismatch_ratio is not None and dense_resrelu_mismatch_ratio > float(
        args.match_tolerance
    ):
        print(
            "Warning: dense_resrelu parameter mismatch exceeds tolerance "
            f"({dense_resrelu_mismatch_ratio:.4f} > {float(args.match_tolerance):.4f})."
        )

    dense_respsann_width, dense_respsann_mismatch = _match_dense_residual_width_with_activation(
        target_params=int(sparse_params_emp),
        input_dim=features,
        output_dim=1,
        depth=int(args.dense_depth),
        activation_type="psann",
        norm="rms",
        max_width=int(args.dense_max_width),
    )
    dense_respsann_model = DenseResidualNet(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_respsann_width),
        depth=int(args.dense_depth),
        activation_type="psann",
        activation_config=activation_cfg,
        norm="rms",
        bias=True,
    ).to(dtype=dtype)
    dense_respsann_params_emp = count_params(dense_respsann_model)
    dense_respsann_params_trainable = count_params(dense_respsann_model, trainable_only=True)
    dense_respsann_params_analytic = _dense_residual_params_with_activation(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_respsann_width),
        depth=int(args.dense_depth),
        activation_type="psann",
        norm="rms",
        bias=True,
    )
    dense_respsann_mismatch_ratio = (
        abs(dense_respsann_params_emp - sparse_params_emp) / float(sparse_params_emp)
        if sparse_params_emp > 0
        else None
    )
    if dense_respsann_mismatch_ratio is not None and dense_respsann_mismatch_ratio > float(
        args.match_tolerance
    ):
        print(
            "Warning: dense_respsann parameter mismatch exceeds tolerance "
            f"({dense_respsann_mismatch_ratio:.4f} > {float(args.match_tolerance):.4f})."
        )

    sparse_train = _train_model(
        sparse_model,
        train_loader=train_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        amp=bool(args.amp),
        amp_dtype=amp_dtype,
        compile_model=bool(args.compile),
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        timing_warmup_steps=int(args.timing_warmup_steps),
        timing_epochs=int(args.timing_epochs),
        curve_every=1,
        curve_test=(dataset.X_test, dataset.y_test),
        curve_test_unscaled=y_test_unscaled,
        curve_test_y_scaler=y_scaler,
    )
    dense_resrelu_train = _train_model(
        dense_resrelu_model,
        train_loader=train_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        amp=bool(args.amp),
        amp_dtype=amp_dtype,
        compile_model=bool(args.compile),
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        timing_warmup_steps=int(args.timing_warmup_steps),
        timing_epochs=int(args.timing_epochs),
        curve_every=1,
        curve_test=(dataset.X_test, dataset.y_test),
        curve_test_unscaled=y_test_unscaled,
        curve_test_y_scaler=y_scaler,
    )
    dense_respsann_train = _train_model(
        dense_respsann_model,
        train_loader=train_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        amp=bool(args.amp),
        amp_dtype=amp_dtype,
        compile_model=bool(args.compile),
        compile_backend=args.compile_backend,
        compile_mode=args.compile_mode,
        timing_warmup_steps=int(args.timing_warmup_steps),
        timing_epochs=int(args.timing_epochs),
        curve_every=1,
        curve_test=(dataset.X_test, dataset.y_test),
        curve_test_unscaled=y_test_unscaled,
        curve_test_y_scaler=y_scaler,
    )

    sparse_curve = sparse_train.pop("curve", [])
    dense_resrelu_curve = dense_resrelu_train.pop("curve", [])
    dense_respsann_curve = dense_respsann_train.pop("curve", [])

    sparse_metrics_train = _evaluate_metrics(
        sparse_model,
        X=X_train,
        y=y_train,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_train_unscaled,
    )
    sparse_metrics_val = (
        _evaluate_metrics(
            sparse_model,
            X=X_val,
            y=y_val,
            device=device,
            y_scaler=y_scaler,
            y_unscaled=y_val_unscaled,
        )
        if X_val.size
        else None
    )
    sparse_metrics_test = _evaluate_metrics(
        sparse_model,
        X=X_test,
        y=y_test,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_test_unscaled,
    )

    dense_resrelu_metrics_train = _evaluate_metrics(
        dense_resrelu_model,
        X=X_train,
        y=y_train,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_train_unscaled,
    )
    dense_resrelu_metrics_val = (
        _evaluate_metrics(
            dense_resrelu_model,
            X=X_val,
            y=y_val,
            device=device,
            y_scaler=y_scaler,
            y_unscaled=y_val_unscaled,
        )
        if X_val.size
        else None
    )
    dense_resrelu_metrics_test = _evaluate_metrics(
        dense_resrelu_model,
        X=X_test,
        y=y_test,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_test_unscaled,
    )

    dense_respsann_metrics_train = _evaluate_metrics(
        dense_respsann_model,
        X=X_train,
        y=y_train,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_train_unscaled,
    )
    dense_respsann_metrics_val = (
        _evaluate_metrics(
            dense_respsann_model,
            X=X_val,
            y=y_val,
            device=device,
            y_scaler=y_scaler,
            y_unscaled=y_val_unscaled,
        )
        if X_val.size
        else None
    )
    dense_respsann_metrics_test = _evaluate_metrics(
        dense_respsann_model,
        X=X_test,
        y=y_test,
        device=device,
        y_scaler=y_scaler,
        y_unscaled=y_test_unscaled,
    )

    env = _get_env_info(device)

    manifest = {
        "shape": list(shape),
        "features": features,
        "depth_sparse": int(args.depth),
        "k": int(args.k),
        "dense_depth": int(args.dense_depth),
        "dense_resrelu_width": int(dense_resrelu_width),
        "dense_respsann_width": int(dense_respsann_width),
        "pattern": str(args.pattern),
        "radius": int(args.radius),
        "wrap_mode": str(args.wrap_mode),
        "sparse_activation": str(args.sparse_activation),
        "activation_config": activation_cfg,
        "task": str(args.task),
        "dataset_meta": dataset.meta,
        "train_size": int(args.train_size),
        "val_fraction": float(args.val_fraction),
        "val_size": int(X_val.shape[0]),
        "test_size": int(args.test_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "amp": bool(args.amp),
        "amp_dtype": str(args.amp_dtype),
        "amp_dtype_effective": str(amp_dtype),
        "tf32": bool(args.tf32),
        "compile": bool(args.compile),
        "compile_backend": str(args.compile_backend),
        "compile_mode": str(args.compile_mode),
        "timing_warmup_steps": int(args.timing_warmup_steps),
        "timing_epochs": int(args.timing_epochs),
        "match_tolerance": float(args.match_tolerance),
    }

    results = {
        "environment": env,
        "manifest": manifest,
        "models": [
            {
                "name": "geo_sparse",
                "params_empirical": sparse_params_emp,
                "params_trainable": sparse_params_trainable,
                "params_analytic": sparse_params_analytic,
                "train": sparse_train,
                "curve": sparse_curve,
                "metrics_train": sparse_metrics_train,
                "metrics_val": sparse_metrics_val,
                "metrics_test": sparse_metrics_test,
                "mse_train": sparse_metrics_train.get("mse"),
                "mse_val": sparse_metrics_val.get("mse") if sparse_metrics_val else None,
                "mse_test": sparse_metrics_test.get("mse"),
            },
            {
                "name": "dense_resrelu",
                "params_empirical": dense_resrelu_params_emp,
                "params_trainable": dense_resrelu_params_trainable,
                "params_analytic": dense_resrelu_params_analytic,
                "param_mismatch": int(abs(dense_resrelu_params_emp - sparse_params_emp)),
                "param_mismatch_ratio": dense_resrelu_mismatch_ratio,
                "train": dense_resrelu_train,
                "curve": dense_resrelu_curve,
                "metrics_train": dense_resrelu_metrics_train,
                "metrics_val": dense_resrelu_metrics_val,
                "metrics_test": dense_resrelu_metrics_test,
                "mse_train": dense_resrelu_metrics_train.get("mse"),
                "mse_val": dense_resrelu_metrics_val.get("mse") if dense_resrelu_metrics_val else None,
                "mse_test": dense_resrelu_metrics_test.get("mse"),
            },
            {
                "name": "dense_respsann",
                "params_empirical": dense_respsann_params_emp,
                "params_trainable": dense_respsann_params_trainable,
                "params_analytic": dense_respsann_params_analytic,
                "param_mismatch": int(abs(dense_respsann_params_emp - sparse_params_emp)),
                "param_mismatch_ratio": dense_respsann_mismatch_ratio,
                "train": dense_respsann_train,
                "curve": dense_respsann_curve,
                "metrics_train": dense_respsann_metrics_train,
                "metrics_val": dense_respsann_metrics_val,
                "metrics_test": dense_respsann_metrics_test,
                "mse_train": dense_respsann_metrics_train.get("mse"),
                "mse_val": dense_respsann_metrics_val.get("mse") if dense_respsann_metrics_val else None,
                "mse_test": dense_respsann_metrics_test.get("mse"),
            },
        ],
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary_rows = []
    for entry in results["models"]:
        row = {
            "model": entry["name"],
            "params_empirical": entry["params_empirical"],
            "params_analytic": entry["params_analytic"],
            "mse_train": entry.get("mse_train"),
            "mse_val": entry.get("mse_val"),
            "mse_test": entry["mse_test"],
        }
        if "param_mismatch" in entry:
            row["param_mismatch"] = entry["param_mismatch"]
        if "param_mismatch_ratio" in entry:
            row["param_mismatch_ratio"] = entry["param_mismatch_ratio"]
        if isinstance(entry.get("metrics_train"), dict):
            row.update(_flatten_dict("train_", entry["metrics_train"]))
        if isinstance(entry.get("metrics_val"), dict):
            row.update(_flatten_dict("val_", entry["metrics_val"]))
        if isinstance(entry.get("metrics_test"), dict):
            row.update(_flatten_dict("test_", entry["metrics_test"]))
        row.update(_flatten_dict("", entry["train"]))
        summary_rows.append(row)
    _write_summary(summary_rows, out_dir / "summary.csv")

    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
