#!/usr/bin/env python
"""Benchmark GeoSparseNet vs dense ReLU MLP with matched parameters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from psann.nn_geo_sparse import GeoSparseNet
from psann.params import count_params, dense_mlp_params, geo_sparse_net_params, match_dense_width
from psann.utils import choose_device, seed_all


@dataclass
class DatasetBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: Dict[str, Any]


def _parse_shape(text: str) -> Tuple[int, int]:
    for sep in ("x", "X", ","):
        if sep in text:
            parts = text.split(sep)
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    raise ValueError("shape must be formatted as HxW or H,W")


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
        meta={"features": features, "train_size": n_train, "test_size": n_test},
    )


class DenseMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int,
        depth: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        layers = []
        in_dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.size(0), -1))


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

    step_events = []
    total_samples = 0
    start = time.perf_counter()
    for epoch in range(int(epochs)):
        model.train()
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
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if record_step and device.type == "cuda":
                end_event.record()
                step_events.append((start_event, end_event))
            total_samples += int(xb.shape[0])
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
    }


def _evaluate(model: nn.Module, *, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    X_t = torch.from_numpy(X).to(device=device)
    y_t = torch.from_numpy(y).to(device=device)
    with torch.no_grad():
        pred = model(X_t)
        return float(torch.mean((pred - y_t) ** 2).item())


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
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32
        if torch.cuda.is_available()
        else None,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
        "matmul_precision": torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else None,
        "torch_compile_available": hasattr(torch, "compile"),
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = torch.cuda.get_device_capability(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info["gpu_mem_total_bytes"] = int(total_mem)
            info["gpu_mem_free_bytes"] = int(free_mem)
        except Exception:
            pass
    return info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", type=str, default="4x4", help="Layer shape HxW.")
    p.add_argument("--depth", type=int, default=4, help="Sparse depth (blocks).")
    p.add_argument("--k", type=int, default=8, help="Sparse fan-in per output.")
    p.add_argument("--pattern", type=str, default="local", choices=["local", "random", "hash"])
    p.add_argument("--radius", type=int, default=1, help="Neighborhood radius for connectivity.")
    p.add_argument("--wrap-mode", type=str, default="clamp", choices=["clamp", "wrap"])
    p.add_argument("--sparse-activation", type=str, default="psann")
    p.add_argument(
        "--activation-config",
        type=str,
        default=None,
        help="JSON string for activation config (optional).",
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


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    shape = _parse_shape(args.shape)
    features = shape[0] * shape[1]
    if args.dense_depth is None:
        args.dense_depth = int(args.depth)

    out_dir = Path(args.out) if args.out else Path("reports") / "geo_sparse" / time.strftime(
        "%Y%m%d_%H%M%S"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    activation_cfg = json.loads(args.activation_config) if args.activation_config else None

    dataset = _build_tabular_sine(
        seed=int(args.seed),
        n_train=int(args.train_size),
        n_test=int(args.test_size),
        features=features,
    )
    train_loader = _build_loader(
        dataset.X_train,
        dataset.y_train,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    device = choose_device(args.device)
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

    dense_width, dense_mismatch = match_dense_width(
        target_params=int(sparse_params_emp),
        input_dim=features,
        output_dim=1,
        depth=int(args.dense_depth),
        max_width=int(args.dense_max_width),
    )
    dense_model = DenseMLP(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_width),
        depth=int(args.dense_depth),
        bias=True,
    ).to(dtype=dtype)
    dense_params_emp = count_params(dense_model)
    dense_params_trainable = count_params(dense_model, trainable_only=True)
    sparse_params_trainable = count_params(sparse_model, trainable_only=True)
    dense_params_analytic = dense_mlp_params(
        input_dim=features,
        output_dim=1,
        hidden_dim=int(dense_width),
        depth=int(args.dense_depth),
        bias=True,
    )
    mismatch_ratio = (
        abs(dense_params_emp - sparse_params_emp) / float(sparse_params_emp)
        if sparse_params_emp > 0
        else None
    )
    if mismatch_ratio is not None and mismatch_ratio > float(args.match_tolerance):
        print(
            "Warning: parameter mismatch exceeds tolerance "
            f"({mismatch_ratio:.4f} > {float(args.match_tolerance):.4f})."
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
    )
    dense_train = _train_model(
        dense_model,
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
    )

    sparse_mse = _evaluate(
        sparse_model, X=dataset.X_test, y=dataset.y_test, device=device
    )
    dense_mse = _evaluate(
        dense_model, X=dataset.X_test, y=dataset.y_test, device=device
    )

    env = _get_env_info(device)

    manifest = {
        "shape": list(shape),
        "features": features,
        "depth_sparse": int(args.depth),
        "k": int(args.k),
        "dense_depth": int(args.dense_depth),
        "dense_width": int(dense_width),
        "pattern": str(args.pattern),
        "radius": int(args.radius),
        "wrap_mode": str(args.wrap_mode),
        "sparse_activation": str(args.sparse_activation),
        "activation_config": activation_cfg,
        "train_size": int(args.train_size),
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
                "mse_test": sparse_mse,
            },
            {
                "name": "dense_relu",
                "params_empirical": dense_params_emp,
                "params_trainable": dense_params_trainable,
                "params_analytic": dense_params_analytic,
                "param_mismatch": int(abs(dense_params_emp - sparse_params_emp)),
                "param_mismatch_ratio": mismatch_ratio,
                "train": dense_train,
                "mse_test": dense_mse,
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
            "mse_test": entry["mse_test"],
        }
        if "param_mismatch" in entry:
            row["param_mismatch"] = entry["param_mismatch"]
        if "param_mismatch_ratio" in entry:
            row["param_mismatch_ratio"] = entry["param_mismatch_ratio"]
        row.update(_flatten_dict("", entry["train"]))
        summary_rows.append(row)
    _write_summary(summary_rows, out_dir / "summary.csv")

    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
