"""Lightweight PSANN probe runner ported from the Colab notebook.

This script mirrors the behaviour of `PSANN_Light_Probes_Colab.ipynb` while
providing a regular Python entry point that can be executed locally.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if _SRC_ROOT.exists():
    sys.path.insert(0, str(_SRC_ROOT))

from psann.conv import PSANNConv1dNet
from psann.params import count_params, match_dense_width
from psann.utils import choose_device
from psann.utils import seed_all as psann_seed_all
from gpu_env_report import gather_env_info


def _maybe_install(module: str, package: str | None = None) -> None:
    """Install a dependency if it cannot be imported."""
    try:
        importlib.import_module(module)
    except ImportError:
        target = package or module
        print(f"[deps] Installing {target}...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", target])


_DEPENDENCIES = [
    ("psann", "psann"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("torchaudio", "torchaudio"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
]


def ensure_dependencies() -> None:
    for module_name, package_name in _DEPENDENCIES:
        _maybe_install(module_name, package_name)


def _ensure_torch_dynamo_stub() -> None:
    """Provide minimal torch._dynamo API pieces when missing."""

    def _make_disable():
        def _disable(fn=None, recursive=True):
            if fn is None:

                def decorator(f):
                    return f

                return decorator
            return fn

        return _disable

    def _graph_break(*_args, **_kwargs):
        return None

    try:
        dynamo = importlib.import_module("torch._dynamo")
    except Exception:  # pragma: no cover - defensive stub
        dynamo = None

    if dynamo is None:
        stub = types.ModuleType("torch._dynamo")
        stub.disable = _make_disable()
        stub.graph_break = _graph_break
        sys.modules["torch._dynamo"] = stub
        torch._dynamo = stub  # type: ignore[attr-defined]
        return

    if not getattr(dynamo, "disable", None):
        dynamo.disable = _make_disable()  # type: ignore[attr-defined]
    if not getattr(dynamo, "graph_break", None):
        dynamo.graph_break = _graph_break  # type: ignore[attr-defined]
    torch._dynamo = dynamo  # type: ignore[attr-defined]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("PSANN_DATA_ROOT", PROJECT_ROOT / "datasets")).resolve()
DEFAULT_RESULTS_ROOT = (PROJECT_ROOT / "colab_results_light").resolve()
RESULTS_ROOT = DEFAULT_RESULTS_ROOT


def configure_results_root(path: Optional[str]) -> Path:
    """Resolve and create the results directory, updating the module global."""
    global RESULTS_ROOT
    target = Path(path).expanduser().resolve() if path else DEFAULT_RESULTS_ROOT
    target.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT = target
    return target


REQUIRED_DATASETS = [
    DATA_ROOT / "Beijing Air Quality",
    DATA_ROOT / "Industrial Data from the Electric Arc Furnace",
]

JENA_ZIP_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
)


def _datasets_ready() -> bool:
    for path in REQUIRED_DATASETS:
        if not path.exists():
            return False
        if path.is_dir() and not any(path.rglob("*")):
            return False
    return True


def ensure_jena_dataset() -> Path:
    base = DATA_ROOT / "Jena Climate 2009-2016"
    csv = base / "jena_climate_2009_2016.csv"
    if csv.exists():
        return csv
    base.mkdir(parents=True, exist_ok=True)
    tmp_zip = base / "jena_climate_2009_2016.csv.zip"
    import urllib.request

    try:
        print(f"[data] Downloading Jena Climate dataset to {base} ...")
        urllib.request.urlretrieve(JENA_ZIP_URL, tmp_zip)
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            zf.extractall(base)
    except Exception as exc:  # pragma: no cover - network failure surfaces to user
        print(f"[warn] Failed to download Jena dataset: {exc}")
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink()
    if csv.exists():
        return csv
    matches = list(base.glob("**/jena_climate_2009_2016.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate Jena climate CSV under {base}")


def seed_all(seed: int) -> None:
    psann_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(arg: str) -> torch.device:
    return choose_device(arg)


def _fix_backslash_artifacts(root: Path) -> None:
    for leftover in root.iterdir():
        name = leftover.name
        if "\\\\" in name and name.lower().startswith("datasets"):
            rel = Path(*Path(name.replace("\\", "/")).parts)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                leftover.replace(dest)
                print(f"[fix] Moved stray {leftover} -> {dest}")
            except Exception as exc:
                print(f"[warn] Could not move {leftover}: {exc}")


def maybe_extract_datasets_zip() -> None:
    _fix_backslash_artifacts(PROJECT_ROOT)
    if _datasets_ready():
        return
    zip_path = PROJECT_ROOT / "datasets.zip"
    if not zip_path.exists():
        print(f"[warn] Required datasets missing under {DATA_ROOT} and datasets.zip not found.")
        return
    print(f"[info] Extracting {zip_path} to {PROJECT_ROOT}/datasets (robust normalisation)...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for zi in z.infolist():
            name = zi.filename
            norm = name.replace("\\", "/").lstrip("./")
            parts = [p for p in norm.split("/") if p]
            if not parts:
                continue
            if parts[0].lower() != "datasets":
                parts = ["datasets"] + parts
            dest = PROJECT_ROOT.joinpath(*parts)
            if zi.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(zi) as src, open(dest, "wb") as f:
                    f.write(src.read())
    _fix_backslash_artifacts(PROJECT_ROOT)


class PSANNConvSpine(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int,
        depth: int,
        kernel_size: int,
        horizon: int,
        aggregator: str = "last",
    ):
        super().__init__()
        self.aggregator = aggregator
        self.core = PSANNConv1dNet(
            in_channels=in_ch,
            out_dim=hidden,
            hidden_layers=depth,
            conv_channels=hidden,
            hidden_channels=hidden,
            kernel_size=kernel_size,
            segmentation_head=True,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.core(x.transpose(1, 2))
        pooled = features[:, :, -1] if self.aggregator == "last" else features.mean(dim=-1)
        return self.head(pooled)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        return self.net(flat)


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


def build_windows(
    df: pd.DataFrame, feature_cols: List[str], target_col: str, context: int, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    values = df[feature_cols].values.astype(np.float32)
    target = df[target_col].values.astype(np.float32)
    idxs = []
    for start in range(context, len(df) - horizon):
        end = start
        idxs.append((start - context, start, end, end + horizon))
    Xw = np.stack([values[s:e] for (s, e, _, __) in idxs], axis=0)
    Yw = np.stack([target[s2:e2] for (_, __, s2, e2) in idxs], axis=0)
    return Xw, Yw


def split_train_val_test(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15
):
    n = X.shape[0]
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return (
        X[:n_train],
        y[:n_train],
        X[n_train : n_train + n_val],
        y[n_train : n_train + n_val],
        X[n_train + n_val :],
        y[n_train + n_val :],
    )


def load_jena_light(context: int = 72, horizon: int = 36, subset_days: Optional[int] = 120):
    def _norm(s: str) -> str:
        trans = {"–": "-", "—": "-", "‑": "-", "−": "-"}
        return "".join(trans.get(ch, ch) for ch in s).lower()

    base = DATA_ROOT / "Jena Climate 2009-2016"
    csv = base / "jena_climate_2009_2016.csv"
    if not csv.exists():
        try:
            csv = ensure_jena_dataset()
        except FileNotFoundError:
            csv = None
    if csv is None or not Path(csv).exists():
        candidates = [d for d in DATA_ROOT.iterdir() if d.is_dir() and "jena" in _norm(d.name)]
        found = None
        for d in candidates:
            hits = list(d.rglob("jena_climate_2009_2016.csv"))
            if hits:
                found = hits[0]
                break
            hits = list(d.rglob("*jena*climate*2016*.csv"))
            if hits:
                found = hits[0]
                break
        if found is None:
            raise FileNotFoundError(f"Could not find Jena climate CSV under {DATA_ROOT}")
        csv = found
    df = pd.read_csv(csv)
    target_col = next(
        (c for c in df.columns if c.strip().lower().startswith("t ") or "degc" in c.lower()), None
    )
    if target_col is None:
        raise RuntimeError("Could not find temperature column (e.g., T (degC))")
    num_df = df.select_dtypes(include=[np.number]).copy()
    if subset_days is not None:
        num_df = num_df.tail(subset_days * 144)
    num_df = (num_df - num_df.mean()) / (num_df.std().replace(0, 1.0))
    feature_cols = list(num_df.columns)
    Xw, Yw = build_windows(num_df, feature_cols, target_col, context, horizon)
    return split_train_val_test(Xw, Yw)


def load_beijing_light(
    station_name: str = "Guanyuan",
    context: int = 24,
    horizon: int = 6,
    subset_days: Optional[int] = 120,
):
    base = DATA_ROOT / "Beijing Air Quality"
    station_file = None
    for p in base.glob("PRSA_Data_*_20130301-20170228.csv"):
        if station_name.lower() in p.name.lower():
            station_file = p
            break
    if station_file is None:
        raise FileNotFoundError(f"Could not find station file containing {station_name}")
    df = pd.read_csv(station_file)
    target_col = (
        "PM2.5" if "PM2.5" in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    )
    num_df = df.select_dtypes(include=[np.number]).copy().ffill().bfill().fillna(0.0)
    if subset_days is not None:
        num_df = num_df.tail(subset_days * 24)
    num_df = (num_df - num_df.mean()) / (num_df.std().replace(0, 1.0))
    feature_cols = list(num_df.columns)
    Xw, Yw = build_windows(num_df, feature_cols, target_col, context, horizon)
    return split_train_val_test(Xw, Yw)


def load_eaf_temp_lite(
    context: int = 16, horizon: int = 1, heats_limit: int = 5, min_rows: int = 120
):
    path = DATA_ROOT / "Industrial Data from the Electric Arc Furnace" / "eaf_temp.csv"
    df = pd.read_csv(path)
    if not {"HEATID", "DATETIME", "TEMP"}.issubset(df.columns):
        raise RuntimeError("Missing expected columns in eaf_temp.csv")
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    df = df.dropna(subset=["DATETIME"]).sort_values(["HEATID", "DATETIME"])
    heats_df = df.groupby("HEATID").size().reset_index(name="n").sort_values("n", ascending=False)
    selected = heats_df.query("n >= @min_rows").head(heats_limit)
    if selected.empty:
        fallback = heats_df.head(heats_limit)
        if fallback.empty:
            raise RuntimeError("No heats found in eaf_temp.csv")
        print(
            f"[warn] No EAF heats with >= {min_rows} rows; using top {len(fallback)} heats with >= {int(fallback['n'].min())} rows instead."
        )
        selected = fallback
    parts = []
    for hid in selected["HEATID"]:
        seg = df[df["HEATID"] == hid].copy()
        num_cols = ["TEMP"] + (["VALO2_PPM"] if "VALO2_PPM" in seg.columns else [])
        seg_num = seg[num_cols]
        seg_num = (seg_num - seg_num.mean()) / (seg_num.std().replace(0, 1.0))
        Xw, Yw = build_windows(
            seg_num, feature_cols=num_cols, target_col="TEMP", context=context, horizon=horizon
        )
        if Xw.size == 0:
            continue
        parts.append((Xw, Yw))
    if not parts:
        raise RuntimeError("No EAF heats with sufficient rows found for lite run")
    X = np.concatenate([p[0] for p in parts], axis=0)
    Y = np.concatenate([p[1] for p in parts], axis=0)
    return split_train_val_test(X, Y)


@dataclass
class TrainSpec:
    model: str
    hidden: int
    depth: int
    epochs: int
    kernel_size: int = 5
    lr: float = 1e-3
    batch_size: int = 256


def train_regressor(
    model: nn.Module, train_X, train_y, val_X, val_y, spec: TrainSpec, device: torch.device
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    loss_fn = nn.MSELoss()
    tX = torch.from_numpy(train_X).float().to(device)
    ty = torch.from_numpy(train_y).float().to(device)
    vX = torch.from_numpy(val_X).float().to(device)
    vy = torch.from_numpy(val_y).float().to(device)
    total_steps = 0
    history: List[dict] = []
    total_start = time.perf_counter()
    for epoch in range(spec.epochs):
        model.train()
        epoch_start = time.perf_counter()
        perm = torch.randperm(tX.size(0), device=device)
        epoch_loss_sum = 0.0
        epoch_samples = 0
        step_times: List[float] = []
        grad_norms: List[float] = []
        loss_nonfinite = 0
        grad_nonfinite = 0
        for i in range(0, len(perm), spec.batch_size):
            idx = perm[i : i + spec.batch_size]
            xb, yb = tX.index_select(0, idx), ty.index_select(0, idx)
            opt.zero_grad()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_start = time.perf_counter()
            loss = loss_fn(model(xb), yb)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                loss_nonfinite += 1
            loss.backward()
            grad_norm = _grad_norm(model)
            if grad_norm is not None:
                grad_norms.append(grad_norm)
                if not math.isfinite(grad_norm):
                    grad_nonfinite += 1
            opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            step_times.append(time.perf_counter() - step_start)
            bs = xb.shape[0]
            epoch_loss_sum += loss_value * bs
            epoch_samples += bs
            total_steps += 1
        train_loss = epoch_loss_sum / max(epoch_samples, 1)
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vX), vy).item())
        epoch_time = time.perf_counter() - epoch_start
        finite_grad_norms = [g for g in grad_norms if math.isfinite(g)]
        grad_norm_mean = (
            float(sum(finite_grad_norms) / len(finite_grad_norms)) if finite_grad_norms else None
        )
        grad_norm_max = float(max(finite_grad_norms)) if finite_grad_norms else None
        step_time_mean = float(sum(step_times) / len(step_times)) if step_times else None
        history.append(
            {
                "epoch": int(epoch + 1),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "epoch_time_s": float(epoch_time),
                "step_time_s_mean": step_time_mean,
                "steps": int(len(step_times)),
                "samples": int(epoch_samples),
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_max": grad_norm_max,
                "loss_nonfinite_steps": int(loss_nonfinite),
                "grad_nonfinite_steps": int(grad_nonfinite),
            }
        )
    model.eval()
    with torch.no_grad():
        vloss = loss_fn(model(vX), vy).item()
    total_time = time.perf_counter() - total_start
    if history:
        history[-1]["train_time_s_total"] = float(total_time)
    return model, {
        "epochs": spec.epochs,
        "val_loss": float(vloss),
        "steps": total_steps,
        "train_size": int(len(train_X)),
        "history": history,
    }


def evaluate_regressor(
    model: nn.Module, test_X: np.ndarray, test_y: np.ndarray, device: torch.device
):
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(test_X).float().to(device)).cpu().numpy()
    preds = preds.reshape(test_y.shape)
    return _regression_metrics(test_y, preds)


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _history_stats(history: List[dict]) -> dict:
    if not history:
        return {}
    train_losses = [h.get("train_loss") for h in history if h.get("train_loss") is not None]
    val_losses = [h.get("val_loss") for h in history if h.get("val_loss") is not None]
    epoch_times = [h.get("epoch_time_s") for h in history if h.get("epoch_time_s") is not None]
    step_times = [h.get("step_time_s_mean") for h in history if h.get("step_time_s_mean") is not None]
    grad_norms = [h.get("grad_norm_max") for h in history if h.get("grad_norm_max") is not None]
    loss_nonfinite = sum(int(h.get("loss_nonfinite_steps") or 0) for h in history)
    grad_nonfinite = sum(int(h.get("grad_nonfinite_steps") or 0) for h in history)
    stats = {
        "history_length": int(len(history)),
        "loss_curve_volatility": _loss_curve_volatility([float(x) for x in train_losses])
        if train_losses
        else None,
        "loss_curve_mean_abs_diff": _loss_curve_mean_abs_diff([float(x) for x in train_losses])
        if train_losses
        else None,
        "val_curve_volatility": _loss_curve_volatility([float(x) for x in val_losses])
        if val_losses
        else None,
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


def run_light_task(
    task: str,
    seeds: List[int],
    device: torch.device,
    epochs: int,
    pr_snapshots: bool,
    match_params: bool,
    metrics_rows: List[dict],
    history_rows: List[dict],
) -> None:
    if task == "jena":
        train_X, train_y, val_X, val_y, test_X, test_y = load_jena_light(72, 36, 120)
        in_ch, horizon = train_X.shape[-1], train_y.shape[-1]
        specs = [
            (
                "psann_conv",
                TrainSpec("psann_conv", hidden=48, depth=2, kernel_size=5, epochs=epochs),
            ),
            ("mlp", TrainSpec("mlp", hidden=64, depth=2, epochs=epochs)),
        ]
        target_params = None
        mlp_mismatch = None
        if match_params:
            psann_tmp = PSANNConvSpine(in_ch, specs[0][1].hidden, specs[0][1].depth, specs[0][1].kernel_size, horizon)
            target_params = count_params(psann_tmp)
            mlp_hidden, mlp_mismatch = _match_mlp_hidden(
                target_params=target_params,
                input_dim=train_X.shape[1] * in_ch,
                output_dim=horizon,
                depth=specs[1][1].depth,
            )
            specs[1][1].hidden = mlp_hidden
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                model = (
                    PSANNConvSpine(in_ch, spec.hidden, spec.depth, spec.kernel_size, horizon)
                    if spec.model == "psann_conv"
                    else MLPRegressor(train_X.shape[1] * in_ch, spec.hidden, spec.depth, horizon)
                )
                model, info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                info = dict(info)
                history = info.pop("history", [])
                train_metrics = evaluate_regressor(model, train_X, train_y, device)
                val_metrics = evaluate_regressor(model, val_X, val_y, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                params_total = count_params(model)
                params_trainable = count_params(model, trainable_only=True)
                history_rows.extend(
                    {
                        "task": "jena_light",
                        "model": name,
                        "seed": seed,
                        **entry,
                    }
                    for entry in history
                )
                history_stats = _history_stats(history)
                metrics_rows.append(
                    {
                        "task": "jena_light",
                        "model": name,
                        "seed": seed,
                        "params_total": params_total,
                        "params_trainable": params_trainable,
                        "param_target": target_params,
                        "param_mismatch": mlp_mismatch if name == "mlp" else 0,
                        "param_mismatch_ratio": (
                            float(mlp_mismatch) / float(target_params)
                            if (name == "mlp" and target_params)
                            else 0.0
                        ),
                        "train_size": int(train_X.shape[0]),
                        "val_size": int(val_X.shape[0]),
                        "test_size": int(test_X.shape[0]),
                        "input_shape": list(train_X.shape[1:]),
                        **info,
                        **_prefix_metrics("train", train_metrics),
                        **_prefix_metrics("val", val_metrics),
                        **_prefix_metrics("test", test_metrics),
                        **history_stats,
                    }
                )
                if pr_snapshots and name == "psann_conv":
                    idx = np.random.choice(
                        test_X.shape[0], size=min(32, test_X.shape[0]), replace=False
                    )
                    top_sv, sum_sv, pr = jacobian_pr(model, test_X[idx], device)
                    pr_df = pd.DataFrame(
                        [
                            {
                                "task": "jena_light",
                                "model": name,
                                "seed": seed,
                                "phase": "end",
                                "top_sv": top_sv,
                                "sum_sv": sum_sv,
                                "pr": pr,
                            }
                        ]
                    )
                    pr_out = RESULTS_ROOT / "jacobian_pr.csv"
                    mode = "a" if pr_out.exists() else "w"
                    pr_df.to_csv(pr_out, index=False, mode=mode, header=(mode == "w"))
    elif task == "beijing":
        train_X, train_y, val_X, val_y, test_X, test_y = load_beijing_light("Guanyuan", 24, 6, 120)
        in_ch, horizon = train_X.shape[-1], train_y.shape[-1]
        specs = [
            (
                "psann_conv",
                TrainSpec("psann_conv", hidden=64, depth=2, kernel_size=5, epochs=epochs),
            ),
            ("mlp", TrainSpec("mlp", hidden=96, depth=2, epochs=epochs)),
        ]
        target_params = None
        mlp_mismatch = None
        if match_params:
            psann_tmp = PSANNConvSpine(in_ch, specs[0][1].hidden, specs[0][1].depth, specs[0][1].kernel_size, horizon)
            target_params = count_params(psann_tmp)
            mlp_hidden, mlp_mismatch = _match_mlp_hidden(
                target_params=target_params,
                input_dim=train_X.shape[1] * in_ch,
                output_dim=horizon,
                depth=specs[1][1].depth,
            )
            specs[1][1].hidden = mlp_hidden
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                model = (
                    PSANNConvSpine(in_ch, spec.hidden, spec.depth, spec.kernel_size, horizon)
                    if spec.model == "psann_conv"
                    else MLPRegressor(train_X.shape[1] * in_ch, spec.hidden, spec.depth, horizon)
                )
                model, info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                info = dict(info)
                history = info.pop("history", [])
                train_metrics = evaluate_regressor(model, train_X, train_y, device)
                val_metrics = evaluate_regressor(model, val_X, val_y, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                params_total = count_params(model)
                params_trainable = count_params(model, trainable_only=True)
                history_rows.extend(
                    {
                        "task": "beijing_light",
                        "model": name,
                        "seed": seed,
                        **entry,
                    }
                    for entry in history
                )
                history_stats = _history_stats(history)
                metrics_rows.append(
                    {
                        "task": "beijing_light",
                        "model": name,
                        "seed": seed,
                        "params_total": params_total,
                        "params_trainable": params_trainable,
                        "param_target": target_params,
                        "param_mismatch": mlp_mismatch if name == "mlp" else 0,
                        "param_mismatch_ratio": (
                            float(mlp_mismatch) / float(target_params)
                            if (name == "mlp" and target_params)
                            else 0.0
                        ),
                        "train_size": int(train_X.shape[0]),
                        "val_size": int(val_X.shape[0]),
                        "test_size": int(test_X.shape[0]),
                        "input_shape": list(train_X.shape[1:]),
                        **info,
                        **_prefix_metrics("train", train_metrics),
                        **_prefix_metrics("val", val_metrics),
                        **_prefix_metrics("test", test_metrics),
                        **history_stats,
                    }
                )
    elif task == "eaf":
        train_X, train_y, val_X, val_y, test_X, test_y = load_eaf_temp_lite(16, 1, 5, 120)
        in_ch, horizon = train_X.shape[-1], train_y.shape[-1]
        specs = [
            (
                "psann_conv",
                TrainSpec("psann_conv", hidden=32, depth=2, kernel_size=3, epochs=max(epochs, 8)),
            ),
            ("mlp", TrainSpec("mlp", hidden=48, depth=2, epochs=max(epochs, 8))),
        ]
        target_params = None
        mlp_mismatch = None
        if match_params:
            psann_tmp = PSANNConvSpine(in_ch, specs[0][1].hidden, specs[0][1].depth, specs[0][1].kernel_size, horizon)
            target_params = count_params(psann_tmp)
            mlp_hidden, mlp_mismatch = _match_mlp_hidden(
                target_params=target_params,
                input_dim=train_X.shape[1] * in_ch,
                output_dim=horizon,
                depth=specs[1][1].depth,
            )
            specs[1][1].hidden = mlp_hidden
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                model = (
                    PSANNConvSpine(in_ch, spec.hidden, spec.depth, spec.kernel_size, horizon)
                    if spec.model == "psann_conv"
                    else MLPRegressor(train_X.shape[1] * in_ch, spec.hidden, spec.depth, horizon)
                )
                model, info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                info = dict(info)
                history = info.pop("history", [])
                train_metrics = evaluate_regressor(model, train_X, train_y, device)
                val_metrics = evaluate_regressor(model, val_X, val_y, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                params_total = count_params(model)
                params_trainable = count_params(model, trainable_only=True)
                history_rows.extend(
                    {
                        "task": "eaf_temp_lite",
                        "model": name,
                        "seed": seed,
                        **entry,
                    }
                    for entry in history
                )
                history_stats = _history_stats(history)
                metrics_rows.append(
                    {
                        "task": "eaf_temp_lite",
                        "model": name,
                        "seed": seed,
                        "params_total": params_total,
                        "params_trainable": params_trainable,
                        "param_target": target_params,
                        "param_mismatch": mlp_mismatch if name == "mlp" else 0,
                        "param_mismatch_ratio": (
                            float(mlp_mismatch) / float(target_params)
                            if (name == "mlp" and target_params)
                            else 0.0
                        ),
                        "train_size": int(train_X.shape[0]),
                        "val_size": int(val_X.shape[0]),
                        "test_size": int(test_X.shape[0]),
                        "input_shape": list(train_X.shape[1:]),
                        **info,
                        **_prefix_metrics("train", train_metrics),
                        **_prefix_metrics("val", val_metrics),
                        **_prefix_metrics("test", test_metrics),
                        **history_stats,
                    }
                )
    else:
        raise ValueError(f"Unknown task: {task}")


def run_all(tasks, seeds, epochs, device_str, pr_snapshots, match_params: bool):
    maybe_extract_datasets_zip()
    device = pick_device(device_str)
    print(f"[env] DATA_ROOT={DATA_ROOT}")
    print(f"[env] RESULTS_ROOT={RESULTS_ROOT}")
    print(f"[env] device={device}")
    metrics_rows: List[dict] = []
    history_rows: List[dict] = []
    env_info = gather_env_info()
    env_info["selected_device"] = str(device)
    env_info["results_root"] = str(RESULTS_ROOT)
    (RESULTS_ROOT / "env.json").write_text(
        json.dumps(env_info, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "tasks": list(tasks),
        "seeds": list(seeds),
        "epochs": int(epochs),
        "device": str(device),
        "match_params": bool(match_params),
        "data_root": str(DATA_ROOT),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    for task in tasks:
        print(f"[run] task={task}")
        run_light_task(
            task,
            seeds,
            device,
            epochs,
            pr_snapshots,
            match_params,
            metrics_rows,
            history_rows,
        )
    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        out = RESULTS_ROOT / "metrics.csv"
        df.to_csv(out, index=False)
        print(df.head())
        print(f"[done] Wrote metrics to {out}")
        metric_cols = [
            c
            for c in df.columns
            if c.startswith(("train_", "val_", "test_"))
            and any(
                key in c
                for key in ("mse", "rmse", "mae", "smape", "r2")
            )
        ]
        stability_cols = [
            c
            for c in df.columns
            if c
            in (
                "loss_curve_volatility",
                "loss_curve_mean_abs_diff",
                "val_curve_volatility",
                "grad_norm_max",
                "loss_nonfinite_steps",
                "grad_nonfinite_steps",
                "epoch_time_s_mean",
                "step_time_s_mean",
            )
        ]
        if metric_cols or stability_cols:
            agg_cols = metric_cols + stability_cols
            summary = df.groupby(["task", "model"])[agg_cols].agg(["mean", "std"])
            summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
            summary["n_runs"] = df.groupby(["task", "model"]).size()
            summary = summary.reset_index()
            summary_path = RESULTS_ROOT / "summary.csv"
            summary.to_csv(summary_path, index=False)
            print(f"[done] Wrote summary to {summary_path}")
        if history_rows:
            hist_path = RESULTS_ROOT / "history.jsonl"
            with hist_path.open("w", encoding="utf-8") as fh:
                for row in history_rows:
                    fh.write(json.dumps(row) + "\n")
            print(f"[done] Wrote history to {hist_path}")
    else:
        print("[warn] No metrics collected")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight PSANN probe benchmarks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["jena", "beijing", "eaf"],
        help="Tasks to run (jena, beijing, eaf).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[7, 8], help="Random seeds to evaluate."
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epoch budget for each model.")
    parser.add_argument("--device", default="auto", help="Device preference: auto | cpu | cuda.")
    parser.add_argument(
        "--pr-snapshots", action="store_true", help="Record Jacobian participation ratio snapshots."
    )
    parser.add_argument(
        "--match-params",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match MLP hidden width to PSANN parameter count.",
    )
    parser.add_argument(
        "--skip-deps", action="store_true", help="Skip dependency installation checks."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory for metrics/log outputs (default: <repo>/colab_results_light).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not args.skip_deps:
        ensure_dependencies()
    _ensure_torch_dynamo_stub()
    configure_results_root(args.results_dir)
    run_all(
        args.tasks,
        args.seeds,
        args.epochs,
        args.device,
        args.pr_snapshots,
        args.match_params,
    )


if __name__ == "__main__":
    main()
