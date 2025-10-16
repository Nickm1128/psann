#!/usr/bin/env python
"""
Lightweight Colab test suite to replicate key results and run a small EAF real-data test.

Targets (reduced compute):
- Jena climate (T degC) forecasting with reduced subset, 72 ctx / 36h horizon, 2 seeds
- Beijing PM2.5 forecasting (single station), 24h ctx / 6h horizon, 2 seeds
- EAF next-step TEMP (few heats), autoregressive
- Optional Jacobian participation ratio snapshots on Jena (start/mid/end)

Outputs are written to colab_results_light/ as small CSVs.

Run only in Colab. Do not run locally.
"""

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error


# -----------------------------
# Utility & environment helpers
# -----------------------------

PROJECT_ROOT = Path(os.getcwd())
DATA_ROOT = Path(os.getenv("PSANN_DATA_ROOT", PROJECT_ROOT / "datasets")).resolve()
RESULTS_ROOT = (PROJECT_ROOT / "colab_results_light").resolve()
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(arg: str) -> torch.device:
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_extract_datasets_zip() -> None:
    # Colab convenience: if datasets/ not present, try extracting datasets.zip if available
    if DATA_ROOT.exists():
        return
    zip_path = PROJECT_ROOT / "datasets.zip"
    if zip_path.exists():
        import zipfile

        print(f"[info] Extracting {zip_path} to {PROJECT_ROOT}/datasets ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(PROJECT_ROOT)
    else:
        print(f"[warn] DATA_ROOT {DATA_ROOT} not found and datasets.zip missing. Upload datasets first.")


# -----------------------------
# Minimal models (MLP and PSANN conv spine)
# -----------------------------


class SineParam(nn.Module):
    def __init__(self, features: int, damping: str = "abs"):
        super().__init__()
        self.features = features
        self.a = nn.Parameter(torch.zeros(features))  # amplitude pre-softplus
        self.b = nn.Parameter(torch.zeros(features))  # frequency pre-softplus
        self.c = nn.Parameter(torch.zeros(features))  # damping pre-softplus
        self.eps_f = 1e-6
        self.damping = damping

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, T) or (B, F)
        A = torch.nn.functional.softplus(self.a)
        f = torch.nn.functional.softplus(self.b) + self.eps_f
        d = torch.nn.functional.softplus(self.c)

        if z.dim() == 3:
            # (B, C, T)
            # broadcast parameters across time
            A_b = A.view(1, -1, 1)
            f_b = f.view(1, -1, 1)
            d_b = d.view(1, -1, 1)
            if self.damping == "abs":
                g = torch.abs(z)
            elif self.damping == "relu":
                g = torch.relu(z)
            else:
                g = torch.zeros_like(z)
            return A_b * torch.exp(-d_b * g) * torch.sin(f_b * z)
        else:
            # (B, F)
            if self.damping == "abs":
                g = torch.abs(z)
            elif self.damping == "relu":
                g = torch.relu(z)
            else:
                g = torch.zeros_like(z)
            return A * torch.exp(-d * g) * torch.sin(f * z)


class PSANNConvSpine(nn.Module):
    def __init__(self, in_ch: int, hidden: int, depth: int, kernel_size: int, horizon: int, aggregator: str = "last"):
        super().__init__()
        ks = kernel_size
        layers = []
        ch = in_ch
        for _ in range(depth):
            layers.append(nn.Conv1d(ch, hidden, ks, padding=ks // 2))
            layers.append(SineParam(hidden))
            ch = hidden
        self.core = nn.Sequential(*layers)
        self.aggregator = aggregator
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        h = self.core(x)
        # (B, C, T)
        if self.aggregator == "last":
            h_last = h[:, :, -1]
        else:
            h_last = h.mean(dim=-1)
        y = self.head(h_last)
        return y


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int):
        super().__init__()
        layers = []
        dim = in_dim
        for _ in range(depth):
            layers += [nn.Linear(dim, hidden), nn.ReLU()]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> flatten last two dims
        b, t, f = x.shape
        return self.net(x.reshape(b, t * f))


@dataclass
class TrainSpec:
    model: str  # 'psann_conv' or 'mlp'
    hidden: int
    depth: int
    kernel_size: int = 5
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 256
    patience: int = 3


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def train_regressor(
    model: nn.Module,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    spec: TrainSpec,
    device: torch.device,
) -> Tuple[nn.Module, dict]:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=spec.lr)
    loss_fn = nn.MSELoss()

    train_X_t = torch.from_numpy(train_X).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    val_X_t = torch.from_numpy(val_X).float().to(device)
    val_y_t = torch.from_numpy(val_y).float().to(device)

    best_state = None
    best_val = math.inf
    bad = 0

    n = train_X_t.size(0)

    for epoch in range(spec.epochs):
        model.train()
        running = 0.0
        # simple minibatch loop
        perm = torch.randperm(n, device=device)
        for i in range(0, n, spec.batch_size):
            idx = perm[i : i + spec.batch_size]
            xb = train_X_t.index_select(0, idx)
            yb = train_y_t.index_select(0, idx)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)

        # val
        model.eval()
        with torch.no_grad():
            vpred = model(val_X_t)
            vloss = loss_fn(vpred, val_y_t).item()

        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= spec.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_mse": best_val}


def evaluate_regressor(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> dict:
    model.eval()
    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        pred = model(X_t).cpu().numpy()
    # use first step if multi-horizon
    y_true = y
    y_pred = pred
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
    r2 = float(r2_score(y_true.reshape(-1), y_pred.reshape(-1)))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# -----------------------------
# Dataset builders (light)
# -----------------------------


def build_windows(
    frame: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    context: int,
    horizon: int,
    stride: int = 1,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X = frame[feature_cols].values.astype(np.float32)
    y = frame[target_col].values.astype(np.float32)
    T = len(frame)
    idxs = []
    for t in range(context, T - horizon + 1, stride):
        idxs.append((t - context, t, t, t + horizon))
    if limit is not None:
        idxs = idxs[:limit]
    Xw = np.stack([X[s:e, :] for (s, e, _, __) in idxs], axis=0)
    Yw = np.stack([y[s2:e2] for (_, __, s2, e2) in idxs], axis=0)
    return Xw, Yw


def split_train_val_test(X: np.ndarray, y: np.ndarray, val_frac=0.15, test_frac=0.15):
    n = X.shape[0]
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    return (
        X[:n_train], y[:n_train],
        X[n_train:n_train + n_val], y[n_train:n_train + n_val],
        X[n_train + n_val:], y[n_train + n_val:]
    )


def load_jena_light(context=72, horizon=36, subset_days=120) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = DATA_ROOT / "Jena Climate 2009-2016" / "jena_climate_2009_2016.csv"
    if not path.exists():
        raise FileNotFoundError(f"Jena climate CSV not found at {path}")
    df = pd.read_csv(path)
    # Prefer explicit columns; fallback to numerics
    target_col = None
    for c in df.columns:
        if c.strip().lower().startswith("t ") or c.strip().lower().startswith("t(") or "degc" in c.lower():
            target_col = c
            break
    if target_col is None:
        raise RuntimeError("Could not find temperature column for Jena (e.g., 'T (degC)')")

    # Numeric features only; include target
    num_df = df.select_dtypes(include=[np.number]).copy()
    # Light subset: last N days to capture seasonality without full volume (6 * 24 * 10-min = 144 per day)
    # 144 samples/day at 10-min intervals; subset_days default 120 days => ~17k rows
    if subset_days is not None:
        num_df = num_df.tail(subset_days * 144)

    # Z-score normalize features
    num_df = (num_df - num_df.mean()) / (num_df.std().replace(0, 1.0))
    feature_cols = [c for c in num_df.columns]
    Xw, Yw = build_windows(num_df, feature_cols, target_col, context, horizon, stride=1)
    return split_train_val_test(Xw, Yw)


def load_beijing_light(station_name: str = "Guanyuan", context=24, horizon=6, subset_days=120):
    base = DATA_ROOT / "Beijing Air Quality"
    station_file = None
    for p in base.glob("PRSA_Data_*_20130301-20170228.csv"):
        if station_name.lower() in p.name.lower():
            station_file = p
            break
    if station_file is None:
        raise FileNotFoundError(f"Could not find Beijing station file containing '{station_name}' in {base}")
    df = pd.read_csv(station_file)
    # Keep key columns; fallback to numeric only
    cols = [c for c in df.columns]
    target_col = "PM2.5" if "PM2.5" in cols else cols[0]

    # Select numeric features and simple imputation (ffill)
    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df = num_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    if subset_days is not None:
        num_df = num_df.tail(subset_days * 24)  # hourly
    # Per-station standardization
    num_df = (num_df - num_df.mean()) / (num_df.std().replace(0, 1.0))
    feature_cols = [c for c in num_df.columns]
    Xw, Yw = build_windows(num_df, feature_cols, target_col, context, horizon, stride=1)
    return split_train_val_test(Xw, Yw)


def load_eaf_temp_lite(context=16, horizon=1, heats_limit=5, min_rows=100):
    path = DATA_ROOT / "Industrial Data from the Electric Arc Furnace" / "eaf_temp.csv"
    if not path.exists():
        raise FileNotFoundError(f"eaf_temp.csv not found at {path}")
    df = pd.read_csv(path)
    # Expect columns: HEATID, DATETIME, TEMP, VALO2_PPM
    expected = {"HEATID", "DATETIME", "TEMP"}
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing expected columns in eaf_temp.csv: {missing}")
    df["DATETIME"] = pd.to_datetime(df["DATETIME"], errors="coerce")
    df = df.dropna(subset=["DATETIME"]).sort_values(["HEATID", "DATETIME"])  # stable order within heats

    heats = (
        df.groupby("HEATID").size().reset_index(name="n").query("n >= @min_rows").sort_values("n", ascending=False)
    )
    heats = heats.head(heats_limit)["HEATID"].tolist()
    parts = []
    for hid in heats:
        seg = df[df["HEATID"] == hid].copy()
        # Numeric features: TEMP and VALO2_PPM if present (simple AR + one exogenous)
        num_cols = ["TEMP"] + (["VALO2_PPM"] if "VALO2_PPM" in seg.columns else [])
        seg_num = seg[num_cols]
        seg_num = (seg_num - seg_num.mean()) / (seg_num.std().replace(0, 1.0))
        Xw, Yw = build_windows(seg_num, feature_cols=num_cols, target_col="TEMP", context=context, horizon=horizon, stride=1)
        parts.append((Xw, Yw))
    if not parts:
        raise RuntimeError("No EAF heats with sufficient rows found for lite run")
    X = np.concatenate([p[0] for p in parts], axis=0)
    Y = np.concatenate([p[1] for p in parts], axis=0)
    return split_train_val_test(X, Y)


# -----------------------------
# Jacobian participation ratio (optional, light)
# -----------------------------


def jacobian_pr(model: nn.Module, X_sample: np.ndarray, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    x = torch.from_numpy(X_sample).float().to(device)
    x.requires_grad_(True)
    y = model(x)  # (B, H)
    # Sum over outputs to get a scalar
    y_sum = y.sum()
    grads = torch.autograd.grad(y_sum, x, create_graph=False, retain_graph=False)[0]
    J = grads.detach().cpu().numpy().reshape(x.size(0), -1)
    # Compute singular values via SVD on J (may be tall/skinny). Use np.linalg.svd on (n x d).
    # To keep it light, use top singular value via power method proxy: fall back to full SVD for small sizes.
    try:
        s = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        # Fallback to eigh on J J^T
        M = J @ J.T
        evals, _ = np.linalg.eigh(M)
        s = np.sqrt(np.clip(evals, 0, None))[::-1]
    top_sv = float(s[0]) if s.size > 0 else 0.0
    sum_sv = float(s.sum())
    pr = float((sum_sv ** 2) / (np.sum(s ** 2) + 1e-8))
    return top_sv, sum_sv, pr


# -----------------------------
# Orchestrators
# -----------------------------


def run_light_task(task: str, seeds: List[int], device: torch.device, epochs: int, pr_snapshots: bool, metrics_rows: List[dict]):
    if task == "jena":
        train_X, train_y, val_X, val_y, test_X, test_y = load_jena_light(context=72, horizon=36, subset_days=120)
        in_ch = train_X.shape[-1]
        horizon = train_y.shape[-1]
        # Two small models
        specs = [
            ("psann_conv", TrainSpec(model="psann_conv", hidden=48, depth=2, kernel_size=5, epochs=epochs)),
            ("mlp", TrainSpec(model="mlp", hidden=64, depth=2, epochs=epochs)),
        ]
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                if spec.model == "psann_conv":
                    model = PSANNConvSpine(in_ch=in_ch, hidden=spec.hidden, depth=spec.depth, kernel_size=spec.kernel_size, horizon=horizon)
                else:
                    model = MLPRegressor(in_dim=train_X.shape[1] * in_ch, hidden=spec.hidden, depth=spec.depth, out_dim=horizon)
                model, train_info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                row = {
                    "task": "jena_light",
                    "model": name,
                    "seed": seed,
                    "params": count_params(model),
                    **train_info,
                    **test_metrics,
                }
                metrics_rows.append(row)

                if pr_snapshots and name == "psann_conv":
                    # PR at start/mid/end: we approximate by reusing the trained model and simulating checkpoints via partial training
                    # Light approach: compute on three random disjoint mini-batches
                    sample_idx = np.random.choice(test_X.shape[0], size=min(32, test_X.shape[0]), replace=False)
                    Xs = test_X[sample_idx]
                    top_sv, sum_sv, pr = jacobian_pr(model, Xs, device)
                    pr_rows = [{
                        "task": "jena_light",
                        "model": name,
                        "seed": seed,
                        "phase": "end",
                        "top_sv": top_sv,
                        "sum_sv": sum_sv,
                        "pr": pr,
                    }]
                    pr_df = pd.DataFrame(pr_rows)
                    pr_out = RESULTS_ROOT / "jacobian_pr.csv"
                    mode = "a" if pr_out.exists() else "w"
                    pr_df.to_csv(pr_out, index=False, mode=mode, header=(mode == "w"))

    elif task == "beijing":
        train_X, train_y, val_X, val_y, test_X, test_y = load_beijing_light(station_name="Guanyuan", context=24, horizon=6, subset_days=120)
        in_ch = train_X.shape[-1]
        horizon = train_y.shape[-1]
        specs = [
            ("psann_conv", TrainSpec(model="psann_conv", hidden=64, depth=2, kernel_size=5, epochs=epochs)),
            ("mlp", TrainSpec(model="mlp", hidden=96, depth=2, epochs=epochs)),
        ]
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                if spec.model == "psann_conv":
                    model = PSANNConvSpine(in_ch=in_ch, hidden=spec.hidden, depth=spec.depth, kernel_size=spec.kernel_size, horizon=horizon)
                else:
                    model = MLPRegressor(in_dim=train_X.shape[1] * in_ch, hidden=spec.hidden, depth=spec.depth, out_dim=horizon)
                model, train_info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                row = {
                    "task": "beijing_light",
                    "model": name,
                    "seed": seed,
                    "params": count_params(model),
                    **train_info,
                    **test_metrics,
                }
                metrics_rows.append(row)

    elif task == "eaf":
        train_X, train_y, val_X, val_y, test_X, test_y = load_eaf_temp_lite(context=16, horizon=1, heats_limit=5, min_rows=120)
        in_ch = train_X.shape[-1]
        horizon = train_y.shape[-1]
        specs = [
            ("psann_conv", TrainSpec(model="psann_conv", hidden=32, depth=2, kernel_size=3, epochs=max(epochs, 8))),
            ("mlp", TrainSpec(model="mlp", hidden=48, depth=2, epochs=max(epochs, 8))),
        ]
        for seed in seeds:
            seed_all(seed)
            for name, spec in specs:
                if spec.model == "psann_conv":
                    model = PSANNConvSpine(in_ch=in_ch, hidden=spec.hidden, depth=spec.depth, kernel_size=spec.kernel_size, horizon=horizon)
                else:
                    model = MLPRegressor(in_dim=train_X.shape[1] * in_ch, hidden=spec.hidden, depth=spec.depth, out_dim=horizon)
                model, train_info = train_regressor(model, train_X, train_y, val_X, val_y, spec, device)
                test_metrics = evaluate_regressor(model, test_X, test_y, device)
                row = {
                    "task": "eaf_temp_lite",
                    "model": name,
                    "seed": seed,
                    "params": count_params(model),
                    **train_info,
                    **test_metrics,
                }
                metrics_rows.append(row)

    else:
        raise ValueError(f"Unknown task: {task}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, default="jena,beijing,eaf", help="Comma-separated: jena,beijing,eaf")
    parser.add_argument("--seeds", type=int, nargs="*", default=[7, 8], help="Seeds to run")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per model")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    parser.add_argument("--pr-snapshots", action="store_true", help="Record Jacobian PR on Jena (end snapshot)")
    args = parser.parse_args()

    maybe_extract_datasets_zip()
    device = pick_device(args.device)
    print(f"[env] DATA_ROOT={DATA_ROOT}")
    print(f"[env] RESULTS_ROOT={RESULTS_ROOT}")
    print(f"[env] device={device}")

    metrics_rows: List[dict] = []
    for task in [s.strip() for s in args.tasks.split(",") if s.strip()]:
        print(f"[run] task={task}")
        run_light_task(task, seeds=args.seeds, device=device, epochs=args.epochs, pr_snapshots=args.pr_snapshots, metrics_rows=metrics_rows)

    if metrics_rows:
        df = pd.DataFrame(metrics_rows)
        out = RESULTS_ROOT / "metrics.csv"
        df.to_csv(out, index=False)
        print(f"[done] Wrote metrics to {out}")
    else:
        print("[warn] No metrics collected (no tasks run?)")


if __name__ == "__main__":
    main()
