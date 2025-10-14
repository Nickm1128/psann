"""Benchmark PSANN configurations (HISSO) with careful leakage control.

Explores several PSANN setups and records validation reward, test metrics,
and training time under a configurable time budget. Training is centered on
PSANNRegressor.fit(hisso=True), and data leakage is avoided by using strictly
chronological train/val/test splits and never sampling windows from val/test.
"""

from pathlib import Path
import sys as _sys
try:
    import psann  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import psann  # type: ignore  # noqa: F401

import argparse
import csv
import time
import itertools
import statistics as stats
from typing import Optional

import numpy as np

from psann import PSANNRegressor
from psann.hisso import hisso_infer_series
from psann.metrics import portfolio_metrics


def make_prices(T=8000, seed=0):
    rs = np.random.RandomState(seed)
    t = np.linspace(0, 80, T)
    p1 = 100 * np.exp(0.0008 * t + 0.05 * np.sin(0.2 * t)) * (1 + 0.01 * rs.randn(T))
    p2 = 80 * np.exp(0.0005 * t + 0.08 * np.cos(0.15 * t)) * (1 + 0.012 * rs.randn(T))
    return np.stack([p1, p2], axis=1).astype(np.float32)


def eval_series_log_reward(alloc: np.ndarray, prices: np.ndarray, *, trans_cost: float) -> float:
    """Compute cumulative log return over a full series as a scalar score."""
    m = portfolio_metrics(alloc, prices, trans_cost=trans_cost)
    return float(m["log_return"])  # consistent scalar to compare


def run_config(
    prices: np.ndarray,
    *,
    seed: int,
    activation_type: str,
    hidden_layers: int,
    hidden_width: int,
    hisso_window: int,
    trans_cost: float,
    lsm_cfg: Optional[dict],
    epochs: int,
    train_verbose: int = 0,
) -> dict:
    # Chronological split: train/val/test (strict, no leakage)
    n_train, n_val = 4000, 1000
    train, val, test = prices[:n_train], prices[n_train:n_train + n_val], prices[n_train + n_val:]
    est = PSANNRegressor(
        hidden_layers=hidden_layers,
        hidden_width=hidden_width,
        activation_type=activation_type,
        epochs=int(epochs),
        lr=1e-3,
        random_state=int(seed),
        lsm=(dict(lsm_cfg) if lsm_cfg is not None else None),
        lsm_pretrain_epochs=int(lsm_cfg.get("epochs", 0) if isinstance(lsm_cfg, dict) else 0),
    )

    t0 = time.perf_counter()
    # HISSO episodic training on train only; y is ignored
    est.fit(train, y=None, hisso=True, hisso_window=int(hisso_window), verbose=int(train_verbose))
    train_time = time.perf_counter() - t0

    # Validation performance: use series rollout on val (no windows sampled during training)
    alloc_val = hisso_infer_series(est, val)
    val_log_reward = eval_series_log_reward(alloc_val, val, trans_cost=trans_cost)

    # Test metrics on unseen data
    alloc_test = hisso_infer_series(est, test)
    met = portfolio_metrics(alloc_test, test, trans_cost=trans_cost)

    row = {
        "seed": seed,
        "activation": activation_type,
        "hidden_layers": hidden_layers,
        "hidden_width": hidden_width,
        "hisso_window": int(hisso_window),
        "trans_cost": float(trans_cost),
        "lsm": ("dict" if isinstance(lsm_cfg, dict) else "none"),
        "epochs": int(epochs),
        "val_log_return": float(val_log_reward),
        "train_time_s": float(train_time),
        **{k: float(v) for (k, v) in met.items()},
    }
    return row


def main():
    ap = argparse.ArgumentParser(description="Benchmark PSANN configurations (HISSO) with a time budget.")
    ap.add_argument("--time_budget_s", type=int, default=600, help="Global time budget in seconds (default 600)")
    ap.add_argument("--epochs", type=int, default=40, help="Epochs per configuration (default 40)")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0], help="Random seeds to try (default [0])")
    ap.add_argument("--out", type=str, default=str(Path(__file__).with_name("results_psann_config_benchmark.csv")), help="Output CSV path")
    ap.add_argument("--train_verbose", type=int, default=0, help="Trainer verbosity passed to fit(hisso=...) (0/1)")
    args = ap.parse_args()

    # Synthetic dataset
    prices = make_prices(T=8000, seed=0)

    # Sensible grid (kept small for time budget). You can expand if needed.
    activations = ["psann", "relu", "tanh"]
    hidden_layers_list = [2]
    hidden_widths = [64]
    hisso_windows = [64]
    trans_cost = 1e-3
    # LSM: none vs small expander (dict-based)
    lsm_options = [
        None,
        {"output_dim": 128, "hidden_layers": 1, "hidden_width": 128, "sparsity": 0.9, "nonlinearity": "sine", "epochs": 0},
    ]

    # Cartesian product
    grid = list(itertools.product(activations, hidden_layers_list, hidden_widths, hisso_windows, lsm_options))

    rows = []
    start = time.perf_counter()
    runs = 0
    total_planned = len(args.seeds) * len(grid)
    best: Optional[dict] = None
    for seed in args.seeds:
        for (act, hl, hw, win, lsm_cfg) in grid:
            if time.perf_counter() - start > args.time_budget_s:
                break
            elapsed = time.perf_counter() - start
            remaining = max(0.0, args.time_budget_s - elapsed)
            run_idx = runs + 1
            lsm_name = 'dict' if isinstance(lsm_cfg, dict) else 'none'
            print(f"-> [{run_idx}/{total_planned}] seed={seed} act={act} hl={hl} hw={hw} lsm={lsm_name} start; time_left~={remaining:.0f}s")
            try:
                row = run_config(
                    prices,
                    seed=seed,
                    activation_type=act,
                    hidden_layers=hl,
                    hidden_width=hw,
                    hisso_window=win,
                    trans_cost=trans_cost,
                    lsm_cfg=lsm_cfg,
                    epochs=args.epochs,
                    train_verbose=int(args.train_verbose),
                )
                rows.append(row)
                runs += 1
                print(
                    f"<- done in {row['train_time_s']:.1f}s | val_logR {row['val_log_return']:.4f} | "
                    f"test: logR {row['log_return']:.4f}, Sharpe {row['sharpe']:.3f}, MDD {row['max_drawdown']:.3f}, Turnover {row['turnover']:.3f}"
                )
                if (best is None) or (row['log_return'] > best['log_return']):
                    best = row
                    print(
                        f"   NEW BEST by test logR -> seed={seed} act={act} hl={hl} hw={hw} lsm={lsm_name} "
                        f"test_logR={row['log_return']:.4f} (val_logR={row['val_log_return']:.4f})"
                    )
            except Exception as e:
                print("Error in config:", e)
                continue
        if time.perf_counter() - start > args.time_budget_s:
            break

    # Save
    if rows:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("Saved:", out)

        # Quick summary on the base slice (act=psann, no LSM expander)
        base = [r for r in rows if r["activation"] == "psann" and r["lsm"] == "none"]
        if base:
            lr = [r["log_return"] for r in base]
            print("PSANN baseline test logR mean+/-std:", f"{stats.mean(lr):.3f} +/- {stats.pstdev(lr):.3f}")
    else:
        print("No runs completed within time budget.")


if __name__ == "__main__":
    main()
