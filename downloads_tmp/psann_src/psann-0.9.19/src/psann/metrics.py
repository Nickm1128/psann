from __future__ import annotations

import numpy as np
import torch


def equity_curve(
    allocations: np.ndarray | torch.Tensor,
    prices: np.ndarray | torch.Tensor,
    *,
    trans_cost: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute equity curve from allocations (T,M) and prices (T,M).

    Uses portfolio growth g_t = a_t · (1 + r_t) with r_t = p_{t+1}/p_t - 1.
    First step has no forward return and is ignored; curve starts at 1.0.
    Transaction cost: proportional L1 change penalty on allocations per step.
    """
    if isinstance(allocations, torch.Tensor):
        allocations = allocations.detach().cpu().numpy()
    if isinstance(prices, torch.Tensor):
        prices = prices.detach().cpu().numpy()
    A = np.asarray(allocations, dtype=np.float64)
    P = np.asarray(prices, dtype=np.float64)
    assert A.shape == P.shape and A.ndim == 2, "allocations/prices must be (T,M)"
    T, M = A.shape
    if T < 2:
        return np.ones((T,), dtype=np.float64)
    r = P[1:] / (P[:-1] + eps) - 1.0  # (T-1, M)
    a_t = A[:-1]
    growth = np.maximum((a_t * (1.0 + r)).sum(axis=1), eps)
    if trans_cost > 0.0:
        delta = np.abs(A[1:] - A[:-1]).sum(axis=1)
        growth = growth - trans_cost * delta
        growth = np.maximum(growth, eps)
    curve = np.empty((T,), dtype=np.float64)
    curve[0] = 1.0
    curve[1:] = np.cumprod(growth)
    return curve


def sharpe_ratio(returns: np.ndarray, *, risk_free: float = 0.0, eps: float = 1e-12) -> float:
    """Compute simple Sharpe ratio from per-step returns r_t.

    No annualization assumed (step=1); scale externally if needed.
    """
    r = np.asarray(returns, dtype=np.float64)
    excess = r - risk_free
    std = np.std(excess, ddof=1) + eps
    return float(np.mean(excess) / std)


def max_drawdown(curve: np.ndarray) -> float:
    c = np.asarray(curve, dtype=np.float64)
    peak = np.maximum.accumulate(c)
    dd = (c - peak) / np.maximum(peak, 1e-12)
    return float(dd.min())  # negative number


def turnover(allocations: np.ndarray) -> float:
    A = np.asarray(allocations, dtype=np.float64)
    if A.shape[0] < 2:
        return 0.0
    return float(np.abs(A[1:] - A[:-1]).sum(axis=1).mean())


def portfolio_metrics(
    allocations: np.ndarray | torch.Tensor,
    prices: np.ndarray | torch.Tensor,
    *,
    trans_cost: float = 0.0,
) -> dict:
    A = allocations.detach().cpu().numpy() if isinstance(allocations, torch.Tensor) else np.asarray(allocations)
    P = prices.detach().cpu().numpy() if isinstance(prices, torch.Tensor) else np.asarray(prices)
    curve = equity_curve(A, P, trans_cost=trans_cost)
    # Compute per-step arithmetic returns aligned to curve[1:]
    rets = curve[1:] / curve[:-1] - 1.0
    return {
        "cum_return": float(curve[-1] - 1.0),
        "log_return": float(np.log(max(curve[-1], 1e-12))),
        "sharpe": sharpe_ratio(rets),
        "max_drawdown": max_drawdown(curve),
        "turnover": turnover(A),
    }

