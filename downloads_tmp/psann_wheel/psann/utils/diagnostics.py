"""
Diagnostics for conditioning and expressivity analysis.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn


def jacobian_spectrum(
    model: nn.Module,
    x: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    *,
    topk: int = 32,
) -> Dict[str, object]:
    """Estimate the leading eigenvalues of ``JᵀJ``."""
    jac = _compute_jacobian_matrix(model, x, c)
    gram = jac.T @ jac
    eigvals = torch.linalg.eigvalsh(gram)
    eigvals = torch.clamp(eigvals, min=0.0).flip(0)
    top_vals = eigvals[: min(topk, eigvals.numel())]
    condition = float(
        (top_vals[0] / top_vals[-1]).item() if top_vals.numel() > 1 and top_vals[-1] > 0 else math.inf
    )
    return {
        "top_eigs": top_vals.cpu(),
        "trace": float(eigvals.sum().item()),
        "condition_number": condition,
    }


def ntk_eigens(
    model: nn.Module,
    x: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    *,
    topk: int = 32,
) -> Dict[str, object]:
    """Compute empirical NTK eigenvalues via ``JJᵀ``."""
    jac = _compute_jacobian_matrix(model, x, c)
    kernel = jac @ jac.T
    eigvals = torch.linalg.eigvalsh(kernel)
    eigvals = torch.clamp(eigvals, min=0.0).flip(0)
    top_vals = eigvals[: min(topk, eigvals.numel())]
    return {
        "top_eigs": top_vals.cpu(),
        "trace": float(eigvals.sum().item()),
    }


def participation_ratio(features: torch.Tensor, eps: float = 1e-6) -> float:
    """Return the participation ratio of a feature covariance matrix."""
    if features.ndim != 2:
        raise ValueError("features must have shape (N, D)")
    mean_centered = features - features.mean(dim=0, keepdim=True)
    cov = mean_centered.T @ mean_centered / max(1, features.shape[0] - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    total = eigvals.sum()
    if total <= eps:
        return 0.0
    numerator = total.pow(2)
    denominator = eigvals.pow(2).sum().clamp_min(eps)
    return float((numerator / denominator).item())


def mutual_info_proxy(
    features: torch.Tensor,
    contexts: torch.Tensor,
    *,
    sigma: Optional[float] = None,
) -> float:
    """HSIC-based proxy for mutual information between features and contexts."""
    if features.shape[0] != contexts.shape[0]:
        raise ValueError("features and contexts must share the first dimension.")
    K = _rbf_kernel(features, sigma=sigma)
    L = _rbf_kernel(contexts, sigma=sigma)
    hsic = _hsic(K, L)
    return float(hsic.item())


def _compute_jacobian_matrix(
    model: nn.Module, x: torch.Tensor, c: Optional[torch.Tensor]
) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    c_detached = c.detach() if c is not None else None

    def forward_fn(inp: torch.Tensor) -> torch.Tensor:
        out = model(inp, c_detached) if c is not None else model(inp)
        return out.reshape(-1)

    y = forward_fn(x)
    jac = torch.autograd.functional.jacobian(forward_fn, x, vectorize=True)
    return jac.reshape(y.numel(), -1)


def _rbf_kernel(x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    dist_sq = torch.cdist(x, x, p=2).pow(2)
    if sigma is None:
        median = torch.median(dist_sq[dist_sq > 0])
        sigma = torch.sqrt(median / 2) if median > 0 else 1.0
    gamma = 1.0 / (2 * sigma * sigma)
    return torch.exp(-gamma * dist_sq)


def _hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    H = torch.eye(n, device=K.device) - torch.full((n, n), 1.0 / n, device=K.device)
    HKH = H @ K @ H
    HLH = H @ L @ H
    return (HKH * HLH).sum() / ((n - 1) ** 2)
