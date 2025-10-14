"""
Minimal sine-activated recurrent cell for regime exploration.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from ..initializers import apply_siren_init
from ..layers.film import FiLM


class _RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * (self.weight / rms)


class WaveRNNCell(nn.Module):
    """Sine-activated residual recurrent cell with optional context modulation."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        *,
        alpha: float = 0.2,
        w0: float = 1.0,
        norm: Literal["none", "rms"] = "none",
        use_phase_shift: bool = True,
        use_film: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")

        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.alpha = alpha
        self.w0 = w0
        self.use_phase_shift = use_phase_shift and context_dim is not None
        self.use_film = use_film and context_dim is not None

        self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.context_linear: Optional[nn.Linear] = None
        if context_dim is not None:
            self.context_linear = nn.Linear(context_dim, hidden_dim, bias=False)

        self.phase_shift: Optional[nn.Linear] = None
        if self.use_phase_shift:
            self.phase_shift = nn.Linear(context_dim, hidden_dim, bias=True)

        self.film: Optional[FiLM] = None
        if self.use_film:
            self.film = FiLM(context_dim, hidden_dim)

        self.norm_layer: Optional[nn.Module]
        if norm == "rms":
            self.norm_layer = _RMSNorm(hidden_dim)
        elif norm == "none":
            self.norm_layer = None
        else:
            raise ValueError(f"Unsupported norm '{norm}'.")

        apply_siren_init(self, first_layer_w0=w0, hidden_w0=w0)

    def forward(self, h: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Advance the hidden state by one recurrent step."""
        if c is not None and self.context_dim is None:
            raise ValueError("Context tensor provided but context_dim is None.")

        pre = self.recurrent(h)
        if self.context_linear is not None and c is not None:
            pre = pre + self.context_linear(c)
        if self.use_phase_shift and self.phase_shift is not None and c is not None:
            pre = pre + self.phase_shift(c)

        delta = torch.sin(self.w0 * pre)
        if self.use_film and self.film is not None and c is not None:
            delta = self.film(delta, c)

        h_next = h + self.alpha * delta
        if self.norm_layer is not None:
            h_next = self.norm_layer(h_next)
        return h_next


def scan_regimes(
    cell: WaveRNNCell,
    c_grid: torch.Tensor,
    *,
    steps: int = 512,
    burn_in: int = 128,
) -> list[dict[str, object]]:
    """Evaluate attractor regimes over a grid of context values."""
    if cell.context_dim is None:
        raise ValueError("scan_regimes requires a cell constructed with context_dim.")
    if c_grid.ndim == 1:
        c_grid = c_grid.unsqueeze(1)
    if c_grid.shape[-1] != cell.context_dim:
        raise ValueError("Context dimensionality mismatch.")

    device = next(cell.parameters()).device
    results = []
    prev_training = cell.training
    cell.eval()

    for idx in range(c_grid.shape[0]):
        context = c_grid[idx : idx + 1].to(device)
        h = torch.zeros(1, cell.hidden_dim, device=device)
        trajectory = []

        for t in range(steps):
            h = cell(h, context)
            if t >= burn_in:
                trajectory.append(h.detach().cpu().squeeze(0))

        traj_tensor = (
            torch.stack(trajectory, dim=0)
            if trajectory
            else torch.empty((0, cell.hidden_dim))
        )
        attractors = _estimate_attractors(traj_tensor)
        lyapunov = _estimate_lyapunov(cell, h.detach(), context)

        results.append(
            {
                "context": context.squeeze(0).cpu(),
                "attractor_count": attractors,
                "lyapunov": lyapunov,
                "trajectory": traj_tensor,
            }
        )

    cell.train(prev_training)
    return results


def _estimate_attractors(trajectory: torch.Tensor, tol: float = 1e-3) -> int:
    if trajectory.numel() == 0:
        return 0
    samples = trajectory[-32:].reshape(-1, trajectory.shape[-1])
    quantised = torch.round(samples / tol)
    unique = torch.unique(quantised, dim=0)
    return unique.shape[0]


def _estimate_lyapunov(cell: WaveRNNCell, h: torch.Tensor, c: torch.Tensor) -> float:
    if h.ndim != 2 or h.shape[0] != 1:
        raise ValueError("Hidden state must have shape (1, hidden_dim).")
    h_vec = h.squeeze(0).detach().requires_grad_(True)
    context = c.detach()

    def fn(vec: torch.Tensor) -> torch.Tensor:
        return cell(vec.unsqueeze(0), context).squeeze(0)

    jac = torch.autograd.functional.jacobian(fn, h_vec, vectorize=True)
    eigvals = torch.linalg.eigvals(jac)
    spectral_radius = eigvals.abs().max().real
    return float(torch.log(spectral_radius.clamp_min(1e-9)).item())
