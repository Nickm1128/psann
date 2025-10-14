from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import choose_device, seed_all


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def _alloc_transform(logits: torch.Tensor, kind: str = "softmax", eps: float = 1e-8) -> torch.Tensor:
    """Map unconstrained logits to a simplex allocation along the last dim.

    - softmax: standard softmax, strictly positive, sums to 1
    - relu_norm: ReLU + epsilon then L1-normalize (allows near-sparse allocations)
    """
    kind = (kind or "softmax").lower()
    if kind == "softmax":
        return torch.softmax(logits, dim=-1)
    if kind in ("relu_norm", "relu-normalize", "sparse"):
        x = torch.relu(logits) + eps
        return x / (x.sum(dim=-1, keepdim=True) + eps)
    raise ValueError(f"Unknown allocation transform '{kind}'")


def portfolio_log_return_reward(
    allocations: torch.Tensor,
    prices: torch.Tensor,
    *,
    trans_cost: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Episode reward: cumulative log return of a self-financing portfolio.

    allocations: (B, T, M) simplex along last dim
    prices:      (B, T, M) asset prices aligned with allocations

    Uses returns between consecutive timesteps: r_t = p_{t+1}/p_t - 1, for t=0..T-2.
    The final step has no forward return, so it's ignored for reward accumulation.

    trans_cost: proportional L1 change penalty on allocations per step.
    Returns per-episode reward tensor of shape (B,).
    """
    assert allocations.ndim == 3 and prices.ndim == 3, "allocations/prices must be (B,T,M)"
    B, T, M = allocations.shape
    assert prices.shape == allocations.shape, "allocations and prices must align"
    if T < 2:
        raise ValueError("Episode length must be >= 2 to compute returns")

    # Compute asset returns for t -> t+1
    ret = prices[:, 1:, :] / (prices[:, :-1, :] + eps) - 1.0  # (B, T-1, M)
    alloc_t = allocations[:, :-1, :]  # align allocation at t with return to t+1
    # Portfolio growth factor per step: g_t = a_t · (1 + r_t)
    growth = (alloc_t * (1.0 + ret)).sum(dim=-1).clamp_min(eps)  # (B, T-1)
    log_g = torch.log(growth)

    # Transaction cost: penalize allocation changes
    tc = 0.0
    if trans_cost > 0.0:
        delta = torch.abs(allocations[:, 1:, :] - allocations[:, :-1, :]).sum(dim=-1)  # (B, T-1)
        tc = trans_cost * delta

    # Sum over time; higher is better
    rew = log_g - (tc if isinstance(tc, torch.Tensor) else 0.0)
    return rew.sum(dim=1)  # (B,)


@dataclass
class EpisodeConfig:
    episode_length: int
    batch_episodes: int = 32
    allocation_transform: str = "softmax"
    trans_cost: float = 0.0
    spatial_pool: str = "mean"  # for segmentation or spatial outputs
    random_state: Optional[int] = None


class EpisodeTrainer:
    """Episode-based training for strategic sequence optimization.

    Trains a `torch.nn.Module` to maximize a differentiable reward over episodes
    (windows) of length `T`. Designed to work with PSANN+LSM models but kept
    separate from the sklearn wrapper.

    Parameters
    - model: nn.Module
        Module mapping inputs to raw allocation logits (unconstrained). For
        PSANNRegressor, pass `estimator.model_` after construction.
    - reward_fn: Callable
        Function of allocations and episode context returning (B,) rewards.
        Signature: `reward_fn(allocations: Tensor(B,T,M), prices_or_ctx: Tensor(B,T,M or ...)) -> Tensor(B,)`.
        A default `portfolio_log_return_reward` is provided for portfolio tasks.
    - ep_cfg: EpisodeConfig
        Episode sampling and allocation transform options.
    - device: 'auto'|'cpu'|'cuda'|torch.device
    - optimizer: torch optimizer (created if None via Adam)
    - lr: learning rate when creating default optimizer

    Usage
    - Call `train(X_prices, epochs=...)` where `X_prices` is (N, M) or (N, ..., M)
      episode source. The trainer samples random start indices, builds episodes of
      shape (B, T, M), runs the model per-step, maps logits -> allocations, and
      maximizes episode rewards.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = portfolio_log_return_reward,
        ep_cfg: EpisodeConfig,
        device: torch.device | str = "auto",
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        grad_clip: Optional[float] = None,
        price_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.cfg = ep_cfg
        self.grad_clip = grad_clip
        self.device = choose_device(device)
        self.model.to(self.device)
        self.opt = optimizer or torch.optim.Adam(self.model.parameters(), lr=lr)
        seed_all(self.cfg.random_state)
        self.price_extractor = price_extractor

    def _reset_state_if_any(self):
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

    def _commit_state_if_any(self):
        if hasattr(self.model, "commit_state_updates"):
            self.model.commit_state_updates()

    def _forward_allocations(self, X_ep: torch.Tensor) -> torch.Tensor:
        """Run model over episodes, return allocations (B,T,M)."""
        B, T = X_ep.shape[0], X_ep.shape[1]
        # Flatten time into batch for generic modules: (B*T, ...)
        X_bt = X_ep.reshape(B * T, *X_ep.shape[2:])
        logits = self.model(X_bt)
        if logits.ndim == 1:
            logits = logits[:, None]
        # If model outputs segmentation/spatial maps (N, C, H, W ...), pool spatial dims
        if logits.ndim >= 3:
            # Assume channels-first: (N, C, *spatial)
            if self.cfg.spatial_pool == "mean":
                for _ in range(logits.ndim - 2):
                    logits = logits.mean(dim=-1)
            elif self.cfg.spatial_pool == "max":
                for _ in range(logits.ndim - 2):
                    logits = logits.amax(dim=-1)
            else:
                raise ValueError(f"Unknown spatial_pool '{self.cfg.spatial_pool}'")
        M = logits.shape[-1]
        logits_bt = logits.reshape(B, T, M)
        alloc = _alloc_transform(logits_bt, kind=self.cfg.allocation_transform)
        return alloc

    def _sample_episodes(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a minibatch of episodes from a long sequence array.

        Returns X_ep, prices_ep as (B,T,...) and (B,T,M) respectively.
        - If `price_extractor` is provided, it maps X_ep -> prices_ep.
        - Else, if X is (N, M), we treat last dim as asset prices.
        """
        N = X.shape[0]
        T = int(self.cfg.episode_length)
        if N < T:
            raise ValueError(f"Need at least {T} timesteps (got {N})")
        B = int(self.cfg.batch_episodes)
        # Random start indices in [0, N-T]
        starts = np.random.randint(0, N - T + 1, size=B)
        # Build batch episodes
        ep_list = [X[s : s + T] for s in starts]
        X_ep_np = np.stack(ep_list, axis=0).astype(np.float32)
        X_ep = _to_tensor(X_ep_np, self.device)
        # Map to prices
        if self.price_extractor is not None:
            prices_ep = self.price_extractor(X_ep)
            if not isinstance(prices_ep, torch.Tensor):
                prices_ep = torch.as_tensor(prices_ep, dtype=X_ep.dtype, device=X_ep.device)
            if prices_ep.shape[:2] != X_ep.shape[:2]:
                raise ValueError("price_extractor must return (B,T,M)")
        else:
            # Default: last dimension is assets/prices if 3D
            if X_ep.ndim == 3:
                prices_ep = X_ep  # (B,T,M)
            else:
                raise ValueError(
                    "Provide price_extractor for multi-dimensional inputs (e.g., conv)."
                )
        return X_ep, prices_ep

    def train(
        self,
        X: np.ndarray,
        *,
        epochs: int = 100,
        verbose: int = 1,
    ) -> None:
        self.model.train()
        for e in range(epochs):
            X_ep, prices_ep = self._sample_episodes(X)
            self._reset_state_if_any()
            alloc = self._forward_allocations(X_ep)
            rewards = self.reward_fn(alloc, prices_ep, trans_cost=self.cfg.trans_cost)
            # Maximize reward -> minimize negative
            loss = -rewards.mean()
            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.opt.step()
            self._commit_state_if_any()
            if verbose:
                print(f"[EpisodeTrainer] epoch {e+1}/{epochs}  reward={rewards.mean().item():.6f}")

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, *, n_batches: int = 16) -> float:
        self.model.eval()
        vals = []
        for _ in range(n_batches):
            X_ep, prices_ep = self._sample_episodes(X)
            self._reset_state_if_any()
            alloc = self._forward_allocations(X_ep)
            rew = self.reward_fn(alloc, prices_ep, trans_cost=self.cfg.trans_cost)
            vals.append(rew.mean().item())
        return float(np.mean(vals))


def make_episode_trainer_from_estimator(
    est, *, ep_cfg: EpisodeConfig, reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = portfolio_log_return_reward, device: torch.device | str = "auto", lr: float = 1e-3
) -> EpisodeTrainer:
    """Helper to create an EpisodeTrainer from a fitted PSANNRegressor.

    Example
        model = PSANNRegressor(..., output_shape=(M,))
        trainer = make_episode_trainer_from_estimator(model, ep_cfg=EpisodeConfig(episode_length=64))
    """
    if not hasattr(est, "model_"):
        raise RuntimeError("Estimator not fitted; call fit() first or attach .model_ manually.")
    trainer = EpisodeTrainer(est.model_, reward_fn=reward_fn, ep_cfg=ep_cfg, device=device, lr=lr)
    return trainer
