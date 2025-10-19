from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import choose_device, seed_all


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)


def _alloc_transform(
    logits: torch.Tensor, kind: str = "softmax", eps: float = 1e-8
) -> torch.Tensor:
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


class RewardStrategy(Protocol):
    """Callable that maps per-step actions and context to scalar rewards per episode."""

    def __call__(
        self,
        actions: torch.Tensor,
        context: torch.Tensor,
        *,
        transition_penalty: float = 0.0,
    ) -> torch.Tensor: ...


def _resolve_reward_kwarg(reward_fn: Callable[..., torch.Tensor]) -> Optional[str]:
    """Detect which keyword name to use when passing transition penalties."""

    try:
        sig = inspect.signature(reward_fn)
    except (TypeError, ValueError):
        return None

    params = sig.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return "transition_penalty"

    tp = params.get("transition_penalty")
    if tp is not None and tp.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return "transition_penalty"

    legacy = params.get("trans_cost")
    if legacy is not None and legacy.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        return "trans_cost"

    return None


def multiplicative_return_reward(
    actions: torch.Tensor,
    context: torch.Tensor,
    *,
    transition_penalty: Optional[float] = None,
    trans_cost: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Episode reward using multiplicative growth across horizon steps.

    actions: (B, T, M) transformed decisions (e.g., allocations) along final dim.
    context: (B, T, M) reward context aligned with actions (e.g., asset returns).

    Computes per-step growth factors g_t = a_t · (1 + r_t) with returns derived
    from adjacent context slices. The last step has no forward return and is
    ignored for accumulation.

    transition_penalty / trans_cost: proportional L1 change penalty per step.
    Returns a reward tensor of shape (B,).
    """
    if transition_penalty is None:
        transition_penalty = trans_cost if trans_cost is not None else 0.0
    if trans_cost is not None and transition_penalty != trans_cost:
        warnings.warn(
            "Ignoring trans_cost because transition_penalty was provided; prefer the new name.",
            RuntimeWarning,
            stacklevel=2,
        )
    penalty = float(transition_penalty or 0.0)

    assert actions.ndim == 3 and context.ndim == 3, "actions/context must be (B,T,M)"
    B, T, M = actions.shape
    assert context.shape == actions.shape, "actions and context must align"
    if T < 2:
        raise ValueError("Episode length must be >= 2 to compute returns")

    # Compute reward context returns for t -> t+1
    ctx_ret = context[:, 1:, :] / (context[:, :-1, :] + eps) - 1.0  # (B, T-1, M)
    actions_t = actions[:, :-1, :]  # align action at t with return to t+1
    growth = (actions_t * (1.0 + ctx_ret)).sum(dim=-1).clamp_min(eps)  # (B, T-1)
    log_g = torch.log(growth)

    penalty_term = 0.0
    if penalty > 0.0:
        delta = torch.abs(actions[:, 1:, :] - actions[:, :-1, :]).sum(dim=-1)  # (B, T-1)
        penalty_term = penalty * delta

    reward = log_g - (penalty_term if isinstance(penalty_term, torch.Tensor) else 0.0)
    return reward.sum(dim=1)  # (B,)


def portfolio_log_return_reward(
    allocations: torch.Tensor,
    prices: torch.Tensor,
    *,
    trans_cost: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Legacy finance-specific reward helper (delegates to multiplicative_return_reward)."""
    return multiplicative_return_reward(
        allocations,
        prices,
        transition_penalty=trans_cost,
        eps=eps,
    )


@dataclass
class EpisodeConfig:
    episode_length: int
    batch_episodes: int = 32
    allocation_transform: str = "softmax"
    trans_cost: float = 0.0
    transition_penalty: Optional[float] = None
    spatial_pool: str = "mean"  # for segmentation or spatial outputs
    random_state: Optional[int] = None

    def resolved_transition_penalty(self) -> float:
        """Return the configured transition penalty, falling back to legacy trans_cost."""
        penalty = self.transition_penalty
        if penalty is None:
            penalty = self.trans_cost
        return float(penalty)


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
        Function mapping transformed actions and reward context to (B,) rewards.
        Signature: `reward_fn(actions: Tensor(B,T,M), context: Tensor(B,T, ...)) -> Tensor(B,)`.
        A finance-oriented helper (`portfolio_log_return_reward`) remains for backward compatibility; prefer supplying callables that follow the :class:`RewardStrategy` protocol.
        The :mod:`psann.rewards` registry exposes reusable bundles and helpers for common domains.
    - ep_cfg: EpisodeConfig
        Episode sampling and allocation transform options.
    - device: 'auto'|'cpu'|'cuda'|torch.device
    - optimizer: torch optimizer (created if None via Adam)
    - lr: learning rate when creating default optimizer

    Usage
    - Call `train(X_context, epochs=...)` where `X_context` is (N, M) or (N, ..., M)
      context source. The trainer samples random start indices, builds episodes of
      shape (B, T, M), runs the model per-step, maps logits -> allocations, and
      maximizes episode rewards.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        reward_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = multiplicative_return_reward,
        ep_cfg: EpisodeConfig,
        device: torch.device | str = "auto",
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-3,
        grad_clip: Optional[float] = None,
        price_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
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
        if context_extractor is not None:
            self.context_extractor = context_extractor
        else:
            self.context_extractor = price_extractor
            if price_extractor is not None:
                warnings.warn(
                    "EpisodeTrainer(price_extractor=...) is deprecated; use context_extractor instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        self._reward_kwarg = _resolve_reward_kwarg(self.reward_fn)

    def _reset_state_if_any(self):
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

    def _commit_state_if_any(self):
        if hasattr(self.model, "commit_state_updates"):
            self.model.commit_state_updates()

    def _compute_reward(self, actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Evaluate the configured reward, respecting legacy keyword names."""
        kwarg = self._reward_kwarg
        if kwarg is None:
            return self.reward_fn(actions, context)
        penalty = self.cfg.resolved_transition_penalty()
        if kwarg == "transition_penalty":
            return self.reward_fn(actions, context, transition_penalty=penalty)
        if kwarg == "trans_cost":
            return self.reward_fn(actions, context, trans_cost=penalty)
        return self.reward_fn(actions, context, **{kwarg: penalty})

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

        Returns X_ep, ctx_ep as (B,T,...) pairs.
        - If `context_extractor` is provided, it maps X_ep -> ctx_ep.
        - Else, if X is (N, M), we treat last dim as the reward context.
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
        # Map to reward context
        if self.context_extractor is not None:
            ctx_ep = self.context_extractor(X_ep)
            if not isinstance(ctx_ep, torch.Tensor):
                ctx_ep = torch.as_tensor(ctx_ep, dtype=X_ep.dtype, device=X_ep.device)
            if ctx_ep.shape[:2] != X_ep.shape[:2]:
                raise ValueError("context_extractor must return tensors shaped (B,T,...)")
        else:
            # Default: last dimension is assets/prices if 3D
            if X_ep.ndim == 3:
                ctx_ep = X_ep  # (B,T,M)
            else:
                raise ValueError(
                    "Provide context_extractor for multi-dimensional inputs (e.g., conv)."
                )
        return X_ep, ctx_ep

    def train(
        self,
        X: np.ndarray,
        *,
        epochs: int = 100,
        verbose: int = 1,
    ) -> None:
        self.model.train()
        for e in range(epochs):
            X_ep, ctx_ep = self._sample_episodes(X)
            self._reset_state_if_any()
            alloc = self._forward_allocations(X_ep)
            rewards = self._compute_reward(alloc, ctx_ep)
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
            X_ep, ctx_ep = self._sample_episodes(X)
            self._reset_state_if_any()
            alloc = self._forward_allocations(X_ep)
            rew = self._compute_reward(alloc, ctx_ep)
            vals.append(rew.mean().item())
        return float(np.mean(vals))


def make_episode_trainer_from_estimator(
    est,
    *,
    ep_cfg: EpisodeConfig,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = multiplicative_return_reward,
    device: torch.device | str = "auto",
    lr: float = 1e-3,
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
