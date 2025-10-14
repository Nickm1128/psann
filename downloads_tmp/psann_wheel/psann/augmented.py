from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import choose_device, seed_all


def _as_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device=device, dtype=dtype)


def _apply_transform(x: torch.Tensor, kind: str, eps: float = 1e-8) -> torch.Tensor:
    k = (kind or "identity").lower()
    if k == "identity":
        return x
    if k == "softmax":
        return torch.softmax(x, dim=-1)
    if k == "tanh":
        return torch.tanh(x)
    if k == "sigmoid":
        return torch.sigmoid(x)
    if k == "relu_norm":
        y = torch.relu(x) + eps
        return y / (y.sum(dim=-1, keepdim=True) + eps)
    raise ValueError(f"Unknown transform '{kind}'")


@dataclass
class PredictiveExtrasConfig:
    episode_length: int
    batch_episodes: int = 16
    primary_dim: int = 1            # first outputs used for reward
    extras_dim: int = 1             # last K outputs predict next extras
    primary_transform: str = "softmax"   # map primary logits -> allocations
    extras_transform: str = "tanh"       # bound extras in (-1,1)
    random_state: Optional[int] = None
    extras_l2: float = 0.0          # regularize extras magnitudes
    extras_smooth: float = 0.0      # regularize changes over time
    trans_cost: float = 0.0         # passed to reward if applicable
    extras_supervision_weight: float = 0.0  # weight for extras reconstruction
    extras_supervision_mode: str = "joint"  # joint | alternate
    extras_supervision_cycle: int = 2           # periods for alternate schedule


class PredictiveExtrasTrainer:
    '''Episode trainer where the model predicts next-step extras.

    The model is assumed to output `primary_dim + extras_dim` values per step:
    - primary: used for reward (e.g., allocation logits, then transformed)
    - extras: transformed to produce next-step extras, concatenated to inputs

    You must provide observed feature episodes X of shape (N, F) and set the
    estimator to accept inputs of shape (F + extras_dim).
    '''

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        cfg: PredictiveExtrasConfig,
        device: torch.device | str = "auto",
        lr: float = 1e-3,
        optimizer: Optional[torch.optim.Optimizer] = None,
        grad_clip: Optional[float] = None,
        context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_noise_std: Optional[float] = None,
        noise_decay: float = 1.0,
        extras_cache: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.cfg = cfg
        self.device = choose_device(device)
        self.model.to(self.device)

        self.primary_dim = int(self.cfg.primary_dim)
        self.extras_dim = int(self.cfg.extras_dim)

        params: list[nn.Parameter] = list(self.model.parameters())
        self.initial_extras: Optional[nn.Parameter]
        if self.extras_dim > 0:
            init_vec = torch.zeros(self.extras_dim, device=self.device)
            self.initial_extras = nn.Parameter(init_vec)
            params.append(self.initial_extras)
        else:
            self.initial_extras = None

        if optimizer is None:
            self.opt = torch.optim.Adam(params, lr=lr)
        else:
            self.opt = optimizer
            missing: list[nn.Parameter] = []
            if self.initial_extras is not None:
                in_group = any(
                    self.initial_extras is p
                    for group in self.opt.param_groups
                    for p in group["params"]
                )
                if not in_group:
                    missing.append(self.initial_extras)
            if missing:
                base_lr = self.opt.param_groups[0].get("lr", lr) if self.opt.param_groups else lr
                self.opt.add_param_group({"params": missing, "lr": base_lr})

        self.grad_clip = grad_clip
        self.context_extractor = context_extractor
        seed_all(self.cfg.random_state)
        self.input_noise_std = float(input_noise_std) if input_noise_std is not None else None
        self.noise_decay = float(noise_decay)
        self.history: list[dict] = []
        self.profile: Dict[str, Any] = {
            "device": str(self.device),
            "epochs": 0,
            "total_time_s": 0.0,
            "episode_length": int(self.cfg.episode_length),
            "batch_episodes": int(self.cfg.batch_episodes),
        }

        self.extras_cache: Optional[torch.Tensor] = None
        if self.extras_dim > 0 and extras_cache is not None:
            self.extras_cache = self._coerce_cache_tensor(extras_cache)
            self._sync_initial_extras()

    def _coerce_cache_tensor(self, value: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.from_numpy(np.asarray(value, dtype=np.float32)).to(self.device)
        if tensor.ndim != 2 or tensor.shape[1] != self.extras_dim:
            raise ValueError("extras_cache has incompatible shape")
        return tensor.contiguous()

    def _sync_initial_extras(self) -> None:
        if self.initial_extras is None or self.extras_cache is None or self.extras_cache.shape[0] == 0:
            return
        self.extras_cache[0].copy_(self.initial_extras.detach())

    def _ensure_extras_cache(self, length: int) -> None:
        if self.extras_dim <= 0:
            self.extras_cache = None
            return
        size = int(length) + 1
        if self.extras_cache is None:
            cache = torch.randn((size, self.extras_dim), device=self.device, dtype=torch.float32) * 0.1
            cache[0].zero_()
            self.extras_cache = cache
        else:
            current = self.extras_cache
            if current.shape[0] < size:
                pad = torch.randn(
                    (size - current.shape[0], self.extras_dim),
                    device=self.device,
                    dtype=torch.float32,
                ) * 0.1
                self.extras_cache = torch.cat([current, pad], dim=0)
            elif current.shape[0] > size:
                self.extras_cache = current[:size].contiguous()
        self._sync_initial_extras()

    def _reset_state_if_any(self) -> None:
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

    def _commit_state_if_any(self) -> None:
        if hasattr(self.model, "commit_state_updates"):
            self.model.commit_state_updates()

    def _noise_std_for_epoch(self, epoch_idx: Optional[int]) -> Optional[float]:
        if self.input_noise_std is None:
            return None
        if epoch_idx is None:
            return self.input_noise_std
        power = max(int(epoch_idx), 0)
        return float(self.input_noise_std * (self.noise_decay ** power))

    def _sample_batch(self, episodes: torch.Tensor, epoch_idx: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        num_windows = episodes.shape[0]
        if num_windows <= 0:
            raise ValueError("Need at least one episode window to sample.")
        batch_size = int(self.cfg.batch_episodes)
        starts = torch.randint(0, num_windows, (batch_size,), device=self.device)
        batch = episodes.index_select(0, starts)
        noise_std = self._noise_std_for_epoch(epoch_idx)
        if noise_std is not None and noise_std > 0:
            batch = batch + torch.randn_like(batch) * noise_std
        return batch, starts

    def _rollout(self, X_ep: torch.Tensor, E0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = X_ep.shape
        P = self.primary_dim
        K = self.extras_dim
        if K > 0:
            extras_current = E0 if E0 is not None else torch.zeros((B, K), device=self.device, dtype=X_ep.dtype)
        else:
            extras_current = torch.zeros((B, 0), device=self.device, dtype=X_ep.dtype)
        extras_seq = [extras_current]
        primaries: list[torch.Tensor] = []
        self._reset_state_if_any()
        for t in range(T):
            xt = torch.cat([X_ep[:, t, :], extras_seq[-1]], dim=-1)
            yt = self.model(xt)
            if yt.ndim == 1:
                yt = yt.unsqueeze(0)
            y_primary = yt[:, :P]
            primaries.append(_apply_transform(y_primary, self.cfg.primary_transform))
            if K > 0:
                y_extras = yt[:, P:P + K]
                next_extras = _apply_transform(y_extras, self.cfg.extras_transform)
            else:
                next_extras = extras_seq[-1]
            extras_seq.append(next_extras)
        primaries_t = torch.stack(primaries, dim=1)
        if K > 0:
            extras_t = torch.stack(extras_seq, dim=1)
        else:
            extras_t = torch.zeros((B, T + 1, 0), device=self.device, dtype=X_ep.dtype)
        self._commit_state_if_any()
        return primaries_t, extras_t

    def _gather_targets(self, starts: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        if self.extras_cache is None:
            raise RuntimeError("extras_cache has not been initialised")
        idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        flat_idx = torch.clamp(idx.reshape(-1), max=self.extras_cache.shape[0] - 1)
        gathered = self.extras_cache.index_select(0, flat_idx)
        return gathered.view(starts.shape[0], offsets.shape[0], self.extras_dim)

    def _update_cache_sequences(self, starts: torch.Tensor, extras: torch.Tensor) -> None:
        if self.extras_cache is None or self.extras_dim <= 0 or extras.numel() == 0:
            return
        offsets = torch.arange(extras.shape[1], device=starts.device, dtype=torch.long)
        idx = starts.unsqueeze(1) + offsets.unsqueeze(0)
        flat_idx = torch.clamp(idx.reshape(-1), max=self.extras_cache.shape[0] - 1)
        flat_vals = extras.detach().reshape(-1, self.extras_dim)
        self.extras_cache.index_copy_(0, flat_idx, flat_vals)
        self._sync_initial_extras()

    def train(
        self,
        X_obs: np.ndarray,
        *,
        epochs: int = 100,
        verbose: int = 0,
        lr_max: Optional[float] = None,
        lr_min: Optional[float] = None,
    ) -> None:
        X_obs = np.asarray(X_obs, dtype=np.float32)
        if X_obs.ndim != 2:
            raise ValueError("X_obs must be (N, F) for predictive extras training")
        series = _as_tensor(X_obs, self.device)
        N = int(series.shape[0])
        T = int(self.cfg.episode_length)
        if N < T:
            raise ValueError(f"Need at least {T} timesteps (got {N})")
        episodes = series.unfold(0, T, 1).permute(0, 2, 1)
        self._ensure_extras_cache(N)
        offsets_full = torch.arange(0, T + 1, device=self.device, dtype=torch.long)
        target_offsets = offsets_full[1:]

        extras_weight = float(getattr(self.cfg, "extras_supervision_weight", 0.0) or 0.0)
        extras_mode = (getattr(self.cfg, "extras_supervision_mode", "joint") or "joint").lower()
        extras_cycle = max(1, int(getattr(self.cfg, "extras_supervision_cycle", 1) or 1))
        if extras_mode not in {"joint", "alternate"}:
            raise ValueError("extras_supervision_mode must be 'joint' or 'alternate'")

        for e in range(epochs):
            if lr_max is not None and lr_min is not None:
                if epochs <= 1:
                    lr_e = float(lr_min)
                else:
                    frac = float(e) / float(max(epochs - 1, 1))
                    lr_e = float(lr_max) + (float(lr_min) - float(lr_max)) * frac
                for group in self.opt.param_groups:
                    group["lr"] = lr_e
            t0 = time.perf_counter()
            X_ep, starts = self._sample_batch(episodes, epoch_idx=e)
            K = self.extras_dim
            E0 = None
            if K > 0:
                self._ensure_extras_cache(N)
                start_state = self.extras_cache.index_select(0, starts)
                if self.initial_extras is not None:
                    base_init = self.initial_extras.unsqueeze(0).expand_as(start_state)
                    E0 = base_init + (start_state - base_init).detach()
                else:
                    E0 = start_state
            primary, extras = self._rollout(X_ep, E0=E0)
            ctx = self.context_extractor(X_ep) if self.context_extractor is not None else X_ep
            rewards = self.reward_fn(primary, ctx)
            loss_reward = -rewards.mean()
            reward_loss_value = float(loss_reward.detach().cpu().item())
            train_reward_value = -reward_loss_value
            extras_sup_loss = None
            extras_sup_value = None
            extras_step = False
            loss = loss_reward
            if K > 0 and extras_weight > 0.0:
                targets_t = self._gather_targets(starts, target_offsets)
                extras_sup_loss = F.mse_loss(extras[:, 1:, :], targets_t)
                extras_sup_value = float(extras_sup_loss.detach().cpu().item())
                if extras_mode == "joint":
                    loss = loss_reward + extras_weight * extras_sup_loss
                else:
                    if extras_cycle <= 1:
                        loss = extras_weight * extras_sup_loss
                        extras_step = True
                    else:
                        step_in_cycle = e % extras_cycle
                        if step_in_cycle == 0:
                            loss = extras_weight * extras_sup_loss
                            extras_step = True
                        else:
                            loss = loss_reward
            if self.cfg.extras_l2 > 0 and K > 0:
                loss = loss + self.cfg.extras_l2 * extras[:, 1:, :].pow(2).mean()
            if self.cfg.extras_smooth > 0 and K > 0:
                dE = extras[:, 1:, :] - extras[:, :-1, :]
                loss = loss + self.cfg.extras_smooth * dE.pow(2).mean()

            self.opt.zero_grad()
            loss.backward()
            if self.grad_clip is not None and self.grad_clip > 0:
                params_for_clip = [p for group in self.opt.param_groups for p in group["params"] if p.grad is not None]
                if params_for_clip:
                    torch.nn.utils.clip_grad_norm_(params_for_clip, self.grad_clip)
            self.opt.step()

            with torch.no_grad():
                if K > 0:
                    self._update_cache_sequences(starts, extras)

            dt = time.perf_counter() - t0
            self.profile["device"] = str(self.device)
            self.profile["epochs"] = int(self.profile.get("epochs", 0)) + 1
            self.profile["total_time_s"] = float(self.profile.get("total_time_s", 0.0) + dt)
            self.profile["episode_length"] = int(self.cfg.episode_length)
            self.profile["batch_episodes"] = int(self.cfg.batch_episodes)
            rec = {
                "epoch": len(self.history) + 1,
                "train_reward": train_reward_value,
                "reward_loss": reward_loss_value,
                "time_s": float(dt),
            }
            if extras_sup_loss is not None:
                rec["extras_loss"] = extras_sup_value
                rec["loss_phase"] = "joint" if extras_mode == "joint" else ("extras" if extras_step else "reward")
            else:
                rec["loss_phase"] = "reward"
            if lr_max is not None and lr_min is not None:
                rec["lr"] = float(self.opt.param_groups[0].get("lr", 0.0))
            self.history.append(rec)
            if verbose:
                msg = f"[PredictiveExtras] epoch {e+1}/{epochs}"
                if lr_max is not None and lr_min is not None:
                    msg += f" lr={rec['lr']:.6g}"
                msg += f" reward={train_reward_value:.6f}"
                if extras_sup_loss is not None:
                    phase = "joint" if extras_mode == "joint" else ("extras" if extras_step else "reward")
                    msg += f" extras_loss={extras_sup_value:.6f} phase={phase}"
                print(msg)

    @torch.no_grad()
    def infer_series(self, X_obs: np.ndarray, *, E0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        '''Roll out predictions and extras over the full series.

        Returns (primary, extras) as numpy arrays with shapes (N, P) and (N+1, K).
        '''
        self.model.eval()
        X = _as_tensor(X_obs, self.device)
        if X.ndim != 2:
            raise ValueError("X_obs must be (N, F) for series inference")
        N, _ = X.shape
        P = self.primary_dim
        K = self.extras_dim
        self._ensure_extras_cache(N)
        if K > 0:
            if E0 is not None:
                e0 = _as_tensor(E0, self.device).reshape(1, K)
            elif self.extras_cache is not None and self.extras_cache.shape[0] > 0:
                self._sync_initial_extras()
                e0 = self.extras_cache[0:1].clone()
            elif self.initial_extras is not None:
                e0 = self.initial_extras.detach().unsqueeze(0)
            else:
                e0 = torch.zeros((1, K), device=self.device, dtype=X.dtype)
        else:
            e0 = torch.zeros((1, 0), device=self.device, dtype=X.dtype)
        prim = []
        extras = [e0[0]]
        self._reset_state_if_any()
        for t in range(N):
            xt = torch.cat([X[t : t + 1], extras[-1].reshape(1, -1)], dim=-1)
            yt = self.model(xt)
            y_primary = yt[:, :P]
            prim.append(_apply_transform(y_primary, self.cfg.primary_transform)[0])
            if K > 0:
                y_extras = yt[:, P:P + K]
                next_E = _apply_transform(y_extras, self.cfg.extras_transform)[0]
            else:
                next_E = extras[-1]
            extras.append(next_E)
        self._commit_state_if_any()
        primaries_t = torch.stack(prim, dim=0)
        extras_t = torch.stack(extras, dim=0)
        if K > 0 and self.extras_cache is not None:
            length = min(self.extras_cache.shape[0], extras_t.shape[0])
            self.extras_cache[:length] = extras_t[:length]
            self._sync_initial_extras()
        return primaries_t.cpu().numpy(), extras_t.cpu().numpy()

    @torch.no_grad()
    def evaluate_reward(self, X_obs: np.ndarray, *, n_batches: int = 8) -> float:
        self.model.eval()
        X_obs = np.asarray(X_obs, dtype=np.float32)
        if X_obs.ndim != 2:
            raise ValueError("X_obs must be (N, F) for predictive extras training")
        series = _as_tensor(X_obs, self.device)
        N = int(series.shape[0])
        T = int(self.cfg.episode_length)
        if N < T:
            raise ValueError(f"Need at least {T} timesteps (got {N})")
        episodes = series.unfold(0, T, 1).permute(0, 2, 1)
        self._ensure_extras_cache(N)
        values: list[float] = []
        for _ in range(n_batches):
            X_ep, starts = self._sample_batch(episodes)
            K = self.extras_dim
            if K > 0 and self.extras_cache is not None:
                start_state = self.extras_cache.index_select(0, starts)
                if self.initial_extras is not None:
                    base_init = self.initial_extras.unsqueeze(0).expand_as(start_state)
                    E0 = base_init + (start_state - base_init).detach()
                else:
                    E0 = start_state
            else:
                E0 = None
            primary, extras = self._rollout(X_ep, E0=E0)
            if K > 0:
                self._update_cache_sequences(starts, extras)
            ctx = self.context_extractor(X_ep) if self.context_extractor is not None else X_ep
            values.append(float(self.reward_fn(primary, ctx).mean().item()))
        return float(np.mean(values))

    def profile_summary(self) -> Dict[str, Any]:
        summary = dict(self.profile)
        epochs = int(summary.get("epochs", 0))
        total = float(summary.get("total_time_s", 0.0))
        if epochs > 0:
            summary["avg_epoch_time_s"] = total / epochs if epochs else 0.0
        else:
            summary["avg_epoch_time_s"] = 0.0
        return summary

    def load_extras_cache(self, extras_cache: Optional[Union[np.ndarray, torch.Tensor]]) -> None:
        if self.extras_dim <= 0:
            self.extras_cache = None
            return
        if extras_cache is None:
            return
        cache_tensor = self._coerce_cache_tensor(extras_cache)
        self.extras_cache = cache_tensor
        if self.initial_extras is not None and cache_tensor.shape[0] > 0:
            with torch.no_grad():
                self.initial_extras.copy_(cache_tensor[0])
        self._sync_initial_extras()


    def export_extras_cache(self) -> Optional[np.ndarray]:
        if self.extras_cache is None:
            return None
        return self.extras_cache.detach().cpu().numpy()
def make_predictive_extras_trainer_from_estimator(
    est, *, cfg: PredictiveExtrasConfig, reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], device: torch.device | str = "auto", lr: float = 1e-3, context_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> PredictiveExtrasTrainer:
    if not hasattr(est, "model_"):
        raise RuntimeError("Estimator not fitted; call fit() first.")
    return PredictiveExtrasTrainer(
        est.model_,
        reward_fn,
        cfg=cfg,
        device=device,
        lr=lr,
        context_extractor=context_extractor,
        extras_cache=getattr(est, "_hisso_extras_cache_", None),
    )


