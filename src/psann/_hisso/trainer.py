from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from ..types import ContextExtractor, NoiseSpec, RewardFn
from .amp import _autocast_context, _guard_cuda_capture
from .config import HISSOTrainerConfig
from .context import _call_context_extractor
from .reward import (
    _align_context_for_reward,
    _compute_reward,
    _default_reward_fn,
    _resolve_reward_kwarg,
)

if TYPE_CHECKING:
    from ..sklearn import PSANNRegressor


class HISSOTrainer:
    """Simple episodic trainer that optimises the primary head via rewards."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        cfg: HISSOTrainerConfig,
        device: torch.device,
        lr: float,
        reward_fn: Optional[RewardFn],
        context_extractor: Optional[ContextExtractor],
        input_noise_std: Optional[float],
        stateful: bool = False,
        state_reset: str = "batch",
        use_amp: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.reward_fn = reward_fn or _default_reward_fn
        self._reward_kwarg = _resolve_reward_kwarg(self.reward_fn)
        self.context_extractor = context_extractor
        self.input_noise_std = float(input_noise_std) if input_noise_std is not None else None
        self.primary_dim = int(cfg.primary_dim)
        self.stateful = bool(stateful)
        state_reset_value = str(state_reset or "batch").lower()
        if state_reset_value not in {"batch", "epoch", "none"}:
            raise ValueError(
                f"Unsupported state_reset '{state_reset}'. Expected one of {{'batch', 'epoch', 'none'}}."
            )
        self.state_reset = state_reset_value
        legacy_episodes_per_epoch = max(1, int(cfg.episodes_per_batch))
        episode_batch_size = int(cfg.resolved_episode_batch_size())
        updates_per_epoch = int(cfg.resolved_updates_per_epoch())
        self.history: list[dict[str, Any]] = []
        self.profile: dict[str, Any] = {
            "device": str(device),
            "epochs": 0,
            "total_time_s": 0.0,
            "episode_length": int(cfg.episode_length),
            "batch_episodes": legacy_episodes_per_epoch,
            "episode_batch_size": episode_batch_size,
            "updates_per_epoch": updates_per_epoch,
            "dataset_bytes": 0,
            "dataset_transfer_batches": 0,
            "dataset_numpy_to_tensor_time_s": 0.0,
            "dataset_transfer_time_s": 0.0,
            "episodes_sampled": 0,
            "episode_view_is_shared": None,
            "episode_time_s_total": 0.0,
            "episode_gather_time_s_total": 0.0,
            "forward_time_s_total": 0.0,
            "reward_time_s_total": 0.0,
            "backward_time_s_total": 0.0,
            "optimizer_time_s_total": 0.0,
            "amp_enabled": False,
            "amp_dtype": None,
        }
        self.optimizer = torch.optim.Adam(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=float(lr),
        )
        self._rng = np.random.default_rng(cfg.random_state)
        self.use_amp = bool(use_amp and device.type == "cuda" and torch.cuda.is_available())
        self.amp_dtype = amp_dtype if amp_dtype is not None else torch.float16
        self.scaler: Optional[Any] = None
        if self.use_amp:
            self.profile["amp_enabled"] = True
            self.profile["amp_dtype"] = str(self.amp_dtype)
            if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            else:  # pragma: no cover - legacy fallback
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def train(
        self,
        X_train: np.ndarray,
        *,
        epochs: int,
        verbose: int,
        lr_max: Optional[float],
        lr_min: Optional[float],
    ) -> None:
        """Optimise the underlying model against sampled HISSO episodes."""

        del verbose, lr_max, lr_min

        data = np.asarray(X_train, dtype=np.float32)
        if data.size == 0:
            raise ValueError("HISSO training requires non-empty inputs.")

        transfer_start = time.perf_counter()
        tensor_cpu = torch.from_numpy(data)
        after_from_numpy = time.perf_counter()
        x_tensor = tensor_cpu.to(self.device)
        after_to_device = time.perf_counter()

        numpy_to_tensor_time = after_from_numpy - transfer_start
        host_to_device_time = after_to_device - after_from_numpy
        self.profile.update(
            {
                "dataset_bytes": int(data.nbytes),
                "dataset_transfer_batches": 1,
                "dataset_numpy_to_tensor_time_s": max(numpy_to_tensor_time, 0.0),
                "dataset_transfer_time_s": max(host_to_device_time, 0.0),
                "episode_batch_size": int(self.cfg.resolved_episode_batch_size()),
                "updates_per_epoch": int(self.cfg.resolved_updates_per_epoch()),
                "episodes_sampled": 0,
                "episode_view_is_shared": None,
                "episode_time_s_total": 0.0,
                "episode_gather_time_s_total": 0.0,
                "forward_time_s_total": 0.0,
                "reward_time_s_total": 0.0,
                "backward_time_s_total": 0.0,
                "optimizer_time_s_total": 0.0,
            }
        )

        episode_len = max(1, int(self.cfg.episode_length))
        total_steps = int(x_tensor.shape[0])
        episode_len = min(episode_len, total_steps)
        single_window_only = total_steps <= episode_len

        self.model.to(self.device)
        self.history.clear()

        with _guard_cuda_capture():
            for epoch_idx in range(max(1, int(epochs))):
                self.model.train()
                self._reset_state_if_needed("epoch")
                epoch_start = time.perf_counter()

                total_reward = 0.0
                episode_count = 0
                epoch_episode_time = 0.0
                updates_per_epoch = (
                    1 if single_window_only else int(self.cfg.resolved_updates_per_epoch())
                )
                for update_idx in range(updates_per_epoch):
                    if single_window_only:
                        episode_batch = 1
                    else:
                        episode_batch = int(self.cfg.episodes_in_update(update_idx))
                    if episode_batch <= 0:
                        continue

                    batch_start = time.perf_counter()
                    self._reset_state_if_needed("batch")
                    gather_start = time.perf_counter()
                    episodes, is_shared = self._sample_episode_batch(
                        x_tensor,
                        total_steps=total_steps,
                        episode_length=episode_len,
                        count=episode_batch,
                    )
                    self.profile["episode_gather_time_s_total"] += (
                        time.perf_counter() - gather_start
                    )

                    inputs = episodes
                    if self.input_noise_std:
                        noise = torch.randn_like(inputs) * float(self.input_noise_std)
                        inputs = inputs + noise

                    context = self._extract_context(inputs)
                    amp_ctx = (
                        _autocast_context(self.device, self.amp_dtype)
                        if self.use_amp
                        else contextlib.nullcontext()
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    with amp_ctx:
                        forward_start = time.perf_counter()
                        model_inputs = inputs.reshape(
                            episode_batch * episode_len, *tuple(inputs.shape[2:])
                        )
                        outputs = self.model(model_inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        if outputs.ndim == 1:
                            outputs = outputs.reshape(-1, 1)
                        if outputs.ndim > 2:
                            outputs = outputs.view(outputs.shape[0], -1)
                        outputs_bt = outputs.reshape(episode_batch, episode_len, -1)
                        primary = self._apply_primary_transform(outputs_bt)
                        self.profile["forward_time_s_total"] += time.perf_counter() - forward_start

                        reward_start = time.perf_counter()
                        reward_tensor = self._coerce_reward(primary, context)
                        self.profile["reward_time_s_total"] += time.perf_counter() - reward_start
                        loss = -reward_tensor.mean()

                    backward_start = time.perf_counter()
                    if self.use_amp:
                        if self.scaler is None:  # pragma: no cover - defensive
                            raise RuntimeError("AMP enabled but GradScaler is unavailable.")
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), 1.0)
                        self.profile["backward_time_s_total"] += (
                            time.perf_counter() - backward_start
                        )
                        optim_start = time.perf_counter()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        clip_grad_norm_(self.model.parameters(), 1.0)
                        self.profile["backward_time_s_total"] += (
                            time.perf_counter() - backward_start
                        )
                        optim_start = time.perf_counter()
                        self.optimizer.step()
                    self.profile["optimizer_time_s_total"] += time.perf_counter() - optim_start
                    self._commit_state_if_any()

                    total_reward += float(reward_tensor.detach().mean().item()) * float(
                        episode_batch
                    )
                    episode_count += int(episode_batch)
                    self.profile["episodes_sampled"] = int(
                        self.profile.get("episodes_sampled", 0)
                    ) + int(episode_batch)

                    if self.profile.get("episode_view_is_shared") is None:
                        self.profile["episode_view_is_shared"] = bool(is_shared)

                    epoch_episode_time += time.perf_counter() - batch_start

                duration = time.perf_counter() - epoch_start

                avg_reward = total_reward / max(episode_count, 1)
                self.history.append(
                    {
                        "epoch": epoch_idx + 1,
                        "reward": avg_reward,
                        "episodes": episode_count,
                    }
                )
                self.profile["total_time_s"] += duration
                self.profile["episode_time_s_total"] += epoch_episode_time

        self.profile["epochs"] = len(self.history)
        if self.profile.get("episode_view_is_shared") is None:
            self.profile["episode_view_is_shared"] = True
        self.model.eval()

    def _reset_state_if_needed(self, cadence: str) -> None:
        if not self.stateful or self.state_reset != cadence:
            return
        if hasattr(self.model, "reset_state"):
            try:
                self.model.reset_state()
            except Exception:
                pass

    def _commit_state_if_any(self) -> None:
        if hasattr(self.model, "commit_state_updates"):
            try:
                self.model.commit_state_updates()
            except Exception:
                pass

    def _sample_episode_batch(
        self,
        x_tensor: torch.Tensor,
        *,
        total_steps: int,
        episode_length: int,
        count: int,
    ) -> tuple[torch.Tensor, bool]:
        if count <= 0:
            raise ValueError("episode batch size must be positive.")
        if total_steps <= episode_length:
            if count == 1:
                return x_tensor[:episode_length].unsqueeze(0), True
            episodes = x_tensor[:episode_length].unsqueeze(0).repeat(count, *([1] * x_tensor.ndim))
            return episodes, False

        max_start = total_steps - episode_length
        starts = self._rng.integers(0, max_start + 1, size=count, endpoint=False)
        starts_arr = np.asarray(starts, dtype=np.int64).reshape(count)
        if count == 1:
            start = int(starts_arr[0])
            return x_tensor[start : start + episode_length].unsqueeze(0), True

        starts_t = torch.from_numpy(starts_arr).to(device=x_tensor.device)
        offsets = torch.arange(episode_length, device=x_tensor.device, dtype=torch.int64)
        indices = starts_t.unsqueeze(1) + offsets.unsqueeze(0)
        gathered = x_tensor.index_select(0, indices.reshape(-1))
        episodes = gathered.reshape(count, episode_length, *tuple(x_tensor.shape[1:]))
        return episodes, False

    def _extract_context(self, inputs: torch.Tensor) -> torch.Tensor:
        return _call_context_extractor(self.context_extractor, inputs)

    def _apply_primary_transform(self, primary: torch.Tensor) -> torch.Tensor:
        transform = (self.cfg.primary_transform or "identity").lower()
        if transform == "identity":
            return primary
        if transform == "softmax":
            return torch.softmax(primary, dim=-1)
        if transform == "tanh":
            return torch.tanh(primary)
        raise ValueError(f"Unsupported primary_transform '{self.cfg.primary_transform}'.")

    def _coerce_reward(self, primary: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        actions = primary
        ctx = context
        if actions.ndim > 3:
            actions = actions.reshape(actions.shape[0], actions.shape[1], -1)
        if ctx.ndim > 3:
            ctx = ctx.reshape(ctx.shape[0], ctx.shape[1], -1)
        if actions.ndim == 1:
            actions = actions.reshape(1, actions.shape[0], 1)
        elif actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if ctx.ndim == 1:
            ctx = ctx.reshape(1, ctx.shape[0], 1)
        elif ctx.ndim == 2:
            if (
                actions.ndim == 3
                and ctx.shape[0] == actions.shape[0]
                and ctx.shape[1] == actions.shape[1]
            ):
                ctx = ctx.unsqueeze(-1)
            elif actions.ndim == 3 and ctx.shape[0] == actions.shape[1]:
                ctx = ctx.unsqueeze(0)
            else:
                ctx = ctx.unsqueeze(0)
        ctx = ctx.to(device=actions.device, dtype=actions.dtype)
        ctx = _align_context_for_reward(actions, ctx)
        reward = _compute_reward(
            self.reward_fn,
            self._reward_kwarg,
            actions,
            ctx,
            transition_penalty=self.cfg.resolved_transition_penalty(),
        )
        if isinstance(reward, torch.Tensor):
            reward_tensor = reward
        else:
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=primary.device)
        if reward_tensor.ndim == 0:
            reward_tensor = reward_tensor.reshape(1)
        elif reward_tensor.ndim > 1:
            reward_tensor = reward_tensor.reshape(reward_tensor.shape[0], -1).mean(dim=1)
        return reward_tensor


def run_hisso_training(
    estimator: "PSANNRegressor",
    X_train_arr: np.ndarray,
    *,
    trainer_cfg: HISSOTrainerConfig,
    lr: float,
    device: torch.device,
    reward_fn: Optional[RewardFn] = None,
    context_extractor: Optional[ContextExtractor] = None,
    lr_max: Optional[float] = None,
    lr_min: Optional[float] = None,
    input_noise_std: Optional[NoiseSpec] = None,
    verbose: int = 0,
    use_amp: bool = False,
    amp_dtype: Optional[torch.dtype] = None,
) -> HISSOTrainer:
    """Instantiate the lightweight HISSO trainer and execute one optimisation run."""

    device_t = device if isinstance(device, torch.device) else torch.device(device)
    trainer = HISSOTrainer(
        estimator.model_,
        cfg=trainer_cfg,
        device=device_t,
        lr=float(lr),
        reward_fn=reward_fn,
        context_extractor=context_extractor,
        input_noise_std=input_noise_std,
        stateful=bool(getattr(estimator, "stateful", False)),
        state_reset=str(getattr(estimator, "state_reset", "batch")),
        use_amp=use_amp,
        amp_dtype=amp_dtype,
    )
    trainer.train(
        X_train_arr,
        epochs=int(estimator.epochs),
        verbose=int(verbose),
        lr_max=lr_max,
        lr_min=lr_min,
    )
    estimator._hisso_reward_fn_ = trainer.reward_fn
    estimator._hisso_context_extractor_ = context_extractor
    return trainer
