from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:  # Optional scikit-learn import for API compatibility
    from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
    from sklearn.metrics import r2_score as _sk_r2_score  # type: ignore
except Exception:  # Fallbacks if sklearn isn't installed at runtime

    class BaseEstimator:  # minimal stub
        def get_params(self, deep: bool = True):
            # Return non-private, non-callable attributes
            params = {}
            for k, v in self.__dict__.items():
                if k.endswith("_"):
                    continue
                if not k.startswith("_") and not callable(v):
                    params[k] = v
            return params

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class RegressorMixin:
        pass

    def _sk_r2_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        return 1.0 - (u / v if v != 0 else np.nan)


from ..hisso import HISSOOptions, HISSOTrainerConfig, ensure_hisso_trainer_config
from ..layers import SpectralGate1D
from ..models import WaveResNet

ValidationDataLike = Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]


def _serialize_hisso_cfg(cfg: Optional[HISSOTrainerConfig]) -> Optional[Dict[str, Any]]:
    if cfg is None:
        return None
    return {
        "episode_length": int(cfg.episode_length),
        "episodes_per_batch": int(cfg.episodes_per_batch),
        "episode_batch_size": (
            int(cfg.episode_batch_size) if cfg.episode_batch_size is not None else None
        ),
        "updates_per_epoch": (
            int(cfg.updates_per_epoch) if cfg.updates_per_epoch is not None else None
        ),
        "primary_dim": int(cfg.primary_dim),
        "primary_transform": cfg.primary_transform,
        "random_state": cfg.random_state,
        "transition_penalty": float(cfg.transition_penalty),
    }


def _deserialize_hisso_cfg(data: Any) -> Optional[HISSOTrainerConfig]:
    if data is None:
        return None
    if isinstance(data, HISSOTrainerConfig):
        return data
    if isinstance(data, Mapping):
        return ensure_hisso_trainer_config(data)
    raise TypeError(f"Unable to deserialize HISSO trainer config from type {type(data)!r}")


def _serialize_hisso_options(options: Optional[HISSOOptions]) -> Optional[Dict[str, Any]]:
    if options is None:
        return None
    return {
        "episode_length": int(options.episode_length),
        "batch_episodes": int(options.batch_episodes),
        "updates_per_epoch": (
            int(options.updates_per_epoch) if options.updates_per_epoch is not None else None
        ),
        "transition_penalty": float(options.transition_penalty),
        "primary_transform": options.primary_transform,
        "reward_fn": options.reward_fn,
        "context_extractor": options.context_extractor,
        "input_noise_std": options.input_noise_std,
        "supervised": options.supervised,
    }


def _deserialize_hisso_options(data: Any) -> Optional[HISSOOptions]:
    if data is None:
        return None
    if isinstance(data, HISSOOptions):
        return data
    if isinstance(data, Mapping):
        updates_per_epoch_raw = data.get("updates_per_epoch")
        return HISSOOptions(
            episode_length=int(data.get("episode_length", 64)),
            batch_episodes=int(data.get("batch_episodes", data.get("episodes_per_batch", 32))),
            updates_per_epoch=(
                int(updates_per_epoch_raw) if updates_per_epoch_raw is not None else None
            ),
            transition_penalty=float(data.get("transition_penalty", 0.0)),
            primary_transform=str(data.get("primary_transform", "identity")),
            reward_fn=data.get("reward_fn"),
            context_extractor=data.get("context_extractor"),
            input_noise_std=data.get("input_noise_std"),
            supervised=data.get("supervised"),
        )
    raise TypeError(f"Unable to deserialize HISSO options from type {type(data)!r}")


class _AttentionDenseModel(nn.Module):
    """Wrap token-level backbone + attention pooling for flattened inputs."""

    def __init__(
        self,
        token_backbone: nn.Module,
        attention_module: Optional[nn.Module],
        *,
        seq_len: int,
        token_dim: int,
        embed_dim: int,
        output_dim: int,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        self.token_backbone = token_backbone
        self.attention = attention_module
        self.seq_len = int(seq_len)
        self.token_dim = int(token_dim)
        self.embed_dim = int(embed_dim)
        self.readout = nn.Linear(self.embed_dim, output_dim)
        pool = str(pool).lower()
        if pool not in {"mean", "last"}:
            raise ValueError("pool must be 'mean' or 'last'.")
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Attention-enabled models expect 2D inputs (batch, features); received rank {x.ndim}."
            )
        if x.shape[1] != self.seq_len * self.token_dim:
            raise ValueError(
                "Input feature dimension does not match inferred attention shape "
                f"(features={x.shape[1]}, expected={self.seq_len * self.token_dim})."
            )
        batch = x.shape[0]
        tokens = x.view(batch, self.seq_len, self.token_dim)
        embeds = self.token_backbone(tokens.reshape(batch * self.seq_len, self.token_dim))
        embeds = embeds.view(batch, self.seq_len, self.embed_dim)
        ctx = embeds
        if self.attention is not None:
            ctx, _ = self.attention(embeds, embeds, embeds)
        if self.pool == "last":
            pooled = ctx[:, -1, :]
        else:
            pooled = ctx.mean(dim=1)
        return self.readout(pooled)


class _AttentionConvModel(nn.Module):
    """Wrap convolutional backbones with sequence attention."""

    def __init__(
        self,
        conv_core: nn.Module,
        attention_module: nn.Module,
        *,
        spatial_shape: Tuple[int, ...],
        segmentation_head: bool,
    ) -> None:
        super().__init__()
        if not hasattr(conv_core, "forward_tokens"):
            raise TypeError("attention requires conv cores exposing forward_tokens.")
        self.conv_core = conv_core
        self.attention = attention_module
        self.segmentation_head = bool(segmentation_head)
        self.spatial_shape = tuple(int(d) for d in spatial_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.conv_core.forward_tokens(x)
        if tokens.ndim < 3:
            raise ValueError("attention expects convolutional tokens with spatial dimensions.")
        batch = tokens.shape[0]
        channels = tokens.shape[1]
        spatial = tokens.shape[2:]
        seq = tokens.view(batch, channels, -1).transpose(1, 2)  # (B, T, C)
        ctx, _ = self.attention(seq, seq, seq)
        ctx = ctx.transpose(1, 2).reshape(batch, channels, *spatial)
        if self.segmentation_head:
            head = getattr(self.conv_core, "head", None)
            if head is None:
                raise RuntimeError("Convolutional core missing segmentation head.")
            return head(ctx)
        pool = getattr(self.conv_core, "pool", None)
        fc = getattr(self.conv_core, "fc", None)
        if pool is None or fc is None:
            raise RuntimeError("Convolutional core missing pool/fc required for attention.")
        pooled = pool(ctx)
        if pooled.ndim > 2:
            pooled = pooled.flatten(1)
        return fc(pooled)


class _WaveResNetSpectralDenseModel(nn.Module):
    """Apply spectral gating over an inferred sequence axis, then a WaveResNet readout."""

    def __init__(
        self,
        wave_core: WaveResNet,
        spectral_gate: SpectralGate1D,
        *,
        seq_len: int,
        token_dim: int,
    ) -> None:
        super().__init__()
        self.wave = wave_core
        self.spectral = spectral_gate
        self.seq_len = int(seq_len)
        self.token_dim = int(token_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                "WaveResNet spectral-gate models expect 2D inputs (batch, features); "
                f"received rank {x.ndim}."
            )
        expected = self.seq_len * self.token_dim
        if x.shape[1] != expected:
            raise ValueError(
                "Input feature dimension does not match inferred spectral gate shape "
                f"(features={x.shape[1]}, expected={expected})."
            )
        batch = x.shape[0]
        tokens = x.reshape(batch, self.seq_len, self.token_dim)
        if self.seq_len > 1:
            tokens = tokens + self.spectral(tokens)
        flat = tokens.reshape(batch, -1)
        if context is None:
            return self.wave(flat)
        return self.wave(flat, context)


class _WaveResNetConvModel(nn.Module):
    """Apply a convolutional stem, optional attention, then a WaveResNet readout."""

    def __init__(
        self,
        conv_core: nn.Module,
        wave_core: WaveResNet,
        *,
        spatial_shape: Tuple[int, ...],
        attention_module: Optional[nn.Module] = None,
        spectral_gate: Optional[SpectralGate1D] = None,
    ) -> None:
        super().__init__()
        if not hasattr(conv_core, "forward_tokens"):
            raise TypeError("WaveResNet convolutional mode requires conv_core.forward_tokens.")
        self.conv_core = conv_core
        self.wave = wave_core
        self.attention = attention_module
        self.spectral = spectral_gate
        self.spatial_shape = tuple(int(d) for d in spatial_shape)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_core.forward_tokens(x)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        if tokens.ndim < 3:
            raise ValueError("WaveResNetConvModel expects tensors with spatial dimensions.")
        batch = tokens.shape[0]
        channels = tokens.shape[1]
        seq = tokens.view(batch, channels, -1).transpose(1, 2)  # (B, T, C)
        if self.attention is not None:
            seq, _ = self.attention(seq, seq, seq)
        if self.spectral is not None and seq.shape[1] > 1:
            seq = seq + self.spectral(seq)
        flat = seq.reshape(batch, -1)
        if context is None:
            return self.wave(flat)
        return self.wave(flat, context)


__all__ = [
    "BaseEstimator",
    "RegressorMixin",
    "ValidationDataLike",
    "_AttentionDenseModel",
    "_AttentionConvModel",
    "_WaveResNetSpectralDenseModel",
    "_WaveResNetConvModel",
    "_deserialize_hisso_cfg",
    "_deserialize_hisso_options",
    "_serialize_hisso_cfg",
    "_serialize_hisso_options",
    "_sk_r2_score",
]
