from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

import torch.nn as nn


@dataclass
class AttentionConfig:
    """Configuration for optional attention modules attached to PSANN models."""

    kind: str = "none"
    num_heads: int = 4
    dropout: float = 0.0
    bias: bool = True
    batch_first: bool = True
    add_bias_kv: bool = False
    add_zero_attn: bool = False

    def is_enabled(self) -> bool:
        return str(self.kind or "none").lower() not in {"", "none", "off"}


def ensure_attention_config(value: AttentionConfig | Mapping[str, Any] | None) -> AttentionConfig:
    if value is None:
        return AttentionConfig()
    if isinstance(value, AttentionConfig):
        return value
    cfg = AttentionConfig()
    for k, v in dict(value).items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown AttentionConfig field '{k}'.")
        setattr(cfg, k, v)
    return cfg


AttentionFactory = Callable[[AttentionConfig, int], nn.Module]
_ATTENTION_REGISTRY: Dict[str, AttentionFactory] = {}


def register_attention(kind: str, factory: AttentionFactory) -> None:
    key = kind.strip().lower()
    if not key:
        raise ValueError("Attention kind cannot be empty.")
    _ATTENTION_REGISTRY[key] = factory


def build_attention_module(cfg: AttentionConfig, embed_dim: int) -> Optional[nn.Module]:
    kind = str(cfg.kind or "none").lower()
    if kind in {"", "none", "off"}:
        return None
    factory = _ATTENTION_REGISTRY.get(kind)
    if factory is None:
        available = ", ".join(sorted(_ATTENTION_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown attention kind '{cfg.kind}'. Available: {available}")
    return factory(cfg, int(embed_dim))


def _mha_factory(cfg: AttentionConfig, embed_dim: int) -> nn.Module:
    return nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=int(cfg.num_heads),
        dropout=float(cfg.dropout),
        bias=bool(cfg.bias),
        batch_first=bool(cfg.batch_first),
        add_bias_kv=bool(cfg.add_bias_kv),
        add_zero_attn=bool(cfg.add_zero_attn),
    )


register_attention("mha", _mha_factory)

__all__ = [
    "AttentionConfig",
    "AttentionFactory",
    "ensure_attention_config",
    "register_attention",
    "build_attention_module",
]
