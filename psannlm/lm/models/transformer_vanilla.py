"""Vanilla transformer baseline for language modeling.

This is a "regular transformer" baseline to compare against the PSANN-based
variants in this repo. It reuses the same attention implementation (including
RoPE/ALiBi support) for apples-to-apples throughput and perplexity comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from .transformer_respsann import SelfAttention, _sinusoidal_positions
from ..config import normalize_positional_encoding


@dataclass
class VanillaTransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_mlp: int = 2048
    mlp_activation: str = "gelu"  # "gelu" | "relu"
    dropout: float = 0.0
    positional_encoding: str = "rope"
    rope: Optional[bool] = None
    attn_impl: str = "math"  # "math" | "sdpa" | "auto"

    def __post_init__(self) -> None:
        if self.rope is not None:
            self.positional_encoding = "rope" if self.rope else "sinusoidal"
        self.positional_encoding = normalize_positional_encoding(self.positional_encoding)
        self.rope = self.positional_encoding == "rope"
        act = (self.mlp_activation or "gelu").lower()
        if act in {"gelu", "relu"}:
            self.mlp_activation = act
        elif act in {"sine", "psann", "phase_psann"}:
            self.mlp_activation = "gelu"
        else:
            raise ValueError("mlp_activation must be one of: 'gelu', 'relu'")


def _build_mlp_activation(name: str) -> nn.Module:
    key = (name or "gelu").lower()
    if key == "relu":
        return nn.ReLU()
    return nn.GELU()


class VanillaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        *,
        dropout: float,
        mlp_activation: str,
        positional_encoding: str,
        attn_impl: str,
    ) -> None:
        super().__init__()
        self.attn = SelfAttention(
            d_model,
            n_heads,
            dropout=dropout,
            positional_encoding=positional_encoding,
            attn_impl=attn_impl,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            _build_mlp_activation(mlp_activation),
            nn.Linear(d_mlp, d_model),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.norm1(x)
        attn_out = self.attn(h, past_kv=past_kv, use_cache=use_cache)
        if isinstance(attn_out, tuple):
            attn_y, present = attn_out
        else:
            attn_y, present = attn_out, None
        x = x + attn_y
        x = x + self.mlp(self.norm2(x))
        if use_cache or past_kv is not None:
            assert present is not None
            return x, present
        return x


class VanillaTransformer(nn.Module):
    def __init__(self, cfg: VanillaTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing: bool = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                VanillaBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_mlp,
                    dropout=cfg.dropout,
                    mlp_activation=cfg.mlp_activation,
                    positional_encoding=cfg.positional_encoding,
                    attn_impl=cfg.attn_impl,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        def _init_weights(m: nn.Module) -> None:  # pragma: no cover
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        use_cache: bool = False,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T = input_ids.shape
        x = self.embed(input_ids)
        if self.cfg.positional_encoding == "sinusoidal":
            past_len = int(past_kvs[0][0].size(-2)) if (use_cache and past_kvs) else 0
            pe = _sinusoidal_positions(past_len + T, self.cfg.d_model, x.device).unsqueeze(0)
            x = x + pe[:, past_len : past_len + T, :]

        if use_cache:
            presents: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for i, blk in enumerate(self.blocks):
                pkv = past_kvs[i] if past_kvs is not None else None
                out = blk(x, past_kv=pkv, use_cache=True)
                x, present = out  # type: ignore[misc]
                presents.append(present)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits, presents

        if self.gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint as _cp

            for blk in self.blocks:
                x = _cp(blk, x, use_reentrant=False)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def build_vanilla_transformer(**kwargs) -> VanillaTransformer:
    # Tolerate PSANN-specific kwargs when used in shared harnesses.
    kwargs = dict(kwargs)
    for k in (
        "sine",
        "use_spectral_gate",
        "k_fft",
        "gate_type",
        "gate_groups",
        "gate_init",
        "gate_strength",
        "wave_interleave",
        "wave_replace",
        "wave_kernel_size",
        "wave_dilation_growth",
        "wave_dropout",
    ):
        kwargs.pop(k, None)
    return VanillaTransformer(VanillaTransformerConfig(**kwargs))
