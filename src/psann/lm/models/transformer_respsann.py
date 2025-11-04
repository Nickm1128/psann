"""ResPSANN-based transformer for language modeling.

Implements a standard transformer stack with:
- Token embeddings
- Rotary (RoPE) or absolute sinusoidal positional encoding
- Multi-head self-attention (causal) with optional RoPE
- MLP using parametric sine activation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import nn
import math

from ...activations import SineParam
from ...layers.sine_residual import RMSNorm
from .sine import SineConfig, build_sine


def _sinusoidal_positions(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(seq_len, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (T, D)


class PSANNMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        *,
        sine: Optional[SineConfig] = None,
        mlp_activation: str = "sine",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_mlp)
        mlp_activation = (mlp_activation or "sine").lower()
        if mlp_activation == "sine":
            self.act: nn.Module = build_sine(d_mlp, sine)
        elif mlp_activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported mlp_activation: {mlp_activation}")
        self.fc2 = nn.Linear(d_mlp, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        return self.drop(h)


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Create RoPE cosine and sine caches of shape (1, 1, T, D)."""
    assert head_dim % 2 == 0, "RoPE head_dim must be even"
    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
    # positions
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", t, inv_freq)  # (T, half_dim)
    # interleave for complex rotation
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (T, D)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)  # (T, D)
    cos = cos.to(dtype=dtype).unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x: (..., D) with even D; split into pairs and rotate
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)  # (..., D)


def _build_rope_cache_offset(offset: int, seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE cache for a window starting at position `offset`.

    Returns (cos, sin) with shape (1,1,seq_len,head_dim).
    """
    assert head_dim % 2 == 0
    half_dim = head_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
    t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("t,f->tf", t, inv_freq)
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    cos = cos.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, *, dropout: float = 0.0, rope: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0 and rope:
            # enforce even for RoPE
            raise ValueError("head_dim must be even when using RoPE")
        self.rope = bool(rope)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Self-attention with optional KV-cache.

        Args:
            x: (B, T, D)
            past_kv: optional tuple (k, v), each (B, H, Tp, hd)
            use_cache: if True, returns (out, present_kv)
        """
        B, T, D = x.shape
        device = x.device
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        past_len = int(past_kv[0].size(-2)) if past_kv is not None else 0
        if self.rope:
            # Apply RoPE with positional offset equal to past length
            cos, sin = _build_rope_cache_offset(past_len, T, self.head_dim, device, q.dtype)
            q = (q * cos) + (_rope_rotate_half(q) * sin)
            k = (k * cos) + (_rope_rotate_half(k) * sin)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # causal mask for (T, K): take last T rows from a (K,K) upper triangular
        Klen = k.size(-2)
        base = torch.full((Klen, Klen), float("-inf"), device=device, dtype=att.dtype)
        base = torch.triu(base, diagonal=1)
        mask = base[-T:, :]
        att = att + mask
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        out = torch.matmul(att, v)  # (B,H,T,hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.o_proj(out)
        out = self.proj_drop(out)
        if use_cache or past_kv is not None:
            return out, (k, v)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_mlp: int,
        *,
        dropout: float = 0.0,
        norm: str = "rms",
        sine: Optional[SineConfig] = None,
        mlp_activation: str = "sine",
        rope: bool = True,
    ) -> None:
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads, dropout=dropout, rope=rope)
        self.mlp = PSANNMLP(d_model, d_mlp, sine=sine, mlp_activation=mlp_activation, dropout=dropout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Optional residual scaling (learnable), default 1.0 to preserve behavior
        self.alpha = nn.Parameter(torch.tensor(1.0))
        if norm == "rms":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
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
        h2 = self.norm2(x)
        x = x + self.alpha * self.mlp(h2)
        if use_cache or past_kv is not None:
            assert present is not None
            return x, present
        return x


@dataclass
class ResPSANNTransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_mlp: int = 2048
    dropout: float = 0.0
    rope: bool = True
    mlp_activation: str = "sine"  # "sine" | "gelu"
    sine: Optional[SineConfig] = None


class ResPSANNTransformer(nn.Module):
    def __init__(self, cfg: ResPSANNTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        sinecfg = cfg.sine if cfg.sine is not None else SineConfig()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cfg.d_model,
                    cfg.n_heads,
                    cfg.d_mlp,
                    dropout=cfg.dropout,
                    norm="rms",
                    sine=sinecfg,
                    mlp_activation=cfg.mlp_activation,
                    rope=cfg.rope,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight initialization
        def _init_weights(m: nn.Module) -> None:  # pragma: no cover - standard init
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        use_cache: bool = False,
        past_kvs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # input_ids: (B, T)
        B, T = input_ids.shape
        x = self.embed(input_ids)  # (B, T, D)
        if not self.cfg.rope:
            past_len = int(past_kvs[0][0].size(-2)) if (use_cache and past_kvs) else 0
            pe = _sinusoidal_positions(past_len + T, self.cfg.d_model, x.device).unsqueeze(0)
            x = x + pe[:, past_len: past_len + T, :]
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
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            return logits  # (B, T, V)


def build_respsann_transformer(**kwargs) -> ResPSANNTransformer:
    return ResPSANNTransformer(ResPSANNTransformerConfig(**kwargs))
