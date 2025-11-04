"""WaveResNet-based transformer for language modeling.

For now, this mirrors the ResPSANN transformer structure with optional
RoPE in attention. Future work may integrate WaveResNet residual
pathways interleaved with attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from torch import nn

from .transformer_respsann import (
    RMSNorm,
    SineConfig,
    TransformerBlock,
    _sinusoidal_positions,
    SelfAttention,
)
from .sine import build_sine


@dataclass
class WaveResNetTransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_mlp: int = 2048
    dropout: float = 0.0
    rope: bool = True
    mlp_activation: str = "sine"  # "sine" | "gelu"
    sine: Optional[SineConfig] = None
    # WaveResNet-temporal options (scaffolding)
    wave_interleave: bool = False  # if True, add a temporal conv residual per block
    wave_replace: bool = False     # if True, replace MLP with temporal conv residual
    wave_kernel_size: int = 3
    wave_dilation_growth: int = 1  # 1 = no dilation growth across layers
    wave_dropout: float = 0.0


class WaveResNetTransformer(nn.Module):
    def __init__(self, cfg: WaveResNetTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        sinecfg = cfg.sine if cfg.sine is not None else SineConfig()
        # Build blocks; optionally interleave a temporal wave block per layer
        blocks: List[nn.Module] = []
        for i in range(cfg.n_layers):
            dilation = 1
            if int(cfg.wave_dilation_growth) > 1:
                try:
                    dilation = int(max(1, (cfg.wave_dilation_growth ** i)))
                except Exception:
                    dilation = 1
            blk = TransformerBlockWRN(
                cfg.d_model,
                cfg.n_heads,
                cfg.d_mlp,
                dropout=cfg.dropout,
                norm="rms",
                sine=sinecfg,
                mlp_activation=cfg.mlp_activation,
                rope=cfg.rope,
                wave_interleave=bool(cfg.wave_interleave),
                wave_replace=bool(cfg.wave_replace),
                wave_kernel_size=int(cfg.wave_kernel_size),
                wave_dilation=int(dilation),
                wave_dropout=float(cfg.wave_dropout),
            )
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
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
        B, T = input_ids.shape
        x = self.embed(input_ids)
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
            return self.lm_head(x)


def build_waveresnet_transformer(**kwargs) -> WaveResNetTransformer:
    return WaveResNetTransformer(WaveResNetTransformerConfig(**kwargs))


class WaveBlock1D(nn.Module):
    """Temporal residual block using depthwise Conv1d over sequence.

    This block operates on (B, T, D) and applies:
      - RMSNorm
      - Depthwise Conv1d over T with padding to preserve length
      - Sine (default) or GELU activation
      - Linear projection and dropout
      - Residual add
    """

    def __init__(
        self,
        d_model: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        activation: str = "sine",
        sine: Optional[SineConfig] = None,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        k = max(1, int(kernel_size))
        dil = max(1, int(dilation))
        self._k = k
        self._dil = dil
        # same-length padding for odd kernels
        pad = (k - 1) // 2 * dil
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=k, groups=d_model, dilation=dil, padding=pad, bias=False)
        activation = (activation or "sine").lower()
        if activation == "sine":
            self.act: nn.Module = build_sine(d_model, sine)
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        # (B,T,D) -> (B,D,T)
        ht = h.transpose(1, 2)
        ht = self.dw(ht)
        # back to (B,T,D)
        h = ht.transpose(1, 2)
        h = self.act(h)
        h = self.proj(h)
        h = self.drop(h)
        return x + h

    def forward_with_cache(
        self,
        x: torch.Tensor,
        *,
        cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal-ish cached forward for step/batch generation.

        Maintains a simple history of the normalized inputs sized to the
        effective receptive field (kernel_size-1)*dilation. This enables
        consistent outputs for step-wise generation without relying on
        future context.

        Args:
            x: (B, T, D)
            cache: optional (B, L, D) of prior normalized inputs, where
                   L = max(0, (k-1)*dil)

        Returns:
            y: (B, T, D) transformed outputs with residual add
            next_cache: (B, L, D) for the next call
        """
        B, T, D = x.shape
        # Build concat over time axis for convolution
        if cache is not None and cache.numel() > 0:
            cat_in = torch.cat([cache, x], dim=1)
        else:
            cat_in = x
        # Normalize across full cat sequence
        h = self.norm(cat_in)
        ht = h.transpose(1, 2)
        ht = self.dw(ht)
        hout = ht.transpose(1, 2)
        hout = self.act(hout)
        hout = self.proj(hout)
        hout = self.drop(hout)
        # Take only the last T positions corresponding to current x
        y_last = hout[:, -T:, :]
        y = x + y_last
        # Compute next cache length = (k-1)*dil (number of prior steps needed)
        l_keep = max(0, (self._k - 1) * self._dil)
        next_cache = h[:, -l_keep:, :] if l_keep > 0 else h[:, :0, :]
        return y, next_cache


class TransformerBlockWRN(nn.Module):
    """Transformer block with optional interleaved temporal wave block.

    Matches the attention/KV-cache interface of the base block so the
    WaveResNetTransformer can support cached generation.
    """

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
        wave_interleave: bool = False,
        wave_replace: bool = False,
        wave_kernel_size: int = 3,
        wave_dilation: int = 1,
        wave_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads, dropout=dropout, rope=rope)
        # reuse PSANN MLP as in base
        from .transformer_respsann import PSANNMLP  # local import to avoid cycle in type checking

        self.wave_replace = bool(wave_replace)
        self.mlp: Optional[nn.Module]
        if not self.wave_replace:
            self.mlp = PSANNMLP(d_model, d_mlp, sine=sine, mlp_activation=mlp_activation, dropout=dropout)
        else:
            self.mlp = None
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Optional residual scaling (learnable), default 1.0
        self.alpha = nn.Parameter(torch.tensor(1.0))
        if norm == "rms":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        self.wave: Optional[WaveBlock1D] = None
        if bool(wave_interleave):
            self.wave = WaveBlock1D(
                d_model,
                kernel_size=wave_kernel_size,
                dilation=wave_dilation,
                dropout=wave_dropout,
                activation=mlp_activation,
                sine=sine,
            )
        # Internal wave cache for cached generation (per block)
        self._wave_cache: Optional[torch.Tensor] = None

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
        if self.wave_replace:
            # Replace MLP path with wave residual
            if self.wave is not None:
                if use_cache:
                    # Reset cache at sequence start indicated by no past_kv
                    if past_kv is None:
                        self._wave_cache = None
                    x, self._wave_cache = self.wave.forward_with_cache(h2, cache=self._wave_cache)
                else:
                    self._wave_cache = None
                    x = self.wave(h2)
        else:
            x = x + self.alpha * (self.mlp(h2) if self.mlp is not None else 0)
            if self.wave is not None:
                if use_cache:
                    if past_kv is None:
                        self._wave_cache = None
                    x, self._wave_cache = self.wave.forward_with_cache(x, cache=self._wave_cache)
                else:
                    self._wave_cache = None
                    x = self.wave(x)
        if use_cache or past_kv is not None:
            assert present is not None
            return x, present
        return x
