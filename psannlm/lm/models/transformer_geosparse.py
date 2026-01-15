"""GeoSparse-based transformer variant for language modeling.

This keeps the attention path identical to other LM bases (for apples-to-apples
comparisons), while swapping the MLP sublayer for a GeoSparse residual MLP that
uses fixed geometric sparse connectivity over the intermediate features.

NOTE: GeoSparseLinear's naive gather implementation can materialize very large
intermediates when applied to LM tensors shaped (B, T, D). This module uses a
chunked implementation to keep peak memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
import re
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn

from ...activations import MixedActivation
from ...layers.geo_sparse import build_geo_connectivity, expand_in_indices_to_edges
from ...layers.sine_residual import RMSNorm
from ...nn import DropPath
from .sine import SineConfig, build_sine
from .transformer_respsann import SelfAttention, _sinusoidal_positions
from ..config import normalize_positional_encoding


def _auto_shape(n: int) -> Tuple[int, int]:
    """Pick a (H,W) factorization for n that is as square as possible."""
    if n <= 0:
        raise ValueError("n must be positive")
    root = int(math.isqrt(n))
    for h in range(root, 0, -1):
        if n % h == 0:
            return (h, n // h)
    return (1, n)


def _parse_shape(value: object | None) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, str):
        m = re.match(r"^\s*(\d+)\s*[x,]\s*(\d+)\s*$", value)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f"Invalid geosparse_shape: {value!r} (expected (H,W) or 'HxW')")


class ChunkedGeoSparseLinear(nn.Module):
    """GeoSparse linear with bounded peak memory via output-feature chunking."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_index_per_out: torch.Tensor,
        *,
        bias: bool = True,
        compute_mode: str = "gather",
        chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        if in_index_per_out.ndim != 2:
            raise ValueError("in_index_per_out must have shape (out_features, k).")
        if int(in_index_per_out.shape[0]) != int(out_features):
            raise ValueError("in_index_per_out first dimension must match out_features.")
        min_idx = int(in_index_per_out.min().item())
        max_idx = int(in_index_per_out.max().item())
        if min_idx < 0 or max_idx >= int(in_features):
            raise ValueError("in_index_per_out contains indices outside [0, in_features).")

        mode = str(compute_mode or "gather").lower()
        if mode not in {"gather", "scatter"}:
            raise ValueError("compute_mode must be 'gather' or 'scatter'.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(in_index_per_out.shape[1])
        self.compute_mode = mode
        self.chunk_size = max(0, int(chunk_size))

        self.register_buffer(
            "in_index_per_out", in_index_per_out.to(dtype=torch.long).contiguous()
        )
        self.weight = nn.Parameter(torch.empty(self.out_features, self.k))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

        if self.compute_mode == "scatter":
            src, dst = expand_in_indices_to_edges(self.in_index_per_out)
            self.register_buffer("src_index", src.contiguous())
            self.register_buffer("dst_index", dst.contiguous())
        else:
            self.register_buffer("src_index", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("dst_index", torch.empty(0, dtype=torch.long), persistent=False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.k) if self.k > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError("x must have shape (..., in_features).")
        if int(x.shape[-1]) != self.in_features:
            raise ValueError("x last dimension must match in_features.")

        batch_shape = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features)

        if self.compute_mode == "gather":
            out = self._forward_gather(x2d)
        else:
            out = self._forward_scatter(x2d)

        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*batch_shape, self.out_features)

    def _forward_gather(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.chunk_size <= 0 or self.chunk_size >= self.out_features:
            gathered = x2d[:, self.in_index_per_out]  # (N, out, k)
            return (gathered * self.weight.unsqueeze(0)).sum(dim=-1)

        out = torch.empty(
            x2d.shape[0], self.out_features, device=x2d.device, dtype=x2d.dtype
        )
        for start in range(0, self.out_features, self.chunk_size):
            end = min(self.out_features, start + self.chunk_size)
            idx = self.in_index_per_out[start:end]
            w = self.weight[start:end]  # (chunk,k)
            gathered = x2d[:, idx]  # (N, chunk, k)
            out[:, start:end] = (gathered * w.unsqueeze(0)).sum(dim=-1)
        return out

    def _forward_scatter(self, x2d: torch.Tensor) -> torch.Tensor:
        # Process edges in chunks to bound peak memory.
        n = x2d.shape[0]
        out = torch.zeros(n, self.out_features, device=x2d.device, dtype=x2d.dtype)
        weight_flat = self.weight.reshape(-1)
        edges = int(self.src_index.numel())
        if edges == 0:
            return out
        edge_chunk = edges if self.chunk_size <= 0 else max(1, self.chunk_size * self.k)
        for start in range(0, edges, edge_chunk):
            end = min(edges, start + edge_chunk)
            src = self.src_index[start:end]
            dst = self.dst_index[start:end]
            w = weight_flat[start:end]
            contrib = x2d.index_select(1, src) * w.unsqueeze(0)
            out.index_add_(1, dst, contrib)
        return out


class GeoSparseResidualBlockLM(nn.Module):
    """Pre-norm residual block using chunked GeoSparse linear layers."""

    def __init__(
        self,
        features: int,
        in_index_per_out: torch.Tensor,
        *,
        activation: nn.Module,
        norm: str = "rms",
        drop_path: float = 0.0,
        residual_alpha_init: float = 1.0,
        bias: bool = True,
        compute_mode: str = "gather",
        chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if features <= 0:
            raise ValueError("features must be positive.")
        if int(in_index_per_out.shape[0]) != int(features):
            raise ValueError("in_index_per_out first dimension must match features.")
        self.features = int(features)
        key = str(norm or "rms").lower()
        if key == "none":
            self.norm = nn.Identity()
        elif key == "layer":
            self.norm = nn.LayerNorm(self.features)
        elif key == "rms":
            self.norm = RMSNorm(self.features)
        else:
            raise ValueError("norm must be one of: 'none', 'layer', 'rms'")

        self.fc1 = ChunkedGeoSparseLinear(
            self.features,
            self.features,
            in_index_per_out,
            bias=bias,
            compute_mode=compute_mode,
            chunk_size=chunk_size,
        )
        self.act = activation
        self.fc2 = ChunkedGeoSparseLinear(
            self.features,
            self.features,
            in_index_per_out,
            bias=bias,
            compute_mode=compute_mode,
            chunk_size=chunk_size,
        )
        self.drop_path = DropPath(float(drop_path)) if float(drop_path) > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.full((1,), float(residual_alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.drop_path(h)
        return x + self.alpha * h


class GeoSparseMLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        *,
        shape: Optional[Tuple[int, int]] = None,
        depth: int = 1,
        k: int = 8,
        pattern: str = "local",
        radius: int = 1,
        offsets: Optional[Sequence[Tuple[int, int]]] = None,
        wrap_mode: str = "clamp",
        activation_type: str = "psann",
        activation_types: Optional[Sequence[str]] = None,
        activation_ratios: Optional[Sequence[float]] = None,
        activation_ratio_sum_tol: float = 1e-3,
        activation_layout: str = "random",
        sine: Optional[SineConfig] = None,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 1.0,
        bias: bool = True,
        compute_mode: str = "gather",
        seed: Optional[int] = None,
        chunk_size: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("geosparse_depth must be positive")
        if k <= 0:
            raise ValueError("geosparse_k must be positive")
        self.fc_in = nn.Linear(int(d_model), int(d_mlp))
        self.fc_out = nn.Linear(int(d_mlp), int(d_model))
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        shp = _auto_shape(int(d_mlp)) if shape is None else (int(shape[0]), int(shape[1]))
        if shp[0] * shp[1] != int(d_mlp):
            raise ValueError(
                f"geosparse_shape {shp} incompatible with d_mlp={int(d_mlp)} (H*W must match)"
            )
        self.shape = shp

        act_key = str(activation_type or "psann").lower()
        if act_key not in {"psann", "sine", "gelu", "relu", "tanh", "mixed"}:
            raise ValueError(
                "geosparse activation_type must be one of: 'psann'/'sine', 'gelu', 'relu', 'tanh', 'mixed'"
            )
        if act_key == "mixed":
            if activation_types is None or len(list(activation_types)) == 0:
                raise ValueError("activation_types must be provided when activation_type='mixed'")

        blocks: list[nn.Module] = []
        for idx in range(int(depth)):
            block_seed = None if seed is None else int(seed) + idx * 9973
            indices = build_geo_connectivity(
                self.shape,
                k=int(k),
                pattern=str(pattern),
                radius=int(radius),
                offsets=offsets,
                wrap_mode=str(wrap_mode),
                seed=block_seed,
            )
            dp = float(drop_path_max) * (idx / max(1, int(depth) - 1)) if int(depth) > 1 else 0.0

            if act_key == "mixed":
                builders = {
                    # Override psann builder so we keep LM SineConfig semantics.
                    "psann": (lambda n, _s=sine: build_sine(int(n), _s)),
                }
                activation = MixedActivation(
                    int(d_mlp),
                    activation_types=list(activation_types or []),
                    activation_ratios=list(activation_ratios) if activation_ratios is not None else None,
                    ratio_sum_tol=float(activation_ratio_sum_tol),
                    seed=block_seed,
                    layout=str(activation_layout),
                    builders=builders,
                )
            elif act_key in {"psann", "sine"}:
                activation = build_sine(int(d_mlp), sine)
            elif act_key == "gelu":
                activation = nn.GELU()
            elif act_key == "relu":
                activation = nn.ReLU()
            else:
                activation = nn.Tanh()
            blocks.append(
                GeoSparseResidualBlockLM(
                    int(d_mlp),
                    indices,
                    activation=activation,
                    norm=str(norm),
                    drop_path=dp,
                    residual_alpha_init=float(residual_alpha_init),
                    bias=bool(bias),
                    compute_mode=str(compute_mode),
                    chunk_size=int(chunk_size),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.fc_out(h)
        return self.drop(h)


class GeoSparseBlock(nn.Module):
    def __init__(self, cfg: "GeoSparseTransformerConfig") -> None:
        super().__init__()
        self.attn = SelfAttention(
            cfg.d_model,
            cfg.n_heads,
            dropout=cfg.dropout,
            positional_encoding=cfg.positional_encoding,
            attn_impl=cfg.attn_impl,
        )
        self.mlp = GeoSparseMLP(
            cfg.d_model,
            cfg.d_mlp,
            shape=cfg.geosparse_shape,
            depth=cfg.geosparse_depth,
            k=cfg.geosparse_k,
            pattern=cfg.geosparse_pattern,
            radius=cfg.geosparse_radius,
            offsets=cfg.geosparse_offsets,
            wrap_mode=cfg.geosparse_wrap_mode,
            activation_type=cfg.geosparse_activation,
            activation_types=cfg.geosparse_activation_types,
            activation_ratios=cfg.geosparse_activation_ratios,
            activation_ratio_sum_tol=cfg.geosparse_activation_ratio_sum_tol,
            activation_layout=cfg.geosparse_activation_layout,
            sine=cfg.sine,
            norm=cfg.geosparse_norm,
            drop_path_max=cfg.geosparse_drop_path_max,
            residual_alpha_init=cfg.geosparse_residual_alpha_init,
            bias=cfg.geosparse_bias,
            compute_mode=cfg.geosparse_compute_mode,
            seed=cfg.geosparse_seed,
            chunk_size=cfg.geosparse_chunk_size,
            dropout=cfg.dropout,
        )
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.alpha = nn.Parameter(torch.ones(1))

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
        x = x + self.alpha * self.mlp(self.norm2(x))
        if use_cache or past_kv is not None:
            assert present is not None
            return x, present
        return x


@dataclass
class GeoSparseTransformerConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_mlp: int = 2048
    dropout: float = 0.0
    positional_encoding: str = "rope"
    rope: Optional[bool] = None
    mlp_activation: str = "sine"  # kept for shared harness compatibility
    sine: Optional[SineConfig] = None
    attn_impl: str = "math"  # "math" | "sdpa" | "auto"

    # GeoSparse MLP hyperparameters
    geosparse_shape: object | None = None  # (H,W) or "HxW"
    geosparse_depth: int = 1
    geosparse_k: int = 8
    geosparse_pattern: str = "local"
    geosparse_radius: int = 1
    geosparse_offsets: Optional[List[Tuple[int, int]]] = None
    geosparse_wrap_mode: str = "clamp"
    geosparse_activation: str = "psann"  # 'psann' | 'gelu' | 'relu' | 'tanh'
    geosparse_activation_types: Optional[List[str]] = None
    geosparse_activation_ratios: Optional[List[float]] = None
    geosparse_activation_ratio_sum_tol: float = 1e-3
    geosparse_activation_layout: str = "random"
    geosparse_norm: str = "rms"
    geosparse_drop_path_max: float = 0.0
    geosparse_residual_alpha_init: float = 1.0
    geosparse_bias: bool = True
    geosparse_compute_mode: str = "gather"  # 'gather' | 'scatter'
    geosparse_seed: Optional[int] = None
    geosparse_chunk_size: int = 32

    def __post_init__(self) -> None:
        if self.rope is not None:
            self.positional_encoding = "rope" if self.rope else "sinusoidal"
        self.positional_encoding = normalize_positional_encoding(self.positional_encoding)
        self.rope = self.positional_encoding == "rope"
        self.geosparse_shape = _parse_shape(self.geosparse_shape)
        if self.geosparse_chunk_size < 0:
            raise ValueError("geosparse_chunk_size must be >= 0")
        if float(self.geosparse_activation_ratio_sum_tol) < 0:
            raise ValueError("geosparse_activation_ratio_sum_tol must be >= 0")


class GeoSparseTransformer(nn.Module):
    def __init__(self, cfg: GeoSparseTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing: bool = False
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([GeoSparseBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
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


def build_geosparse_transformer(**kwargs) -> GeoSparseTransformer:
    # Filter to supported fields to tolerate shared harness kwargs.
    allowed = {f.name for f in fields(GeoSparseTransformerConfig)}
    filtered = {k: v for k, v in dict(kwargs).items() if k in allowed}
    return GeoSparseTransformer(GeoSparseTransformerConfig(**filtered))
