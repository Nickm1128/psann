from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

Offset = Tuple[int, int]


def build_geo_connectivity(
    shape: Tuple[int, int],
    *,
    k: int,
    pattern: str = "local",
    radius: int = 1,
    offsets: Optional[Sequence[Offset]] = None,
    wrap_mode: str = "clamp",
    seed: Optional[int] = None,
    include_self: bool = True,
    allow_repeats: bool = True,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build fixed connectivity for a 2D geometric layer.

    Returns a tensor of shape (n_out, k) with input indices per output neuron.
    """
    height, width = _validate_shape(shape)
    if k <= 0:
        raise ValueError("k must be positive.")
    wrap_mode = wrap_mode.lower()
    if wrap_mode not in {"clamp", "wrap"}:
        raise ValueError("wrap_mode must be 'clamp' or 'wrap'.")
    pattern = pattern.lower()
    if pattern not in {"local", "random", "hash"}:
        raise ValueError("pattern must be 'local', 'random', or 'hash'.")

    offsets_list = _normalize_offsets(offsets, radius=radius, include_self=include_self)
    n_out = height * width

    indices = torch.empty((n_out, k), dtype=torch.long)
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(int(seed))

    for out_idx in range(n_out):
        row = out_idx // width
        col = out_idx % width
        candidates = _collect_candidates(
            row,
            col,
            height=height,
            width=width,
            offsets=offsets_list,
            wrap_mode=wrap_mode,
        )
        if not allow_repeats:
            candidates = _unique_preserve_order(candidates)
        if pattern == "local":
            chosen = _tile_or_truncate(candidates, k, allow_repeats=allow_repeats)
        elif pattern == "random":
            chosen = _sample_candidates(candidates, k, rng if seed is not None else None, allow_repeats)
        else:
            base_seed = 0 if seed is None else int(seed)
            local_rng = torch.Generator()
            local_rng.manual_seed(base_seed + (out_idx + 1) * 1000003)
            chosen = _sample_candidates(candidates, k, local_rng, allow_repeats)
        indices[out_idx] = torch.tensor(chosen, dtype=torch.long)

    if device is not None:
        indices = indices.to(device=device)
    return indices


def expand_in_indices_to_edges(
    in_index_per_out: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert per-output indices into (src_index, dst_index) edge lists."""
    if in_index_per_out.ndim != 2:
        raise ValueError("in_index_per_out must be 2D (n_out, k).")
    n_out, k = in_index_per_out.shape
    src = in_index_per_out.reshape(-1).to(dtype=torch.long)
    dst = torch.arange(n_out, dtype=torch.long, device=in_index_per_out.device).repeat_interleave(
        k
    )
    return src, dst


class GeoSparseLinear(nn.Module):
    """Sparse linear layer with fixed fan-in connectivity."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        in_index_per_out: torch.Tensor,
        *,
        bias: bool = True,
        compute_mode: str = "gather",
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

        mode = compute_mode.lower()
        if mode not in {"gather", "scatter"}:
            raise ValueError("compute_mode must be 'gather' or 'scatter'.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.k = int(in_index_per_out.shape[1])
        self.compute_mode = mode

        self.register_buffer(
            "in_index_per_out", in_index_per_out.to(dtype=torch.long).contiguous()
        )
        self.weight = nn.Parameter(torch.empty(self.out_features, self.k))
        self.bias = nn.Parameter(torch.empty(self.out_features)) if bias else None
        self.reset_parameters()

        src, dst = expand_in_indices_to_edges(self.in_index_per_out)
        self.register_buffer("src_index", src.contiguous())
        self.register_buffer("dst_index", dst.contiguous())

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
            gathered = x2d[:, self.in_index_per_out]
            out = (gathered * self.weight.unsqueeze(0)).sum(dim=-1)
        else:
            edges = x2d[:, self.src_index] * self.weight.reshape(-1)
            out = torch.zeros(
                x2d.shape[0],
                self.out_features,
                device=x2d.device,
                dtype=x2d.dtype,
            )
            out.index_add_(1, self.dst_index, edges)

        if self.bias is not None:
            out = out + self.bias

        return out.reshape(*batch_shape, self.out_features)


def _validate_shape(shape: Tuple[int, int]) -> Tuple[int, int]:
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("shape must be a tuple of (height, width).")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive.")
    return height, width


def _normalize_offsets(
    offsets: Optional[Sequence[Offset]], *, radius: int, include_self: bool
) -> List[Offset]:
    if offsets is None:
        if radius < 0:
            raise ValueError("radius must be non-negative.")
        offsets_list: List[Offset] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not include_self and dr == 0 and dc == 0:
                    continue
                offsets_list.append((dr, dc))
    else:
        if len(offsets) == 0:
            raise ValueError("offsets must be non-empty when provided.")
        offsets_list = []
        for entry in offsets:
            if len(entry) != 2:
                raise ValueError("offsets entries must be (dr, dc).")
            dr, dc = int(entry[0]), int(entry[1])
            if not include_self and dr == 0 and dc == 0:
                continue
            offsets_list.append((dr, dc))
    if not offsets_list:
        raise ValueError("offsets list is empty after filtering.")
    return offsets_list


def _collect_candidates(
    row: int,
    col: int,
    *,
    height: int,
    width: int,
    offsets: Iterable[Offset],
    wrap_mode: str,
) -> List[int]:
    candidates: List[int] = []
    for dr, dc in offsets:
        r = row + dr
        c = col + dc
        if wrap_mode == "wrap":
            r %= height
            c %= width
        else:
            r = 0 if r < 0 else (height - 1 if r >= height else r)
            c = 0 if c < 0 else (width - 1 if c >= width else c)
        candidates.append(r * width + c)
    return candidates


def _unique_preserve_order(values: List[int]) -> List[int]:
    seen = set()
    uniq: List[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def _tile_or_truncate(values: List[int], k: int, *, allow_repeats: bool) -> List[int]:
    if len(values) >= k:
        return values[:k]
    if not allow_repeats:
        raise ValueError("Not enough unique candidates to satisfy k.")
    repeat = (k + len(values) - 1) // len(values)
    return (values * repeat)[:k]


def _sample_candidates(
    values: List[int],
    k: int,
    rng: Optional[torch.Generator],
    allow_repeats: bool,
) -> List[int]:
    if not values:
        raise ValueError("No candidates available to sample.")
    if allow_repeats:
        idx = torch.randint(0, len(values), (k,), generator=rng)
        return [values[i] for i in idx.tolist()]
    if len(values) < k:
        raise ValueError("Not enough unique candidates to satisfy k.")
    perm = torch.randperm(len(values), generator=rng)[:k]
    return [values[i] for i in perm.tolist()]
