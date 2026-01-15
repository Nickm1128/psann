"""Generation utilities for PSANN-LM.

Provides basic top-k/top-p (nucleus) sampling with temperature.
"""

from __future__ import annotations

from typing import Optional

import torch


def filter_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None or top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    cutoff = v[..., -1, None]
    return torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)


def filter_top_p(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p <= 0 or top_p >= 1:
        return logits
    # sort
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > top_p
    # ensure at least one token retained
    mask[..., 0] = False
    sorted_logits = torch.where(mask, torch.full_like(sorted_logits, float("-inf")), sorted_logits)
    # unsort
    unsorted = torch.full_like(sorted_logits, float("-inf"))
    unsorted.scatter_(-1, sorted_idx, sorted_logits)
    return unsorted


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Sample a token id from logits (unnormalized) of shape (B, V)."""
    if temperature <= 0:
        # greedy
        return torch.argmax(logits, dim=-1)
    logits = logits / max(1e-5, float(temperature))
    logits = filter_top_k(logits, top_k)
    logits = filter_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample(
    logits: torch.Tensor,
    *,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convenience wrapper to sample token ids from logits.

    Accepts logits of shape (B, V) or (B, T, V). When (B, T, V), the
    last timestep is used for sampling. Returns a tensor of shape (B,)
    with integer token ids.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if logits.dim() != 2:
        raise ValueError("logits must be 2D (B,V) or 3D (B,T,V)")
    return sample_next_token(
        logits,
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
    )
