from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SineTokenEmbedder(nn.Module):
    """Sinusoidal token embedder with optional learnable amplitude and phase.

    Embedding for token index i, dimension d:
        e[i, d] = A[d] * sin(omega[d] * (i + offset) + phi[d])

    - By default, frequencies follow a transformer-like schedule across dims.
    - If trainable=False, embeddings are fixed (no gradients).
    - Call `set_vocab_size(V)` before use to allocate/calc the table lazily.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        base: float = 10000.0,
        scale: float = 1.0,
        trainable: bool = False,
        device: Optional[torch.device | str] = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.base = float(base)
        self.scale = float(scale)
        self.trainable = bool(trainable)
        self.register_buffer("_table", torch.empty(0), persistent=False)
        # Learnable per-dim amplitude and phase; frequency derived from schedule
        self.A = nn.Parameter(torch.ones(self.embedding_dim), requires_grad=trainable)
        self.phi = nn.Parameter(torch.zeros(self.embedding_dim), requires_grad=trainable)
        self.offset = nn.Parameter(torch.zeros(1), requires_grad=trainable)
        # Cached vocab size
        self._vocab_size: int = 0
        if device is not None:
            self.to(device)

    def _frequencies(self, device: torch.device) -> torch.Tensor:
        # Transformer-like schedule
        d = torch.arange(self.embedding_dim, dtype=torch.float32, device=device)
        inv_freq = torch.exp(-torch.log(torch.tensor(self.base, dtype=torch.float32, device=device)) * (d // 2) / self.embedding_dim)
        # omega per dimension (alternate for sin/cos pairing)
        return 1.0 * inv_freq

    def set_vocab_size(self, vocab_size: int) -> None:
        self._vocab_size = int(vocab_size)
        # Lazily materialize the table for nearest-neighbor queries
        self._rebuild_table()

    def _rebuild_table(self) -> None:
        if self._vocab_size <= 0:
            self._table = torch.empty(0, device=self.A.device)
            return
        V = self._vocab_size
        idx = torch.arange(V, dtype=torch.float32, device=self.A.device).unsqueeze(1)  # (V,1)
        omega = self._frequencies(self.A.device).unsqueeze(0)  # (1,D)
        phase = self.phi.unsqueeze(0)
        val = torch.sin(self.scale * omega * (idx + self.offset) + phase) * self.A.unsqueeze(0)
        # Interleave sin/cos for stability (optional): already sin-only by design
        self._table = val  # (V, D)

    @torch.no_grad()
    def embedding_matrix(self) -> torch.Tensor:
        if self._table.numel() == 0 and self._vocab_size > 0:
            self._rebuild_table()
        return self._table.detach()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Returns embeddings for token indices (N,) or (B,T)."""
        if self._vocab_size <= 0:
            raise RuntimeError("set_vocab_size() must be called before using SineTokenEmbedder")
        # Compute on the fly to support gradient if trainable
        device = self.A.device
        ids = token_ids.to(device=device, dtype=torch.float32)
        omega = self._frequencies(device).view(*([1] * ids.ndim), -1)
        phase = self.phi.view(*([1] * ids.ndim), -1)
        A = self.A.view(*([1] * ids.ndim), -1)
        off = self.offset.view(*([1] * ids.ndim), *([1] * (ids.ndim - 0)))
        # Expand ids to (..., 1)
        ids_e = ids.unsqueeze(-1)
        emb = torch.sin(self.scale * omega * (ids_e + off) + phase) * A
        return emb

