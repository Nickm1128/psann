from __future__ import annotations

import math

import torch
from torch import nn


class SpectralGate1D(nn.Module):
    """Spectral gating residual branch for (B, T, D) sequences."""

    def __init__(
        self,
        channels: int,
        *,
        k_fft: int = 64,
        gate_type: str = "rfft",
        gate_groups: str = "depthwise",
        gate_init: float = 0.0,
        gate_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if k_fft <= 0:
            raise ValueError("k_fft must be positive")
        gate_type = str(gate_type).lower()
        if gate_type not in {"rfft", "fourier_features"}:
            raise ValueError("gate_type must be 'rfft' or 'fourier_features'")
        gate_groups = str(gate_groups).lower()
        if gate_groups not in {"depthwise", "full"}:
            raise ValueError("gate_groups must be 'depthwise' or 'full'")
        if gate_strength < 0:
            raise ValueError("gate_strength must be >= 0")

        self.channels = int(channels)
        self.k_fft = int(k_fft)
        self.gate_type = gate_type
        self.gate_groups = gate_groups
        self.gate_strength = float(gate_strength)

        freq_bins = self.k_fft // 2 + 1
        self.mask = nn.Parameter(torch.full((self.channels, freq_bins), float(gate_init)))

        groups = self.channels if self.gate_groups == "depthwise" else 1
        self.mix = nn.Conv1d(self.channels, self.channels, kernel_size=1, groups=groups, bias=False)
        self._init_mix_identity()

    def _init_mix_identity(self) -> None:
        with torch.no_grad():
            if self.mix.groups == self.channels:
                self.mix.weight.fill_(1.0)
            else:
                self.mix.weight.zero_()
                for i in range(self.channels):
                    self.mix.weight[i, i, 0] = 1.0

    def _window_input(self, x: torch.Tensor) -> torch.Tensor:
        """Return a fixed-length window of shape (B, K, D)."""
        bsz, steps, dim = x.shape
        k = self.k_fft
        if steps >= k:
            return x[:, -k:, :]
        window = x.new_zeros(bsz, k, dim)
        window[:, -steps:, :] = x
        return window

    def _maybe_cast_fft(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
        if x.dtype in {torch.float16, torch.bfloat16}:
            return x.float(), x.dtype
        return x, x.dtype

    def _fourier_features(self, xt: torch.Tensor) -> torch.Tensor:
        k = xt.shape[-1]
        device = xt.device
        dtype = xt.dtype
        freq_bins = self.mask.shape[1]
        t = torch.arange(k, device=device, dtype=dtype)
        freqs = torch.arange(freq_bins, device=device, dtype=dtype)
        angle = 2.0 * math.pi * t[:, None] * freqs[None, :] / float(k)
        cos = torch.cos(angle)
        sin = torch.sin(angle)

        cos_coeff = torch.matmul(xt, cos) / float(k)
        sin_coeff = torch.matmul(xt, sin) / float(k)
        gate = torch.sigmoid(self.mask).to(dtype=cos_coeff.dtype)[None, :, :]
        cos_coeff = cos_coeff * gate
        sin_coeff = sin_coeff * gate
        return torch.matmul(cos_coeff, cos.transpose(0, 1)) + torch.matmul(
            sin_coeff, sin.transpose(0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("SpectralGate1D expects input shaped (B, T, D).")
        bsz, steps, dim = x.shape
        if dim != self.channels:
            raise ValueError(
                f"SpectralGate1D expected {self.channels} channels, received {dim}."
            )

        window = self._window_input(x)
        xt = window.transpose(1, 2)  # (B, D, K)
        xt_cast, orig_dtype = self._maybe_cast_fft(xt)
        if self.gate_type == "rfft":
            xf = torch.fft.rfft(xt_cast, n=self.k_fft, dim=-1)
            gate = torch.sigmoid(self.mask).to(dtype=xf.real.dtype)[None, :, :]
            xf = xf * gate
            xr = torch.fft.irfft(xf, n=self.k_fft, dim=-1)
        else:
            xr = self._fourier_features(xt_cast)

        xr = self.mix(xr)
        if xr.dtype != orig_dtype:
            xr = xr.to(dtype=orig_dtype)
        xr = xr.transpose(1, 2)  # (B, K, D)

        out = x.new_zeros(bsz, steps, dim)
        k_out = steps if steps < self.k_fft else self.k_fft
        out[:, -k_out:, :] = xr[:, -k_out:, :] * self.gate_strength
        return out
