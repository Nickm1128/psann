from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ._aliases import resolve_int_alias
from .activations import SineParam
from .utils import init_siren_linear_


class _PSANNConvBlockNd(nn.Module):
    def __init__(
        self,
        conv: nn.Module,
        out_channels: int,
        *,
        act_kw: Optional[dict] = None,
        activation_type: str = "psann",
    ) -> None:
        super().__init__()
        self.conv = conv
        act_kw = dict(act_kw or {})
        activation_type = activation_type.lower()
        if activation_type == "psann":
            act_kw.setdefault("feature_dim", 1)  # channel dimension
            self.act = SineParam(out_channels, **act_kw)
        elif activation_type == "relu":
            self.act = nn.ReLU()
        elif activation_type == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError("activation_type must be one of: 'psann', 'relu', 'tanh'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        return self.act(z)


def _init_siren_conv_(conv: nn.Module, *, is_first: bool, w0: float = 30.0) -> None:
    """Initialize a ConvNd layer following a SIREN-like heuristic.

    Uses fan-in = in_channels * prod(kernel_size) to compute bounds analogous to linear layers.
    """
    if not hasattr(conv, "weight"):
        return
    weight = conv.weight
    # kernel_size may be tuple
    if hasattr(conv, "kernel_size"):
        ks = conv.kernel_size
        if isinstance(ks, int):
            kprod = ks
        else:
            kprod = 1
            for k in ks:
                kprod *= k
    else:
        kprod = 1
    in_features = weight.shape[1] * max(1, kprod)
    bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / max(w0, 1e-6))
    torch.nn.init.uniform_(weight, -bound, bound)
    if getattr(conv, "bias", None) is not None:
        torch.nn.init.uniform_(conv.bias, -bound, bound)


class PSANNConv1dNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        *,
        hidden_layers: int = 2,
        conv_channels: Optional[int] = None,
        hidden_channels: Optional[int] = 64,
        kernel_size: int | Sequence[int] = 1,
        act_kw: Optional[dict] = None,
        activation_type: str = "psann",
        w0: float = 30.0,
        segmentation_head: bool = False,
    ) -> None:
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, int) else int(kernel_size[0])
        channels_res = resolve_int_alias(
            primary_value=conv_channels,
            alias_value=hidden_channels,
            primary_name="conv_channels",
            alias_name="hidden_channels",
            context="PSANNConv1dNet",
            default=64,
            mismatch_strategy="error",
            mismatch_message="PSANNConv1dNet: `conv_channels` and `hidden_channels` must agree when both provided.",
        )
        channels = channels_res.value if channels_res.value is not None else 64
        hidden_channels = channels
        self.conv_channels = channels
        self.hidden_channels = channels
        self.hidden_layers = int(hidden_layers)
        self.out_dim = int(out_dim)

        layers = []
        c = in_channels
        for i in range(hidden_layers):
            conv = nn.Conv1d(c, hidden_channels, kernel_size=ks, padding=(ks // 2 if ks > 1 else 0))
            block = _PSANNConvBlockNd(
                conv, hidden_channels, act_kw=act_kw, activation_type=activation_type
            )
            layers.append(block)
            c = hidden_channels
        self.body = nn.Sequential(*layers)
        self.segmentation_head = segmentation_head
        if segmentation_head:
            self.head = nn.Conv1d(c, out_dim, kernel_size=1)
        else:
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(c, out_dim)
        # Initialization
        if hidden_layers > 0:
            # First conv as first, others as deeper
            _init_siren_conv_(self.body[0].conv, is_first=True, w0=w0)
            for blk in list(self.body)[1:]:
                _init_siren_conv_(blk.conv, is_first=False, w0=w0)
        if segmentation_head:
            _init_siren_conv_(self.head, is_first=False, w0=w0)
        else:
            init_siren_linear_(self.fc, is_first=False, w0=w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        if len(self.body) > 0:
            x = self.body(x)
        if self.segmentation_head:
            return self.head(x)  # (N, out_dim, L)
        x = self.pool(x).squeeze(-1)  # (N, C)
        return self.fc(x)


class PSANNConv2dNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        *,
        hidden_layers: int = 2,
        conv_channels: Optional[int] = None,
        hidden_channels: Optional[int] = 64,
        kernel_size: int | Sequence[int] = 1,
        act_kw: Optional[dict] = None,
        activation_type: str = "psann",
        w0: float = 30.0,
        segmentation_head: bool = False,
    ) -> None:
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, int) else int(kernel_size[0])
        channels_res = resolve_int_alias(
            primary_value=conv_channels,
            alias_value=hidden_channels,
            primary_name="conv_channels",
            alias_name="hidden_channels",
            context="PSANNConv2dNet",
            default=64,
            mismatch_strategy="error",
            mismatch_message="PSANNConv2dNet: `conv_channels` and `hidden_channels` must agree when both provided.",
        )
        channels = channels_res.value if channels_res.value is not None else 64
        hidden_channels = channels
        self.conv_channels = channels
        self.hidden_channels = channels
        self.hidden_layers = int(hidden_layers)
        self.out_dim = int(out_dim)

        layers = []
        c = in_channels
        for i in range(hidden_layers):
            conv = nn.Conv2d(c, hidden_channels, kernel_size=ks, padding=(ks // 2 if ks > 1 else 0))
            block = _PSANNConvBlockNd(
                conv, hidden_channels, act_kw=act_kw, activation_type=activation_type
            )
            layers.append(block)
            c = hidden_channels
        self.body = nn.Sequential(*layers)
        self.segmentation_head = segmentation_head
        if segmentation_head:
            self.head = nn.Conv2d(c, out_dim, kernel_size=1)
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(c, out_dim)
        # Initialization
        if hidden_layers > 0:
            _init_siren_conv_(self.body[0].conv, is_first=True, w0=w0)
            for blk in list(self.body)[1:]:
                _init_siren_conv_(blk.conv, is_first=False, w0=w0)
        if segmentation_head:
            _init_siren_conv_(self.head, is_first=False, w0=w0)
        else:
            init_siren_linear_(self.fc, is_first=False, w0=w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        if len(self.body) > 0:
            x = self.body(x)
        if self.segmentation_head:
            return self.head(x)  # (N, out_dim, H, W)
        x = self.pool(x).flatten(1)  # (N, C)
        return self.fc(x)


class PSANNConv3dNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        *,
        hidden_layers: int = 2,
        conv_channels: Optional[int] = None,
        hidden_channels: Optional[int] = 64,
        kernel_size: int | Sequence[int] = 1,
        act_kw: Optional[dict] = None,
        activation_type: str = "psann",
        w0: float = 30.0,
        segmentation_head: bool = False,
    ) -> None:
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, int) else int(kernel_size[0])
        channels_res = resolve_int_alias(
            primary_value=conv_channels,
            alias_value=hidden_channels,
            primary_name="conv_channels",
            alias_name="hidden_channels",
            context="PSANNConv3dNet",
            default=64,
            mismatch_strategy="error",
            mismatch_message="PSANNConv3dNet: `conv_channels` and `hidden_channels` must agree when both provided.",
        )
        channels = channels_res.value if channels_res.value is not None else 64
        hidden_channels = channels
        self.conv_channels = channels
        self.hidden_channels = channels
        self.hidden_layers = int(hidden_layers)
        self.out_dim = int(out_dim)

        layers = []
        c = in_channels
        for i in range(hidden_layers):
            conv = nn.Conv3d(c, hidden_channels, kernel_size=ks, padding=(ks // 2 if ks > 1 else 0))
            block = _PSANNConvBlockNd(
                conv, hidden_channels, act_kw=act_kw, activation_type=activation_type
            )
            layers.append(block)
            c = hidden_channels
        self.body = nn.Sequential(*layers)
        self.segmentation_head = segmentation_head
        if segmentation_head:
            self.head = nn.Conv3d(c, out_dim, kernel_size=1)
        else:
            self.pool = nn.AdaptiveAvgPool3d(1)
            self.fc = nn.Linear(c, out_dim)
        # Initialization
        if hidden_layers > 0:
            _init_siren_conv_(self.body[0].conv, is_first=True, w0=w0)
            for blk in list(self.body)[1:]:
                _init_siren_conv_(blk.conv, is_first=False, w0=w0)
        if segmentation_head:
            _init_siren_conv_(self.head, is_first=False, w0=w0)
        else:
            init_siren_linear_(self.fc, is_first=False, w0=w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, D, H, W)
        if len(self.body) > 0:
            x = self.body(x)
        if self.segmentation_head:
            return self.head(x)  # (N, out_dim, D, H, W)
        x = self.pool(x).flatten(1)  # (N, C)
        return self.fc(x)


class _DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class _ChannelwiseRMSNorm(nn.Module):
    def __init__(self, channels: int, *, ndims: int, eps: float = 1e-8) -> None:
        super().__init__()
        shape = (1, int(channels)) + (1,) * int(ndims)
        self.weight = nn.Parameter(torch.ones(shape))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            return x
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class ResidualPSANNConvBlock2d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        act_kw: Optional[dict] = None,
        activation_type: str = "psann",
        w0_hidden: float = 1.0,
        norm: str = "rms",
        drop_path: float = 0.0,
        residual_alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        activation_type = activation_type.lower()
        act_kw = dict(act_kw or {})
        if activation_type == "psann":
            act_kw.setdefault("feature_dim", 1)
        padding = kernel_size // 2 if kernel_size > 1 else 0
        self.norm = (
            _ChannelwiseRMSNorm(self.channels, ndims=2)
            if norm == "rms"
            else (nn.GroupNorm(1, self.channels) if norm == "layer" else nn.Identity())
        )
        self.conv1 = nn.Conv2d(
            self.channels, self.channels, kernel_size=kernel_size, padding=padding
        )
        self.conv2 = nn.Conv2d(
            self.channels, self.channels, kernel_size=kernel_size, padding=padding
        )
        if activation_type == "psann":
            self.act1 = SineParam(self.channels, **act_kw)
            self.act2 = SineParam(self.channels, **act_kw)
        elif activation_type == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif activation_type == "tanh":
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
        else:
            raise ValueError("activation_type must be one of: 'psann', 'relu', 'tanh'")
        _init_siren_conv_(self.conv1, is_first=False, w0=w0_hidden)
        _init_siren_conv_(self.conv2, is_first=False, w0=w0_hidden)
        self.drop_path = _DropPath(drop_path)
        self.alpha = nn.Parameter(torch.zeros(()) + float(residual_alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act1(self.conv1(h))
        h = self.act2(self.conv2(h))
        h = self.drop_path(h)
        return x + self.alpha * h


class ResidualPSANNConv2dNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        *,
        hidden_layers: int = 4,
        conv_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        kernel_size: int | Sequence[int] = 3,
        act_kw: Optional[dict] = None,
        activation_type: str = "psann",
        w0_first: float = 12.0,
        w0_hidden: float = 1.0,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
        segmentation_head: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) == 0:
                raise ValueError("kernel_size sequence must be non-empty")
            ks = int(kernel_size[0])
        else:
            ks = int(kernel_size)
        channels_res = resolve_int_alias(
            primary_value=conv_channels,
            alias_value=hidden_channels,
            primary_name="conv_channels",
            alias_name="hidden_channels",
            context="ResidualPSANNConv2dNet",
            default=128,
            mismatch_strategy="error",
            mismatch_message=(
                "ResidualPSANNConv2dNet: `conv_channels` and `hidden_channels` must agree when both provided."
            ),
        )
        channels = channels_res.value if channels_res.value is not None else 128
        hidden_channels = channels
        self.hidden_layers = int(hidden_layers)
        self.hidden_channels = channels
        self.out_dim = int(out_dim)
        self.segmentation_head = bool(segmentation_head)
        act_kw = dict(act_kw or {})
        if activation_type.lower() == "psann":
            act_kw.setdefault("feature_dim", 1)
        self.in_proj = nn.Conv2d(int(in_channels), channels, kernel_size=1)
        _init_siren_conv_(self.in_proj, is_first=True, w0=w0_first)
        blocks = []
        for i in range(self.hidden_layers):
            dp = (
                float(drop_path_max) * (i / max(1, self.hidden_layers - 1))
                if self.hidden_layers > 1
                else 0.0
            )
            blocks.append(
                ResidualPSANNConvBlock2d(
                    channels,
                    kernel_size=ks,
                    act_kw=act_kw,
                    activation_type=activation_type,
                    w0_hidden=w0_hidden,
                    norm=norm,
                    drop_path=dp,
                    residual_alpha_init=residual_alpha_init,
                )
            )
        self.body = nn.Sequential(*blocks)
        if self.segmentation_head:
            self.head_norm = (
                _ChannelwiseRMSNorm(channels, ndims=2)
                if norm == "rms"
                else (nn.GroupNorm(1, channels) if norm == "layer" else nn.Identity())
            )
            self.head = nn.Conv2d(channels, self.out_dim, kernel_size=1)
            _init_siren_conv_(self.head, is_first=False, w0=w0_hidden)
        else:
            self.head_norm = (
                _ChannelwiseRMSNorm(channels, ndims=2)
                if norm == "rms"
                else (nn.GroupNorm(1, channels) if norm == "layer" else nn.Identity())
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(channels, self.out_dim)
            init_siren_linear_(self.fc, is_first=False, w0=w0_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        if len(self.body) > 0:
            z = self.body(z)
        z = self.head_norm(z)
        if self.segmentation_head:
            return self.head(z)
        z = self.pool(z).flatten(1)
        return self.fc(z)
