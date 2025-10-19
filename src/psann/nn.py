from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from ._aliases import resolve_int_alias
from .activations import SineParam
from .state import StateConfig, StateController, ensure_state_config
from .utils import init_siren_linear_


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over feature dimension (assume (N, F))
        if x.ndim < 2:
            return x
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight.unsqueeze(0)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        # Per-sample mask (N, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x * mask / keep


class ResidualPSANNBlock(nn.Module):
    """PreNorm residual block with optional RMS/Layer norm, DropPath, and zero-init scale."""

    def __init__(
        self,
        dim: int,
        *,
        act_kw: Optional[Dict] = None,
        activation_type: str = "psann",
        w0_hidden: float = 1.0,
        norm: str = "rms",  # 'rms'|'layer'|'none'
        drop_path: float = 0.0,
        residual_alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.activation_type = activation_type.lower()
        act_kw = act_kw or {}
        if norm == "layer":
            self.norm = nn.LayerNorm(self.dim)
        elif norm == "rms":
            self.norm = RMSNorm(self.dim)
        else:
            self.norm = nn.Identity()

        self.fc1 = nn.Linear(self.dim, self.dim)
        self.fc2 = nn.Linear(self.dim, self.dim)
        if self.activation_type == "psann":
            self.act1 = SineParam(self.dim, **act_kw)
            self.act2 = SineParam(self.dim, **act_kw)
        elif self.activation_type == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif self.activation_type == "tanh":
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
        else:
            raise ValueError("activation_type must be one of: 'psann', 'relu', 'tanh'")

        init_siren_linear_(self.fc1, is_first=False, w0=w0_hidden)
        init_siren_linear_(self.fc2, is_first=False, w0=w0_hidden)
        self.drop_path = DropPath(drop_path)
        self.alpha = nn.Parameter(torch.zeros(()) + float(residual_alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act1(self.fc1(h))
        h = self.act2(self.fc2(h))
        h = self.drop_path(h)
        return x + self.alpha * h


class ResidualPSANNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: int = 8,
        hidden_units: Optional[int] = None,
        hidden_width: Optional[int] = 128,
        act_kw: Optional[Dict] = None,
        activation_type: str = "psann",
        w0_first: float = 12.0,
        w0_hidden: float = 1.0,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
    ) -> None:
        super().__init__()
        act_kw = act_kw or {}
        alias = resolve_int_alias(
            primary_value=hidden_units,
            alias_value=hidden_width,
            primary_name="hidden_units",
            alias_name="hidden_width",
            context="ResidualPSANNNet",
            default=64,
            mismatch_strategy="error",
            mismatch_message=(
                "ResidualPSANNNet: `hidden_units` and `hidden_width` must agree when both provided."
            ),
        )
        units = alias.value if alias.value is not None else 64
        hidden_width = units
        self.hidden_units = units
        self.hidden_width = units
        self.hidden_layers = int(hidden_layers)
        self.output_dim = int(output_dim)
        self.in_linear = nn.Linear(input_dim, hidden_width)
        # SIREN-inspired init for first layer
        init_siren_linear_(self.in_linear, is_first=True, w0=w0_first)
        # Stack of residual blocks
        blocks = []
        for i in range(hidden_layers):
            dp = (
                float(drop_path_max) * (i / max(1, hidden_layers - 1)) if hidden_layers > 1 else 0.0
            )
            blk = ResidualPSANNBlock(
                hidden_width,
                act_kw=act_kw,
                activation_type=activation_type,
                w0_hidden=w0_hidden,
                norm=norm,
                drop_path=dp,
                residual_alpha_init=residual_alpha_init,
            )
            blocks.append(blk)
        self.body = nn.Sequential(*blocks)
        self.head_norm = (
            nn.LayerNorm(hidden_width)
            if norm == "layer"
            else (RMSNorm(hidden_width) if norm == "rms" else nn.Identity())
        )
        self.head = nn.Linear(hidden_width, output_dim)
        init_siren_linear_(self.head, is_first=False, w0=w0_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_linear(x)
        z = self.body(z) if len(self.body) > 0 else z
        z = self.head_norm(z)
        return self.head(z)


class PSANNBlock(nn.Module):
    """Linear layer followed by parameterized sine activation.

    Optional per-feature persistent state acts as an amplitude modulator.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        act_kw: Optional[Dict] = None,
        state_cfg: Optional[Union[StateConfig, Mapping[str, Any]]] = None,
        activation_type: str = "psann",
    ) -> None:
        super().__init__()
        act_kw = act_kw or {}
        self.linear = nn.Linear(in_features, out_features)
        self.activation_type = activation_type.lower()
        if self.activation_type == "psann":
            self.act = SineParam(out_features, **act_kw)
        elif self.activation_type == "relu":
            self.act = nn.ReLU()
        elif self.activation_type == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError("activation_type must be one of: 'psann', 'relu', 'tanh'")
        cfg = ensure_state_config(state_cfg)
        self.state_ctrl = StateController(out_features, **cfg.to_kwargs()) if cfg else None
        self.enable_state_updates = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        y = self.act(z)
        if self.state_ctrl is not None:
            update_flag = self.training and self.enable_state_updates
            y = self.state_ctrl.apply(y, feature_dim=1, update=update_flag)  # (N, F)
        return y


class PSANNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: int = 2,
        hidden_units: Optional[int] = None,
        hidden_width: Optional[int] = 64,
        act_kw: Optional[Dict] = None,
        state_cfg: Optional[Dict] = None,
        activation_type: str = "psann",
        w0: float = 30.0,
    ) -> None:
        super().__init__()
        act_kw = act_kw or {}
        alias = resolve_int_alias(
            primary_value=hidden_units,
            alias_value=hidden_width,
            primary_name="hidden_units",
            alias_name="hidden_width",
            context="PSANNNet",
            default=64,
            mismatch_strategy="error",
            mismatch_message=(
                "PSANNNet: `hidden_units` and `hidden_width` must agree when both provided."
            ),
        )
        units = alias.value if alias.value is not None else 64
        hidden_width = units
        self.hidden_units = units
        self.hidden_width = units

        layers = []
        prev = input_dim
        for i in range(hidden_layers):
            block = PSANNBlock(
                prev,
                hidden_width,
                act_kw=act_kw,
                state_cfg=state_cfg,
                activation_type=activation_type,
            )
            layers.append(block)
            prev = hidden_width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, output_dim)

        # SIREN-inspired initialization
        if hidden_layers > 0:
            if isinstance(self.body[0], PSANNBlock):
                init_siren_linear_(self.body[0].linear, is_first=True, w0=w0)
            for block in list(self.body)[1:]:
                init_siren_linear_(block.linear, is_first=False, w0=w0)
        init_siren_linear_(self.head, is_first=False, w0=w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.body) > 0:
            x = self.body(x)
        return self.head(x)

    def reset_state(self) -> None:
        for m in self.modules():
            if isinstance(m, PSANNBlock) and getattr(m, "state_ctrl", None) is not None:
                # reset to 1.0 by default
                m.state_ctrl.reset_like_init(1.0)

    def commit_state_updates(self) -> None:
        for m in self.modules():
            if isinstance(m, PSANNBlock) and getattr(m, "state_ctrl", None) is not None:
                m.state_ctrl.commit()

    def set_state_updates(self, enabled: bool) -> None:
        for m in self.modules():
            if isinstance(m, PSANNBlock):
                m.enable_state_updates = bool(enabled)


class WithPreprocessor(nn.Module):
    """Wrap a preprocessor module with a core predictor.

    - preproc: optional nn.Module applied to inputs first (e.g., LSM/LSMConv2d)
    - core: PSANNNet or PSANNConvNdNet that consumes the preprocessed features

    Methods like reset_state/commit_state_updates/set_state_updates are
    forwarded to the core if present.
    """

    def __init__(self, preproc: nn.Module | None, core: nn.Module) -> None:
        super().__init__()
        self.preproc = preproc if preproc is not None else None
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x if self.preproc is None else self.preproc(x)
        return self.core(z)

    # Stateful helpers are delegated to core if available
    def reset_state(self) -> None:
        if hasattr(self.core, "reset_state"):
            self.core.reset_state()

    def commit_state_updates(self) -> None:
        if hasattr(self.core, "commit_state_updates"):
            self.core.commit_state_updates()

    def set_state_updates(self, enabled: bool) -> None:
        if hasattr(self.core, "set_state_updates"):
            self.core.set_state_updates(enabled)
