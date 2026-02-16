from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from ._aliases import resolve_int_alias
from .activations import PhaseSineParam, ReLUSigmoidPSANN, SineParam
from .layers.spectral import SpectralGate1D
from .state import StateConfig, StateController, ensure_state_config
from .utils import init_siren_linear_


def _normalize_base_activation_type(activation_type: str) -> str:
    key = str(activation_type).strip().lower()
    if key in {"sine", "respsann"}:
        return "psann"
    if key in {"rspsann", "rsp", "clipped_psann"}:
        return "relu_sigmoid_psann"
    return key


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
        self.activation_type = _normalize_base_activation_type(activation_type)
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
        elif self.activation_type == "relu_sigmoid_psann":
            self.act1 = ReLUSigmoidPSANN(self.dim, **act_kw)
            self.act2 = ReLUSigmoidPSANN(self.dim, **act_kw)
        elif self.activation_type == "relu":
            self.act1 = nn.ReLU()
            self.act2 = nn.ReLU()
        elif self.activation_type == "tanh":
            self.act1 = nn.Tanh()
            self.act2 = nn.Tanh()
        else:
            raise ValueError(
                "activation_type must be one of: 'psann', 'relu', 'tanh', 'relu_sigmoid_psann'"
            )

        init_siren_linear_(self.fc1, is_first=False, w0=w0_hidden)
        init_siren_linear_(self.fc2, is_first=False, w0=w0_hidden)
        self.drop_path = DropPath(drop_path)
        # FSDP requires parameters to have at least 1 dimension; keep numel=1
        self.alpha = nn.Parameter(torch.full((1,), float(residual_alpha_init)))

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
        self.activation_type = _normalize_base_activation_type(activation_type)
        if self.activation_type == "psann":
            self.act = SineParam(out_features, **act_kw)
        elif self.activation_type == "relu_sigmoid_psann":
            self.act = ReLUSigmoidPSANN(out_features, **act_kw)
        elif self.activation_type == "relu":
            self.act = nn.ReLU()
        elif self.activation_type == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(
                "activation_type must be one of: 'psann', 'relu', 'tanh', 'relu_sigmoid_psann'"
            )
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


class SGRPSANNBlock(nn.Module):
    """PSANN block with per-channel phase shift and optional spectral gate."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        act_kw: Optional[Dict] = None,
        activation_type: str = "psann",
        phase_init: float = 0.0,
        phase_trainable: bool = True,
        use_spectral_gate: bool = True,
        k_fft: int = 64,
        gate_type: str = "rfft",
        gate_groups: str = "depthwise",
        gate_init: float = 0.0,
        gate_strength: float = 1.0,
    ) -> None:
        super().__init__()
        act_kw = dict(act_kw or {})
        act_kw.pop("phase_init", None)
        act_kw.pop("phase_trainable", None)
        self.linear = nn.Linear(in_features, out_features)
        activation_type = _normalize_base_activation_type(activation_type)
        if activation_type != "psann":
            raise ValueError("SGRPSANNBlock requires activation_type='psann'.")
        self.act = PhaseSineParam(
            out_features,
            phase_init=phase_init,
            phase_trainable=phase_trainable,
            **act_kw,
        )
        self.spectral = (
            SpectralGate1D(
                out_features,
                k_fft=k_fft,
                gate_type=gate_type,
                gate_groups=gate_groups,
                gate_init=gate_init,
                gate_strength=gate_strength,
            )
            if use_spectral_gate
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.act(h)
        if self.spectral is not None:
            h = h + self.spectral(h)
        return h


class SGRPSANNSequenceNet(nn.Module):
    """Sequence-aware PSANN core with phase shifts and spectral gating."""

    def __init__(
        self,
        seq_len: int,
        token_dim: int,
        output_dim: int,
        *,
        hidden_layers: int = 2,
        hidden_units: Optional[int] = None,
        hidden_width: Optional[int] = 64,
        act_kw: Optional[Dict] = None,
        activation_type: str = "psann",
        w0: float = 30.0,
        phase_init: float = 0.0,
        phase_trainable: bool = True,
        use_spectral_gate: bool = True,
        k_fft: int = 64,
        gate_type: str = "rfft",
        gate_groups: str = "depthwise",
        gate_init: float = 0.0,
        gate_strength: float = 1.0,
        pool: str = "last",
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        act_kw = act_kw or {}
        activation_type = _normalize_base_activation_type(activation_type)
        if activation_type != "psann":
            raise ValueError("SGRPSANNSequenceNet requires activation_type='psann'.")
        pool = str(pool).lower()
        if pool not in {"mean", "last"}:
            raise ValueError("pool must be 'mean' or 'last'")
        alias = resolve_int_alias(
            primary_value=hidden_units,
            alias_value=hidden_width,
            primary_name="hidden_units",
            alias_name="hidden_width",
            context="SGRPSANNSequenceNet",
            default=64,
            mismatch_strategy="error",
            mismatch_message=(
                "SGRPSANNSequenceNet: `hidden_units` and `hidden_width` must agree when both provided."
            ),
        )
        units = alias.value if alias.value is not None else 64

        self.seq_len = int(seq_len)
        self.token_dim = int(token_dim)
        self.hidden_units = units
        self.hidden_width = units
        self.pool = pool

        layers = []
        prev = int(token_dim)
        for _ in range(int(hidden_layers)):
            block = SGRPSANNBlock(
                prev,
                units,
                act_kw=act_kw,
                activation_type=activation_type,
                phase_init=phase_init,
                phase_trainable=phase_trainable,
                use_spectral_gate=use_spectral_gate,
                k_fft=k_fft,
                gate_type=gate_type,
                gate_groups=gate_groups,
                gate_init=gate_init,
                gate_strength=gate_strength,
            )
            layers.append(block)
            prev = units
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, output_dim)

        if hidden_layers > 0:
            init_siren_linear_(self.body[0].linear, is_first=True, w0=w0)
            for block in list(self.body)[1:]:
                init_siren_linear_(block.linear, is_first=False, w0=w0)
        init_siren_linear_(self.head, is_first=False, w0=w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            expected = self.seq_len * self.token_dim
            if x.shape[1] != expected:
                raise ValueError(
                    f"Expected input with {expected} features, received {x.shape[1]}."
                )
            x = x.reshape(x.shape[0], self.seq_len, self.token_dim)
        elif x.ndim == 3:
            if x.shape[1] != self.seq_len or x.shape[2] != self.token_dim:
                raise ValueError(
                    f"Expected input shaped (B, {self.seq_len}, {self.token_dim}); received {tuple(x.shape)}."
                )
        else:
            raise ValueError("SGRPSANNSequenceNet expects 2D or 3D inputs.")

        if len(self.body) > 0:
            x = self.body(x)
        pooled = x[:, -1, :] if self.pool == "last" else x.mean(dim=1)
        return self.head(pooled)


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
