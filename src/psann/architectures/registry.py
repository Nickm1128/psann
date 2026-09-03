"""Registry-driven construction of the existing PSANN numerical backbones."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, cast

import torch
import torch.nn as nn

from ..attention import AttentionConfig as LegacyAttentionConfig
from ..attention import build_attention_module
from ..conv import PSANNConv1dNet, PSANNConv2dNet, PSANNConv3dNet, ResidualPSANNConv2dNet
from ..layers import SpectralGate1D
from ..models import WaveResNet
from ..nn import PSANNNet, ResidualPSANNNet, SGRPSANNSequenceNet, WithPreprocessor
from ..nn_geo_sparse import GeoSparseNet
from .wrappers import (
    _AttentionConvModel,
    _AttentionDenseModel,
    _WaveResNetAttentionDenseModel,
    _WaveResNetConvModel,
    _WaveResNetSpectralDenseModel,
)
from .config import ArchitectureConfig, _canonical_name, validate_architecture


class ArchitectureLifecycle:
    """No-op lifecycle used by all builders other than Wave.

    Hooks intentionally accept keyword-only runtime data.  Training may add data as
    it becomes necessary without handing an estimator instance to a builder.
    """

    def on_model_built(self, *, model: nn.Module, runtime: dict[str, object]) -> None:
        return None

    def on_fit_start(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        runtime: dict[str, object],
    ) -> None:
        return None

    def before_optimizer_step(
        self, *, model: nn.Module, optimizer: torch.optim.Optimizer, runtime: dict[str, object]
    ) -> None:
        return None

    def on_epoch_end(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: Mapping[str, float],
        runtime: dict[str, object],
    ) -> None:
        return None

    def on_fit_end(self, *, model: nn.Module, runtime: dict[str, object]) -> None:
        return None

    def structure_metadata(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True)
class ArchitectureCapabilities:
    kind: str
    input_topologies: frozenset[str]
    supports_per_element: bool
    supports_attention: bool
    supports_state: bool
    supports_context: bool
    supports_preprocessor: bool
    supports_amp: bool
    supports_compile: bool


@dataclass(frozen=True)
class ArchitectureBuildRequest:
    architecture: ArchitectureConfig
    hidden_layers: int
    hidden_units: int
    input_shape: tuple[int, ...]
    input_dim: int
    output_dim: int
    spatial_shape: tuple[int, ...] | None
    spatial_ndim: int | None
    in_channels: int | None
    sequence_length: int | None
    token_dim: int | None
    per_element: bool
    device: torch.device
    dtype: torch.dtype
    preprocessor: nn.Module | None
    preprocessor_output_dim: int | None
    structure_metadata: Mapping[str, object] | None
    w0: float = 30.0


@dataclass(frozen=True)
class ArchitectureBuildResult:
    model: nn.Module
    capabilities: ArchitectureCapabilities
    lifecycle: ArchitectureLifecycle


ArchitectureBuilder = Callable[[ArchitectureBuildRequest], ArchitectureBuildResult]
_BUILDERS: dict[str, ArchitectureBuilder] = {}


def register_architecture(
    kind: str, builder: ArchitectureBuilder, *, replace: bool = False
) -> None:
    key = _canonical_name(kind)
    if not key:
        raise ValueError("Architecture registry key cannot be empty.")
    if key in _BUILDERS and not replace:
        raise ValueError(f"Architecture {key!r} is already registered.")
    _BUILDERS[key] = builder


def get_architecture_builder(kind: str) -> ArchitectureBuilder:
    key = _canonical_name(kind)
    try:
        return _BUILDERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_BUILDERS)) or "<none>"
        raise ValueError(f"Unknown architecture {kind!r}. Available: {available}.") from exc


def available_architectures() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def build_architecture(request: ArchitectureBuildRequest) -> ArchitectureBuildResult:
    validate_architecture(request.architecture, hidden_layers=request.hidden_layers)
    return get_architecture_builder(request.architecture.kind)(request)


def _activation_kwargs(config: ArchitectureConfig) -> dict[str, object]:
    activation = config.activation
    return {
        "amplitude_init": activation.amplitude_init,
        "frequency_init": activation.frequency_init,
        "decay_init": activation.decay_init,
        "learnable": activation.learnable,
        "decay_mode": activation.decay_mode,
        "bounds": dict(activation.bounds) if activation.bounds else None,
    }


def _legacy_attention(config: ArchitectureConfig) -> LegacyAttentionConfig | None:
    attention = config.attention
    if attention is None:
        return None
    return LegacyAttentionConfig(
        kind=attention.kind,
        num_heads=attention.num_heads,
        dropout=attention.dropout,
        bias=attention.bias,
        batch_first=attention.batch_first,
        add_bias_kv=attention.add_bias_kv,
        add_zero_attn=attention.add_zero_attn,
    )


def _maybe_preprocess(request: ArchitectureBuildRequest, model: nn.Module) -> nn.Module:
    if request.preprocessor is None:
        return model
    return WithPreprocessor(request.preprocessor, model)


def _dense_capabilities(request: ArchitectureBuildRequest) -> ArchitectureCapabilities:
    return ArchitectureCapabilities(
        kind="dense",
        input_topologies=frozenset({"flat", "tokens"}),
        supports_per_element=False,
        supports_attention=True,
        supports_state=request.architecture.residual is None,
        supports_context=False,
        supports_preprocessor=True,
        supports_amp=True,
        supports_compile=True,
    )


def _dense_builder(request: ArchitectureBuildRequest) -> ArchitectureBuildResult:
    cfg = request.architecture
    input_dim = request.preprocessor_output_dim or request.input_dim
    activation = _activation_kwargs(cfg)
    state = cfg.state
    state_kwargs = (
        None
        if state is None
        else {
            "rho": state.rho,
            "beta": state.beta,
            "init": state.init,
            "max_abs": state.max_abs,
            "detach": state.detach,
        }
    )
    if cfg.residual is None:
        core: nn.Module = PSANNNet(
            int(input_dim),
            request.output_dim,
            hidden_layers=request.hidden_layers,
            hidden_units=request.hidden_units,
            hidden_width=request.hidden_units,
            act_kw=activation,
            state_cfg=state_kwargs,
            activation_type=cfg.activation.kind,
            w0=request.w0,
        )
    else:
        residual = cfg.residual
        core = ResidualPSANNNet(
            int(input_dim),
            request.output_dim,
            hidden_layers=request.hidden_layers,
            hidden_units=request.hidden_units,
            hidden_width=request.hidden_units,
            act_kw=activation,
            activation_type=cfg.activation.kind,
            w0_first=residual.first_w0,
            w0_hidden=residual.hidden_w0,
            norm=residual.norm,
            drop_path_max=residual.drop_path,
            residual_alpha_init=residual.alpha_init,
        )
    attention = _legacy_attention(cfg)
    if attention is not None:
        if len(request.input_shape) < 2:
            raise ValueError(
                "attention requires token-shaped inputs with at least two non-batch axes."
            )
        token_dim = int(request.input_shape[-1])
        seq_len = int(math.prod(request.input_shape[:-1]))
        if request.hidden_units % attention.num_heads:
            raise ValueError("attention.num_heads must divide hidden_units.")
        if cfg.residual is None:
            token_core = PSANNNet(
                token_dim,
                request.hidden_units,
                hidden_layers=request.hidden_layers,
                hidden_units=request.hidden_units,
                hidden_width=request.hidden_units,
                act_kw=activation,
                state_cfg=state_kwargs,
                activation_type=cfg.activation.kind,
                w0=request.w0,
            )
        else:
            residual = cfg.residual
            token_core = ResidualPSANNNet(
                token_dim,
                request.hidden_units,
                hidden_layers=request.hidden_layers,
                hidden_units=request.hidden_units,
                hidden_width=request.hidden_units,
                act_kw=activation,
                activation_type=cfg.activation.kind,
                w0_first=residual.first_w0,
                w0_hidden=residual.hidden_w0,
                norm=residual.norm,
                drop_path_max=residual.drop_path,
                residual_alpha_init=residual.alpha_init,
            )
        core = _AttentionDenseModel(
            token_core,
            build_attention_module(attention, request.hidden_units),
            seq_len=seq_len,
            token_dim=token_dim,
            embed_dim=request.hidden_units,
            output_dim=request.output_dim,
            pool="mean",
        )
    return ArchitectureBuildResult(
        _maybe_preprocess(request, core), _dense_capabilities(request), ArchitectureLifecycle()
    )


def _convolution_builder(request: ArchitectureBuildRequest) -> ArchitectureBuildResult:
    cfg = request.architecture
    conv = cfg.convolution
    assert conv is not None
    if request.spatial_ndim is None or request.in_channels is None:
        raise ValueError("convolutional architecture requires shaped input topology.")
    channels = request.preprocessor_output_dim or request.in_channels
    activation = _activation_kwargs(cfg)
    if cfg.residual is not None:
        if request.spatial_ndim != 2:
            raise ValueError("residual convolutional architecture requires a 2D input topology.")
        residual = cfg.residual
        core: nn.Module = ResidualPSANNConv2dNet(
            int(channels),
            request.output_dim,
            hidden_layers=request.hidden_layers,
            conv_channels=conv.channels or request.hidden_units,
            hidden_channels=conv.channels or request.hidden_units,
            kernel_size=conv.kernel_size,
            act_kw=activation,
            activation_type=cfg.activation.kind,
            w0_first=residual.first_w0,
            w0_hidden=residual.hidden_w0,
            norm=residual.norm,
            drop_path_max=residual.drop_path,
            residual_alpha_init=residual.alpha_init,
            segmentation_head=request.per_element,
        )
    else:
        cls = {1: PSANNConv1dNet, 2: PSANNConv2dNet, 3: PSANNConv3dNet}.get(request.spatial_ndim)
        if cls is None:
            raise ValueError("convolutional architecture supports 1D, 2D, and 3D topology only.")
        core = cls(
            int(channels),
            request.output_dim,
            hidden_layers=request.hidden_layers,
            conv_channels=conv.channels or request.hidden_units,
            hidden_channels=conv.channels or request.hidden_units,
            kernel_size=conv.kernel_size,
            act_kw=activation,
            activation_type=cfg.activation.kind,
            w0=request.w0,
            segmentation_head=request.per_element,
        )
    attention = _legacy_attention(cfg)
    if attention is not None:
        embed_dim = int(getattr(core, "conv_channels", getattr(core, "hidden_channels", 0)))
        if not embed_dim or embed_dim % attention.num_heads:
            raise ValueError("attention.num_heads must divide convolutional embedding width.")
        if request.spatial_shape is None:
            raise ValueError("attention requires known spatial dimensions.")
        core = _AttentionConvModel(
            core,
            cast(nn.Module, build_attention_module(attention, embed_dim)),
            spatial_shape=request.spatial_shape,
            segmentation_head=request.per_element,
        )
    capabilities = ArchitectureCapabilities(
        "convolutional",
        frozenset({"conv1d", "conv2d", "conv3d"}),
        True,
        True,
        False,
        False,
        True,
        True,
        True,
    )
    return ArchitectureBuildResult(
        _maybe_preprocess(request, core), capabilities, ArchitectureLifecycle()
    )


class WaveLifecycle(ArchitectureLifecycle):
    """Serializable Wave schedule and clipping behavior without estimator subclass hooks."""

    def __init__(
        self,
        config: ArchitectureConfig,
        hidden_layers: int,
        structure: Mapping[str, object] | None = None,
    ) -> None:
        self.config = config
        self.hidden_layers = hidden_layers
        saved = dict(structure or {})
        self.current_depth = int(
            cast(
                Any,
                saved.get(
                    "current_depth",
                    (
                        config.wave.progressive_depth.initial_layers
                        if config.wave and config.wave.progressive_depth
                        else hidden_layers
                    ),
                ),
            )
        )
        self.next_expand_epoch = saved.get("next_expand_epoch")
        self.warmup_step = int(cast(Any, saved.get("warmup_step", 0)))
        self.warmup_active = bool(
            saved.get("warmup_active", bool(config.wave and config.wave.warmup))
        )

    def before_optimizer_step(
        self, *, model: nn.Module, optimizer: torch.optim.Optimizer, runtime: dict[str, object]
    ) -> None:
        if self.config.wave and self.config.wave.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.wave.grad_clip_norm)

    @staticmethod
    def _wave_core(model: nn.Module) -> Any:
        """Unwrap preprocessor/spectral/convolution containers to WaveResNet."""

        current: Any = model
        while hasattr(current, "core"):
            current = current.core
        return getattr(current, "wave", current)

    def _apply_warmup(self, model: nn.Module, step: int) -> None:
        wave = self.config.wave
        if wave is None or wave.warmup is None:
            return
        ratio = min(1.0, step / max(1, wave.warmup.epochs))
        mutable_model = self._wave_core(model)
        if hasattr(mutable_model, "stem_w0"):
            mutable_model.stem_w0 = wave.warmup.first_initial + ratio * (
                wave.first_w0 - wave.warmup.first_initial
            )
        if hasattr(mutable_model, "blocks"):
            for block in mutable_model.blocks:
                if hasattr(block, "w0"):
                    block.w0 = wave.warmup.hidden_initial + ratio * (
                        wave.hidden_w0 - wave.warmup.hidden_initial
                    )

    def on_model_built(self, *, model: nn.Module, runtime: dict[str, object]) -> None:
        if self.warmup_step:
            self._apply_warmup(model, self.warmup_step)

    def on_epoch_end(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        epoch: int,
        metrics: Mapping[str, float],
        runtime: dict[str, object],
    ) -> None:
        wave = self.config.wave
        if wave and wave.warmup and self.warmup_active:
            self.warmup_step = epoch + 1
            self._apply_warmup(model, self.warmup_step)
            ratio = min(1.0, self.warmup_step / max(1, wave.warmup.epochs))
            self.warmup_active = ratio < 1.0
        progressive = wave.progressive_depth if wave else None
        if (
            progressive
            and optimizer is not None
            and (epoch + 1) % progressive.interval == 0
            and self.current_depth < self.hidden_layers
        ):
            add = min(progressive.growth, self.hidden_layers - self.current_depth)
            wave_core = self._wave_core(model)
            if hasattr(wave_core, "add_blocks"):
                blocks = wave_core.add_blocks(add)
                if blocks:
                    optimizer.add_param_group(
                        {"params": [param for block in blocks for param in block.parameters()]}
                    )
                    self.current_depth += add
                    self.next_expand_epoch = epoch + 1 + progressive.interval
                    self._apply_warmup(model, self.warmup_step)

    def structure_metadata(self) -> dict[str, object]:
        return {
            "current_depth": self.current_depth,
            "next_expand_epoch": self.next_expand_epoch,
            "warmup_step": self.warmup_step,
            "warmup_active": self.warmup_active,
        }


def _wave_builder(request: ArchitectureBuildRequest) -> ArchitectureBuildResult:
    cfg = request.architecture
    wave = cfg.wave
    assert wave is not None and cfg.residual is not None
    activation = _activation_kwargs(cfg)
    context = cfg.context
    initial_depth = int(
        (request.structure_metadata or {}).get(
            "current_depth",
            (
                wave.progressive_depth.initial_layers
                if wave.progressive_depth
                else request.hidden_layers
            ),
        )
    )
    first_w0 = wave.warmup.first_initial if wave.warmup else wave.first_w0
    hidden_w0 = wave.warmup.hidden_initial if wave.warmup else wave.hidden_w0
    if cfg.convolution is None:
        core: Any = WaveResNet(
            request.preprocessor_output_dim or request.input_dim,
            request.hidden_units,
            initial_depth,
            request.output_dim,
            first_layer_w0=first_w0,
            hidden_w0=hidden_w0,
            context_dim=context.dim if context else None,
            norm=cast(Any, wave.norm),
            use_film=context.film if context else True,
            use_phase_shift=context.phase_shift if context else True,
            dropout=wave.dropout,
            residual_alpha_init=cfg.residual.alpha_init,
            activation_config=cast(Any, activation),
        )
        if cfg.spectral is not None and len(request.input_shape) >= 2:
            token_dim = request.input_shape[-1]
            seq_len = int(math.prod(request.input_shape[:-1]))
            if seq_len > 1:
                spectral = cfg.spectral
                core = _WaveResNetSpectralDenseModel(
                    core,
                    SpectralGate1D(
                        token_dim,
                        k_fft=spectral.k_fft,
                        gate_type=spectral.gate_type.replace("-", "_"),
                        gate_groups=spectral.groups,
                        gate_init=spectral.init,
                        gate_strength=spectral.strength,
                    ),
                    seq_len=seq_len,
                    token_dim=token_dim,
                )
        elif cfg.attention is not None:
            if len(request.input_shape) < 2:
                raise ValueError("Wave attention requires token-shaped flat input.")
            token_dim = int(request.input_shape[-1])
            seq_len = int(math.prod(request.input_shape[:-1]))
            attention = _legacy_attention(cfg)
            assert attention is not None
            if token_dim % attention.num_heads:
                raise ValueError("attention.num_heads must divide Wave token width.")
            core = _WaveResNetAttentionDenseModel(
                core,
                cast(nn.Module, build_attention_module(attention, token_dim)),
                seq_len=seq_len,
                token_dim=token_dim,
            )
    else:
        conv = cfg.convolution
        if (
            request.spatial_ndim is None
            or request.in_channels is None
            or request.spatial_shape is None
        ):
            raise ValueError("Wave convolutional architecture requires shaped inputs.")
        cls = {1: PSANNConv1dNet, 2: PSANNConv2dNet, 3: PSANNConv3dNet}.get(request.spatial_ndim)
        if cls is None:
            raise ValueError(
                "Wave convolutional architecture supports 1D, 2D, and 3D topology only."
            )
        channels = request.preprocessor_output_dim or request.in_channels
        width = conv.channels or request.hidden_units
        conv_core = cls(
            int(channels),
            out_dim=width,
            hidden_layers=request.hidden_layers,
            conv_channels=width,
            hidden_channels=width,
            kernel_size=conv.kernel_size,
            act_kw=activation,
            activation_type=cfg.activation.kind,
            w0=request.w0,
            segmentation_head=False,
        )
        embed_dim = int(getattr(conv_core, "conv_channels", width))
        seq_len = int(math.prod(request.spatial_shape))
        wave_core = WaveResNet(
            seq_len * embed_dim,
            request.hidden_units,
            initial_depth,
            request.output_dim,
            first_layer_w0=first_w0,
            hidden_w0=hidden_w0,
            context_dim=context.dim if context else None,
            norm=cast(Any, wave.norm),
            use_film=context.film if context else True,
            use_phase_shift=context.phase_shift if context else True,
            dropout=wave.dropout,
            residual_alpha_init=cfg.residual.alpha_init,
            activation_config=cast(Any, activation),
        )
        attention = _legacy_attention(cfg)
        attn_module = build_attention_module(attention, embed_dim) if attention else None
        spectral_gate = None
        if cfg.spectral is not None:
            if request.spatial_ndim != 1:
                raise ValueError("Wave spectral gating requires a 1D convolutional topology.")
            spectral = cfg.spectral
            spectral_gate = SpectralGate1D(
                embed_dim,
                k_fft=spectral.k_fft,
                gate_type=spectral.gate_type.replace("-", "_"),
                gate_groups=spectral.groups,
                gate_init=spectral.init,
                gate_strength=spectral.strength,
            )
        core = _WaveResNetConvModel(
            conv_core,
            wave_core,
            spatial_shape=request.spatial_shape,
            attention_module=attn_module,
            spectral_gate=spectral_gate,
        )
    caps = ArchitectureCapabilities(
        "wave",
        frozenset({"flat", "conv1d", "conv2d", "conv3d"}),
        False,
        True,
        False,
        True,
        True,
        True,
        True,
    )
    return ArchitectureBuildResult(
        _maybe_preprocess(request, core),
        caps,
        WaveLifecycle(cfg, request.hidden_layers, request.structure_metadata),
    )


def _sequence_builder(request: ArchitectureBuildRequest) -> ArchitectureBuildResult:
    cfg = request.architecture
    sequence = cfg.sequence
    assert sequence is not None
    seq_len = request.sequence_length or 1
    token_dim = request.token_dim or request.input_dim
    spectral = cfg.spectral
    core = SGRPSANNSequenceNet(
        seq_len=seq_len,
        token_dim=token_dim,
        output_dim=request.output_dim,
        hidden_layers=request.hidden_layers,
        hidden_units=request.hidden_units,
        hidden_width=request.hidden_units,
        act_kw=_activation_kwargs(cfg),
        activation_type=cfg.activation.kind.replace("-", "_"),
        w0=request.w0,
        phase_init=sequence.phase_init,
        phase_trainable=sequence.phase_trainable,
        use_spectral_gate=spectral is not None,
        k_fft=spectral.k_fft if spectral else 64,
        gate_type=spectral.gate_type.replace("-", "_") if spectral else "rfft",
        gate_groups=spectral.groups if spectral else "depthwise",
        gate_init=spectral.init if spectral else 0.0,
        gate_strength=spectral.strength if spectral else 1.0,
        pool=sequence.pool,
    )
    caps = ArchitectureCapabilities(
        "sequence", frozenset({"flat", "tokens"}), False, False, False, False, True, True, True
    )
    return ArchitectureBuildResult(_maybe_preprocess(request, core), caps, ArchitectureLifecycle())


def _geometry_builder(request: ArchitectureBuildRequest) -> ArchitectureBuildResult:
    cfg = request.architecture
    geometry = cfg.geometry
    residual = cfg.residual
    assert geometry is not None and residual is not None
    shape = geometry.shape
    if shape is None and len(request.input_shape) == 2:
        shape = (request.input_shape[0], request.input_shape[1])
    if shape is None:
        raise ValueError(
            "geometric-sparse architecture requires geometry.shape or HxW input shape."
        )
    if shape[0] * shape[1] != request.input_dim:
        raise ValueError("geometry.shape product must equal input dimension.")
    core = GeoSparseNet(
        request.input_dim,
        request.output_dim,
        shape=shape,
        depth=request.hidden_layers,
        k=geometry.k,
        pattern=geometry.pattern,
        radius=geometry.radius,
        offsets=geometry.offsets,
        wrap_mode=geometry.wrap_mode,
        activation_type=cfg.activation.kind.replace("-", "_"),
        activation_config={
            **_activation_kwargs(cfg),
            "slope_init": cfg.activation.slope_init,
            "slope_trainable": cfg.activation.slope_trainable,
            "clip_max": cfg.activation.clip_max,
            "activation_types": cfg.activation.activation_types,
            "activation_ratios": cfg.activation.activation_ratios,
        },
        norm=residual.norm,
        drop_path_max=residual.drop_path,
        residual_alpha_init=residual.alpha_init,
        bias=geometry.bias,
        compute_mode=geometry.compute_mode,
        seed=geometry.seed,
    )
    caps = ArchitectureCapabilities(
        "geometric-sparse",
        frozenset({"flat", "grid"}),
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    )
    return ArchitectureBuildResult(_maybe_preprocess(request, core), caps, ArchitectureLifecycle())


for _kind, _builder in {
    "dense": _dense_builder,
    "convolutional": _convolution_builder,
    "wave": _wave_builder,
    "sequence": _sequence_builder,
    "geometric-sparse": _geometry_builder,
}.items():
    register_architecture(_kind, _builder)


__all__ = [
    "ArchitectureBuildRequest",
    "ArchitectureBuildResult",
    "ArchitectureCapabilities",
    "ArchitectureLifecycle",
    "WaveLifecycle",
    "available_architectures",
    "build_architecture",
    "get_architecture_builder",
    "register_architecture",
]
