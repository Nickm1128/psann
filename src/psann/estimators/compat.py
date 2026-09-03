"""Deprecated, thin public facades for the former variant estimators."""

from __future__ import annotations

import warnings
from typing import Any, Mapping

from ..architectures import (
    ActivationConfig,
    ArchitectureConfig,
    AttentionConfig,
    ContextConfig,
    ConvolutionConfig,
    GeometryConfig,
    ProgressiveDepthConfig,
    ResidualConfig,
    SequenceConfig,
    SpectralConfig,
    W0WarmupConfig,
    WaveConfig,
)
from .regressor import PSANNRegressor


def _activation(kwargs: dict[str, Any]) -> ActivationConfig:
    raw = kwargs.pop("activation", None)
    kind = kwargs.pop("activation_type", "psann")
    values = dict(raw) if isinstance(raw, Mapping) else {}
    values.setdefault("kind", kind)
    return ActivationConfig(**values)


def _attention(kwargs: dict[str, Any]) -> AttentionConfig | None:
    raw = kwargs.pop("attention", None)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        values = dict(raw)
    else:
        values = {
            key: getattr(raw, key)
            for key in (
                "kind",
                "num_heads",
                "dropout",
                "bias",
                "batch_first",
                "add_bias_kv",
                "add_zero_attn",
            )
            if hasattr(raw, key)
        }
    if str(values.get("kind", "mha")).strip().lower() in {"", "none", "off"}:
        return None
    return AttentionConfig(**values)


def _common(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Leave only canonical flat estimator arguments for the parent constructor."""

    accepted = {
        "hidden_layers",
        "hidden_units",
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "weight_decay",
        "early_stopping",
        "patience",
        "num_workers",
        "warm_start",
        "loss",
        "loss_params",
        "loss_reduction",
        "scaler",
        "scaler_params",
        "target_scaler",
        "target_scaler_params",
        "output_shape",
        "device",
        "random_state",
        "amp",
        "amp_dtype",
        "compile",
        "compile_backend",
        "compile_mode",
        "compile_fullgraph",
        "compile_dynamic",
        "hidden_width",
        "w0",
        "lsm",
        "lsm_train",
        "lsm_pretrain_epochs",
        "lsm_lr",
    }
    unknown = set(kwargs) - accepted
    if unknown:
        raise TypeError(f"Unexpected legacy estimator argument {sorted(unknown)[0]!r}.")
    return kwargs


class _LegacyFacade(PSANNRegressor):
    """Shared warning/clone behavior; subclasses only construct a config."""

    _legacy_name = "legacy estimator"

    def _warn(self) -> None:
        warnings.warn(
            f"{self._legacy_name} is deprecated; use PSANNRegressor(architecture=...).",
            DeprecationWarning,
            stacklevel=3,
        )


class ResPSANNRegressor(_LegacyFacade):
    _legacy_name = "ResPSANNRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        architecture = kwargs.pop("architecture", None)
        if architecture is None:
            activation = _activation(kwargs)
            preserve_shape = bool(kwargs.pop("preserve_shape", False))
            data_format = kwargs.pop("data_format", "channels_first")
            kernel = kwargs.pop("conv_kernel_size", 1)
            channels = kwargs.pop("conv_channels", None)
            per_element = kwargs.pop("per_element", False)
            residual = ResidualConfig(
                kwargs.pop("norm", "rms"),
                kwargs.pop("residual_alpha_init", 0.0),
                kwargs.pop("drop_path_max", 0.0),
                kwargs.pop("w0_first", 12.0),
                kwargs.pop("w0_hidden", 1.0),
            )
            attention = _attention(kwargs)
            kwargs.pop("stateful", False)
            kwargs.pop("state", None)
            kwargs.pop("state_reset", None)
            kwargs.pop("stream_lr", None)
            if preserve_shape:
                architecture = ArchitectureConfig.convolutional(
                    activation=activation,
                    residual=residual,
                    convolution=ConvolutionConfig(channels, kernel, data_format, per_element),
                    attention=attention,
                )
            else:
                architecture = ArchitectureConfig.dense(
                    activation=activation, residual=residual, attention=attention
                )
        kwargs.setdefault("hidden_layers", 8)
        super().__init__(architecture=architecture, **_common(kwargs))


class ResConvPSANNRegressor(_LegacyFacade):
    _legacy_name = "ResConvPSANNRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        architecture = kwargs.pop("architecture", None)
        if architecture is None:
            activation = _activation(kwargs)
            kwargs.pop("preserve_shape", None)
            data_format = kwargs.pop("data_format", "channels_first")
            kernel = kwargs.pop("conv_kernel_size", 3)
            channels = kwargs.pop("conv_channels", None)
            per_element = kwargs.pop("per_element", False)
            residual = ResidualConfig(
                kwargs.pop("norm", "rms"),
                kwargs.pop("residual_alpha_init", 0.0),
                kwargs.pop("drop_path_max", 0.0),
                kwargs.pop("w0_first", 12.0),
                kwargs.pop("w0_hidden", 1.0),
            )
            _attention(kwargs)
            for name in ("stateful", "state", "state_reset", "stream_lr"):
                kwargs.pop(name, None)
            architecture = ArchitectureConfig.convolutional(
                activation=activation,
                residual=residual,
                convolution=ConvolutionConfig(channels, kernel, data_format, per_element),
            )
        kwargs.setdefault("hidden_layers", 6)
        kwargs.setdefault("batch_size", 64)
        super().__init__(architecture=architecture, **_common(kwargs))


class WaveResNetRegressor(_LegacyFacade):
    _legacy_name = "WaveResNetRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        architecture = kwargs.pop("architecture", None)
        if architecture is None:
            activation = _activation(kwargs)
            preserve_shape = bool(kwargs.pop("preserve_shape", False))
            data_format = kwargs.pop("data_format", "channels_first")
            kernel = kwargs.pop("conv_kernel_size", 1)
            channels = kwargs.pop("conv_channels", None)
            kwargs.pop("per_element", None)
            attention = _attention(kwargs)
            residual = ResidualConfig(alpha_init=kwargs.pop("residual_alpha_init", 0.0))
            warmup_first = kwargs.pop("first_layer_w0_initial", 10.0)
            warmup_hidden = kwargs.pop("hidden_w0_initial", 0.5)
            warmup_epochs = kwargs.pop("w0_warmup_epochs", 10)
            progressive_initial = kwargs.pop("progressive_depth_initial", None)
            wave = WaveConfig(
                first_w0=kwargs.pop("first_layer_w0", 30.0),
                hidden_w0=kwargs.pop("hidden_w0", 1.0),
                norm=kwargs.pop("norm", "none"),
                dropout=kwargs.pop("dropout", 0.0),
                grad_clip_norm=kwargs.pop("grad_clip_norm", 5.0),
                warmup=W0WarmupConfig(warmup_first, warmup_hidden, warmup_epochs),
                progressive_depth=(
                    ProgressiveDepthConfig(
                        progressive_initial,
                        kwargs.pop("progressive_depth_interval", 15),
                        kwargs.pop("progressive_depth_growth", 1),
                    )
                    if progressive_initial is not None
                    else None
                ),
            )
            context_dim = kwargs.pop("context_dim", None)
            context_builder = kwargs.pop("context_builder", None)
            context_params = kwargs.pop("context_builder_params", None)
            context = (
                None
                if context_dim is None and context_builder is None
                else ContextConfig(
                    context_dim,
                    context_builder,
                    context_params,
                    kwargs.pop("use_film", True),
                    kwargs.pop("use_phase_shift", True),
                )
            )
            if context is None:
                kwargs.pop("use_film", None)
                kwargs.pop("use_phase_shift", None)
            use_spectral = kwargs.pop("use_spectral_gate", False)
            spectral = (
                SpectralConfig(
                    kwargs.pop("k_fft", 64),
                    kwargs.pop("gate_type", "rfft"),
                    kwargs.pop("gate_groups", "depthwise"),
                    kwargs.pop("gate_init", 0.0),
                    kwargs.pop("gate_strength", 1.0),
                )
                if use_spectral
                else None
            )
            if not use_spectral:
                for name in ("k_fft", "gate_type", "gate_groups", "gate_init", "gate_strength"):
                    kwargs.pop(name, None)
            for name in ("stateful", "state", "state_reset", "stream_lr"):
                kwargs.pop(name, None)
            architecture = ArchitectureConfig.for_wave(
                activation=activation,
                residual=residual,
                wave=wave,
                convolution=(
                    ConvolutionConfig(channels, kernel, data_format, False)
                    if preserve_shape
                    else None
                ),
                attention=attention,
                context=context,
                spectral=spectral,
            )
        kwargs.setdefault("hidden_layers", 6)
        super().__init__(architecture=architecture, **_common(kwargs))


class SGRPSANNRegressor(_LegacyFacade):
    _legacy_name = "SGRPSANNRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        architecture = kwargs.pop("architecture", None)
        if architecture is None:
            activation = _activation(kwargs)
            if activation.kind != "psann":
                raise ValueError("SGRPSANNRegressor requires activation_type='psann'.")
            for name in (
                "preserve_shape",
                "data_format",
                "conv_kernel_size",
                "conv_channels",
                "per_element",
                "attention",
                "stateful",
                "state",
                "state_reset",
                "stream_lr",
            ):
                kwargs.pop(name, None)
            spectral_enabled = kwargs.pop("use_spectral_gate", True)
            spectral = (
                SpectralConfig(
                    kwargs.pop("k_fft", 64),
                    kwargs.pop("gate_type", "rfft"),
                    kwargs.pop("gate_groups", "depthwise"),
                    kwargs.pop("gate_init", 0.0),
                    kwargs.pop("gate_strength", 1.0),
                )
                if spectral_enabled
                else None
            )
            if not spectral_enabled:
                for name in ("k_fft", "gate_type", "gate_groups", "gate_init", "gate_strength"):
                    kwargs.pop(name, None)
            architecture = ArchitectureConfig.for_sequence(
                activation=activation,
                spectral=spectral,
                sequence=SequenceConfig(
                    kwargs.pop("phase_init", 0.0),
                    kwargs.pop("phase_trainable", True),
                    kwargs.pop("pool", "last"),
                ),
            )
        super().__init__(architecture=architecture, **_common(kwargs))


class GeoSparseRegressor(_LegacyFacade):
    _legacy_name = "GeoSparseRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        architecture = kwargs.pop("architecture", None)
        if architecture is None:
            activation = _activation(kwargs)
            for name in (
                "preserve_shape",
                "data_format",
                "conv_kernel_size",
                "conv_channels",
                "per_element",
                "attention",
                "stateful",
                "state",
                "state_reset",
                "stream_lr",
            ):
                kwargs.pop(name, None)
            residual = ResidualConfig(
                kwargs.pop("norm", "rms"),
                kwargs.pop("residual_alpha_init", 0.0),
                kwargs.pop("drop_path_max", 0.0),
            )
            geometry = GeometryConfig(
                kwargs.pop("shape", None),
                kwargs.pop("k", 8),
                kwargs.pop("pattern", "local"),
                kwargs.pop("radius", 1),
                kwargs.pop("offsets", None),
                kwargs.pop("wrap_mode", "clamp"),
                kwargs.pop("bias", True),
                kwargs.pop("compute_mode", "gather"),
                kwargs.pop("geo_seed", None),
            )
            architecture = ArchitectureConfig.geometric_sparse(
                activation=activation, residual=residual, geometry=geometry
            )
        kwargs.setdefault("hidden_layers", 4)
        super().__init__(architecture=architecture, **_common(kwargs))


__all__ = [
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "WaveResNetRegressor",
    "SGRPSANNRegressor",
    "GeoSparseRegressor",
]
