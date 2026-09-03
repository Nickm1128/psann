"""Deprecated, thin public facades for the former variant estimators."""

from __future__ import annotations

import warnings
from inspect import signature
from typing import Any, Mapping, cast

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
from .._sklearn.geosparse import GeoSparseRegressor as _Phase2GeoSparseRegressor
from .._sklearn.residual import (
    ResConvPSANNRegressor as _Phase2ResConvPSANNRegressor,
    ResPSANNRegressor as _Phase2ResPSANNRegressor,
)
from .._sklearn.sgr import SGRPSANNRegressor as _Phase2SGRPSANNRegressor
from .._sklearn.wave import WaveResNetRegressor as _Phase2WaveResNetRegressor


def _activation(kwargs: dict[str, Any]) -> ActivationConfig:
    raw = kwargs.pop("activation", None)
    kind = kwargs.pop("activation_type", "psann")
    values = dict(raw) if isinstance(raw, Mapping) else {}
    for old_name, new_name in {
        "amp_init": "amplitude_init",
        "freq_init": "frequency_init",
        "damping_init": "decay_init",
    }.items():
        if old_name in values:
            values.setdefault(new_name, values.pop(old_name))
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
        "context_builder",
        "context_builder_params",
    }
    unknown = set(kwargs) - accepted
    if unknown:
        raise TypeError(f"Unexpected legacy estimator argument {sorted(unknown)[0]!r}.")
    return kwargs


_REDUNDANT_ARCHITECTURE_KEYWORDS = {
    "activation",
    "activation_type",
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
    "context_dim",
    "context_builder",
    "context_builder_params",
    "use_film",
    "use_phase_shift",
    "w0_first",
    "w0_hidden",
    "norm",
    "drop_path_max",
    "residual_alpha_init",
    "first_layer_w0",
    "hidden_w0",
    "dropout",
    "grad_clip_norm",
    "first_layer_w0_initial",
    "hidden_w0_initial",
    "w0_warmup_epochs",
    "progressive_depth_initial",
    "progressive_depth_interval",
    "progressive_depth_growth",
    "use_spectral_gate",
    "k_fft",
    "gate_type",
    "gate_groups",
    "gate_init",
    "gate_strength",
    "phase_init",
    "phase_trainable",
    "pool",
    "shape",
    "k",
    "pattern",
    "radius",
    "offsets",
    "wrap_mode",
    "bias",
    "compute_mode",
    "geo_seed",
}


def _discard_redundant_architecture_keywords(kwargs: dict[str, Any]) -> None:
    """Drop flat policy mirrors when a schema-v1 config is already present."""

    for key in _REDUNDANT_ARCHITECTURE_KEYWORDS:
        kwargs.pop(key, None)


class _LegacyFacade(PSANNRegressor):
    """Shared warning/clone behavior; subclasses only construct a config."""

    _legacy_name = "legacy estimator"
    _signature_source: type[object]

    def _warn(self) -> None:
        warnings.warn(
            f"{self._legacy_name} is deprecated; use PSANNRegressor(architecture=...).",
            DeprecationWarning,
            stacklevel=3,
        )

    def _capture_legacy_params(self, supplied: Mapping[str, Any]) -> None:
        values: dict[str, Any] = {}
        for name, parameter in signature(self._signature_source.__init__).parameters.items():
            if name == "self" or parameter.kind.name in {"VAR_KEYWORD", "VAR_POSITIONAL"}:
                continue
            value = supplied.get(name, parameter.default)
            values[name] = value
            # These compatibility attributes intentionally retain their original
            # objects; sklearn.clone verifies constructor identity.
            if not hasattr(self, name):
                setattr(self, name, value)
        self._legacy_params_ = values

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self._legacy_params_)

    def set_params(self, **params: object) -> "_LegacyFacade":
        # Retain the established 0.x channel spelling before reconstructing
        # the thin facade.  The canonical estimator owns the actual rebuild.
        if "hidden_channels" in params:
            params = self._normalize_param_aliases(dict(params))
        candidate = self.get_params(deep=False)
        unknown = set(params) - set(candidate)
        if unknown:
            raise ValueError(
                f"Invalid parameter {sorted(unknown)[0]!r} for {self.__class__.__name__}."
            )
        candidate.update(params)
        rebuilt = cast(Any, self.__class__)(**candidate)
        self.__dict__.clear()
        self.__dict__.update(rebuilt.__dict__)
        return self


class ResPSANNRegressor(_LegacyFacade):
    _legacy_name = "ResPSANNRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        supplied = dict(kwargs)
        stateful = False
        state = None
        architecture = kwargs.pop("architecture", None)
        if architecture is not None:
            _discard_redundant_architecture_keywords(kwargs)
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
            stateful = kwargs.pop("stateful", False)
            state = kwargs.pop("state", None)
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
        if stateful or state is not None:
            self._compat_runtime_warning_ = "ResidualPSANNNet does not currently support stateful configurations; ignoring state_cfg."
        self._capture_legacy_params(supplied)


class ResConvPSANNRegressor(_LegacyFacade):
    _legacy_name = "ResConvPSANNRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        supplied = dict(kwargs)
        if "attention" in kwargs:
            raise TypeError(
                "ResConvPSANNRegressor.__init__() got an unexpected keyword argument 'attention'"
            )
        architecture = kwargs.pop("architecture", None)
        if architecture is not None:
            _discard_redundant_architecture_keywords(kwargs)
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
        self._capture_legacy_params(supplied)


class WaveResNetRegressor(_LegacyFacade):
    _legacy_name = "WaveResNetRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        supplied = dict(kwargs)
        attention = None
        stateful = False
        state = None
        architecture = kwargs.pop("architecture", None)
        if architecture is not None:
            _discard_redundant_architecture_keywords(kwargs)
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
            progressive_interval = kwargs.pop("progressive_depth_interval", 15)
            progressive_growth = kwargs.pop("progressive_depth_growth", 1)
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
                        progressive_interval,
                        progressive_growth,
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
            stateful = kwargs.pop("stateful", False)
            state = kwargs.pop("state", None)
            kwargs.pop("state_reset", None)
            kwargs.pop("stream_lr", None)
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
        self._legacy_context_requested_ = "context_dim" in supplied
        if self.architecture.convolution is not None and self.lsm is not None:
            raise ValueError(
                "WaveResNetRegressor does not support lsm preprocessors when preserve_shape=True."
            )
        self._capture_legacy_params(supplied)
        if stateful or state is not None:
            warnings.warn(
                "WaveResNetRegressor does not support stateful configurations; ignoring state/stateful.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.stateful = False
            self.state = None
            self._legacy_params_["stateful"] = False
            self._legacy_params_["state"] = None

    def _wave_core(self):
        model = getattr(self, "model_", None)
        if model is not None and hasattr(model, "core"):
            model = model.core
        return getattr(model, "wave", model)

    def _initial_w0_values(self):
        wave = self.architecture.wave
        assert wave is not None
        if wave.warmup is None:
            return wave.first_w0, wave.hidden_w0
        return wave.warmup.first_initial, wave.warmup.hidden_initial

    def _target_w0_values(self):
        wave = self.architecture.wave
        assert wave is not None
        return wave.first_w0, wave.hidden_w0

    def _current_w0_values(self):
        core = self._wave_core()
        if core is None:
            return self._initial_w0_values()
        hidden = core.blocks[0].w0 if getattr(core, "blocks", None) else self._target_w0_values()[1]
        return core.stem_w0, hidden

    def _reset_w0_schedule(self) -> None:
        lifecycle = self._architecture_lifecycle_
        core = self._wave_core()
        if lifecycle is not None and core is not None:
            lifecycle.warmup_step = 0
            lifecycle.warmup_active = bool(self.architecture.wave and self.architecture.wave.warmup)
            lifecycle._apply_warmup(core, 0)
        self._w0_schedule_active = bool(self.architecture.wave and self.architecture.wave.warmup)
        self._w0_schedule_step = 0

    def _update_w0_schedule(self, step: int) -> None:
        lifecycle = self._architecture_lifecycle_
        core = self._wave_core()
        if lifecycle is not None and core is not None:
            lifecycle.warmup_step = int(step)
            lifecycle._apply_warmup(core, int(step))
            epochs = (
                self.architecture.wave.warmup.epochs
                if self.architecture.wave and self.architecture.wave.warmup
                else 0
            )
            lifecycle.warmup_active = int(step) < int(epochs)
        self._w0_schedule_step = int(step)
        self._w0_schedule_active = bool(getattr(lifecycle, "warmup_active", False))

    def _reset_progressive_depth(self) -> None:
        lifecycle = self._architecture_lifecycle_
        core = self._wave_core()
        if lifecycle is not None and core is not None:
            lifecycle.current_depth = len(core.blocks)
        self._progressive_depth_current = (
            len(core.blocks) if core is not None else self.hidden_layers
        )


class SGRPSANNRegressor(_LegacyFacade):
    _legacy_name = "SGRPSANNRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        supplied = dict(kwargs)
        attention = None
        stateful = False
        state = None
        architecture = kwargs.pop("architecture", None)
        if architecture is not None:
            _discard_redundant_architecture_keywords(kwargs)
        if architecture is None:
            activation = _activation(kwargs)
            if activation.kind != "psann":
                raise ValueError("SGRPSANNRegressor requires activation_type='psann'.")
            attention = kwargs.pop("attention", None)
            stateful = kwargs.pop("stateful", False)
            state = kwargs.pop("state", None)
            for name in (
                "preserve_shape",
                "data_format",
                "conv_kernel_size",
                "conv_channels",
                "per_element",
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
        if attention is not None:
            self._compat_runtime_warning_ = (
                "SGRPSANNRegressor ignores attention; spectral gating uses the sequence axis."
            )
        self._capture_legacy_params(supplied)
        if stateful or state is not None:
            warnings.warn(
                "SGRPSANNRegressor does not support stateful configurations; ignoring state/stateful.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.stateful = False
            self.state = None
            self._legacy_params_["stateful"] = False
            self._legacy_params_["state"] = None
        if cast(Any, self).lsm is not None:
            warnings.warn(
                "SGRPSANNRegressor does not support LSM preprocessors; ignoring lsm settings.",
                RuntimeWarning,
                stacklevel=2,
            )
            cast(Any, self).lsm = None


class GeoSparseRegressor(_LegacyFacade):
    _legacy_name = "GeoSparseRegressor"

    def __init__(self, **kwargs: Any) -> None:
        self._warn()
        supplied = dict(kwargs)
        stateful = False
        state = None
        architecture = kwargs.pop("architecture", None)
        if architecture is not None:
            _discard_redundant_architecture_keywords(kwargs)
        if architecture is None:
            activation = _activation(kwargs)
            ignored_attention = kwargs.get("attention")
            if ignored_attention is not None:
                warnings.warn(
                    "GeoSparseRegressor ignores attention for now.", RuntimeWarning, stacklevel=2
                )
            stateful = kwargs.pop("stateful", False)
            state = kwargs.pop("state", None)
            for name in (
                "preserve_shape",
                "data_format",
                "conv_kernel_size",
                "conv_channels",
                "per_element",
                "attention",
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
        if stateful or state is not None:
            self._compat_runtime_warning_ = (
                "GeoSparseRegressor does not support stateful configurations; ignoring state_cfg."
            )
        self._capture_legacy_params(supplied)


__all__ = [
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "WaveResNetRegressor",
    "SGRPSANNRegressor",
    "GeoSparseRegressor",
]


ResPSANNRegressor._signature_source = _Phase2ResPSANNRegressor
ResConvPSANNRegressor._signature_source = _Phase2ResConvPSANNRegressor
WaveResNetRegressor._signature_source = _Phase2WaveResNetRegressor
SGRPSANNRegressor._signature_source = _Phase2SGRPSANNRegressor
GeoSparseRegressor._signature_source = _Phase2GeoSparseRegressor
for _facade in (
    ResPSANNRegressor,
    ResConvPSANNRegressor,
    WaveResNetRegressor,
    SGRPSANNRegressor,
    GeoSparseRegressor,
):
    setattr(
        cast(Any, _facade.__init__),
        "__signature__",
        signature(_facade._signature_source.__init__),
    )
