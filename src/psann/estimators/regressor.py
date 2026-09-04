"""The canonical registry-driven sklearn regressor.

The inherited mixins deliberately retain the Phase 2 training, scaling, HISSO, and
streaming implementation.  This class replaces only the former subclass-based model
selection with immutable architecture configuration and registry requests.
"""

# The retained 0.x flat adapter intentionally receives heterogeneous omitted
# sentinel values and normalizes them before construction; its boundary is dynamic.
# mypy cannot narrow that generated compatibility surface argument-by-argument.
# mypy: disable-error-code=arg-type

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from dataclasses import fields, replace
from inspect import Parameter, Signature, signature
from typing import Any, Mapping, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.nn as nn

from .._sklearn.base import PSANNRegressor as _Phase2Regressor
from ..architectures import (
    ArchitectureBuildRequest,
    ArchitectureConfig,
    ArchitectureLike,
    ActivationConfig,
    AttentionConfig,
    ConvolutionConfig,
    ContextConfig,
    GeometryConfig,
    ResidualConfig,
    ProgressiveDepthConfig,
    SequenceConfig,
    SpectralConfig,
    StateConfig,
    WaveConfig,
    W0WarmupConfig,
    build_architecture,
    architecture_to_mapping,
    normalize_architecture,
)
from ..architectures.config import _thaw, replace_architecture_paths, validate_architecture
from ..architectures.wrappers import _FlattenedConvModel
from ..attention import AttentionConfig as LegacyAttentionConfig
from ..state import StateConfig as LegacyStateConfig
from ..nn import WithPreprocessor
from ..preprocessing import (
    LSMConfig,
    ModulePreprocessorConfig,
    PreprocessorBuildRequest,
    PreprocessorCapabilities,
    PreprocessorConfig,
    PreprocessorLike,
    PreprocessorTrainingConfig,
    declared_preprocessor_capabilities,
    normalize_preprocessor,
    normalize_legacy_lsm,
    prepare_preprocessor,
    preprocessor_to_mapping,
    validate_preprocessor_capability,
)
from ..types import LossLike, ScalerSpec

_DEFAULT_ARCHITECTURE = ArchitectureConfig.dense()
_OMITTED = object()
_SCHEMA_MODULE_PREPROCESSOR_PATH = "Schema-v2 estimator_params.preprocessor"


def _normalise_schema_module_preprocessor(value: object, module: nn.Module) -> PreprocessorConfig:
    """Rebuild a custom preprocessor with strict, schema-path diagnostics."""

    path = _SCHEMA_MODULE_PREPROCESSOR_PATH
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    raw = dict(value)
    allowed = {"kind", "input_topology", "output_topology", "output_dim", "training"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{path}.{unknown[0]} is unknown.")
    for key in allowed:
        if key not in raw:
            raise ValueError(f"{path}.{key} is missing.")
    if raw["kind"] != "module":
        raise ValueError(f"{path}.kind must be module.")
    training_raw = raw["training"]
    if not isinstance(training_raw, Mapping):
        raise TypeError(f"{path}.training must be a mapping.")
    training_dict = dict(training_raw)
    training_allowed = {field.name for field in fields(PreprocessorTrainingConfig)}
    unknown_training = sorted(set(training_dict) - training_allowed)
    if unknown_training:
        raise ValueError(f"{path}.training.{unknown_training[0]} is unknown.")
    for key in training_allowed:
        if key not in training_dict:
            raise ValueError(f"{path}.training.{key} is missing.")
    try:
        training = PreprocessorTrainingConfig(**cast(Any, training_dict))
        return PreprocessorConfig(
            ModulePreprocessorConfig(
                module=module,
                input_topology=cast(str, raw["input_topology"]),
                output_topology=cast(str, raw["output_topology"]),
                output_dim=cast(int, raw["output_dim"]),
            ),
            training=training,
        )
    except (TypeError, ValueError) as exc:
        message = str(exc)
        message = message.replace("preprocessor.component.", f"{path}.")
        message = message.replace("preprocessor.training.", f"{path}.training.")
        raise type(exc)(message) from exc


def _schema_v2_preprocessor_with_artifact(value: object, artifacts: Mapping[str, object]) -> object:
    """Validate the schema-v2 custom-module discriminator/artifact pair.

    A module prototype establishes graph structure, while metadata establishes the
    portable configuration.  Neither is meaningful without the other, so validate
    their presence and discriminant before generic configuration normalization can
    route malformed payloads into unrelated topology errors.
    """

    artifact_key = "preprocessor_module"
    has_artifact = artifact_key in artifacts
    metadata_path = _SCHEMA_MODULE_PREPROCESSOR_PATH
    artifact_path = f"Schema-v2 artifacts.{artifact_key}"

    if value is None:
        if has_artifact:
            raise ValueError(f"{artifact_path} is unexpected without module metadata.")
        return value
    if not isinstance(value, Mapping):
        if has_artifact:
            raise TypeError(f"{metadata_path} must be a mapping.")
        return value

    kind = value.get("kind")
    if has_artifact:
        if "kind" not in value:
            raise ValueError(f"{metadata_path}.kind is missing.")
        if kind != "module":
            raise ValueError(
                f"{artifact_path} is unexpected for {metadata_path}.kind={kind!r}; "
                f"{metadata_path}.kind must be module."
            )
        module = artifacts[artifact_key]
        if not isinstance(module, nn.Module):
            raise ValueError(f"{artifact_path} is missing or invalid.")
        return _normalise_schema_module_preprocessor(value, module)

    if kind == "module":
        raise ValueError(f"{artifact_path} is missing for module metadata.")
    return value


def _activation_mapping(config: ArchitectureConfig) -> dict[str, object]:
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
    value = config.attention
    if value is None:
        return None
    return LegacyAttentionConfig(
        value.kind,
        value.num_heads,
        value.dropout,
        value.bias,
        value.batch_first,
        value.add_bias_kv,
        value.add_zero_attn,
    )


def _legacy_state(config: ArchitectureConfig) -> LegacyStateConfig | None:
    value = config.state
    if value is None:
        return None
    return LegacyStateConfig(value.rho, value.beta, value.init, value.max_abs, value.detach)


def _legacy_architecture(
    *,
    activation: object | None,
    activation_type: str,
    preserve_shape: bool,
    data_format: str,
    conv_kernel_size: int,
    conv_channels: int | None,
    per_element: bool,
    attention: object | None,
    stateful: bool,
    state: object | None,
    state_reset: str,
    stream_lr: float | None,
    context_builder: object | None,
    context_builder_params: object | None,
    w0_first: float,
    w0_hidden: float,
    norm: str,
    drop_path_max: float,
    residual_alpha_init: float,
    first_layer_w0: float,
    hidden_w0: float,
    dropout: float,
    grad_clip_norm: float | None,
    first_layer_w0_initial: float | None,
    hidden_w0_initial: float | None,
    w0_warmup_epochs: int,
    progressive_depth_initial: int | None,
    progressive_depth_interval: int,
    progressive_depth_growth: int,
    context_dim: int | None,
    use_film: bool,
    use_phase_shift: bool,
    use_spectral_gate: bool,
    k_fft: int,
    gate_type: str,
    gate_groups: str,
    gate_init: float,
    gate_strength: float,
    phase_init: float,
    phase_trainable: bool,
    pool: str,
    shape: tuple[int, int] | None,
    k: int,
    pattern: str,
    radius: int,
    offsets: object | None,
    wrap_mode: str,
    bias: bool,
    compute_mode: str,
    geo_seed: int | None,
) -> ArchitectureConfig:
    """Translate the retained 0.x flat constructor surface exactly once."""

    activation_values: dict[str, object] = (
        dict(activation) if isinstance(activation, Mapping) else {}
    )
    for legacy_key, canonical_key in {
        "amp_init": "amplitude_init",
        "freq_init": "frequency_init",
        "damping_init": "decay_init",
    }.items():
        if legacy_key in activation_values:
            if (
                canonical_key in activation_values
                and activation_values[canonical_key] != activation_values[legacy_key]
            ):
                raise ValueError(
                    f"activation has conflicting {legacy_key!r} and {canonical_key!r}."
                )
            activation_values[canonical_key] = activation_values.pop(legacy_key)
    activation_values.setdefault("kind", activation_type)
    canonical_activation = ActivationConfig(**cast(Any, activation_values))
    attention_value = None
    if attention is not None:
        if isinstance(attention, Mapping):
            raw = dict(attention)
            if str(raw.get("kind", "mha")).lower() not in {"none", "off", ""}:
                attention_value = AttentionConfig(**raw)
        elif getattr(attention, "is_enabled", lambda: bool(attention))():
            attention_value = AttentionConfig(
                **{field.name: getattr(attention, field.name) for field in fields(AttentionConfig)}
            )
    residual = (
        ResidualConfig(norm, residual_alpha_init, drop_path_max, w0_first, w0_hidden)
        if (
            norm != "rms"
            or w0_first != 12.0
            or w0_hidden != 1.0
            or drop_path_max != 0.0
            or residual_alpha_init != 0.0
        )
        else None
    )
    if shape is not None:
        return ArchitectureConfig.geometric_sparse(
            activation=canonical_activation,
            residual=residual or ResidualConfig(),
            geometry=GeometryConfig(  # type: ignore[arg-type]
                shape, k, pattern, radius, offsets, wrap_mode, bias, compute_mode, geo_seed
            ),
        )
    sequence_requested = any(
        value != default
        for value, default in (
            (phase_init, 0.0),
            (phase_trainable, True),
            (pool, "last"),
        )
    )
    if sequence_requested:
        if preserve_shape or attention_value is not None or residual is not None:
            raise ValueError(
                "sequence flat architecture inputs cannot combine with preserve_shape, attention, or residual inputs."
            )
        return ArchitectureConfig.for_sequence(
            activation=canonical_activation,
            sequence=SequenceConfig(phase_init, phase_trainable, pool),
            spectral=(
                SpectralConfig(k_fft, gate_type, gate_groups, gate_init, gate_strength)
                if use_spectral_gate
                else None
            ),
        )
    wave_requested = any(
        value != default
        for value, default in (
            (first_layer_w0, 30.0),
            (hidden_w0, 1.0),
            (dropout, 0.0),
            (grad_clip_norm, 5.0),
            (first_layer_w0_initial, 10.0),
            (hidden_w0_initial, 0.5),
            (w0_warmup_epochs, 10),
            (progressive_depth_initial, None),
            (context_dim, None),
            (context_builder, None),
            (use_spectral_gate, False),
        )
    )
    if wave_requested:
        warmup = (
            W0WarmupConfig(first_layer_w0_initial, hidden_w0_initial, w0_warmup_epochs)
            if first_layer_w0_initial is not None and hidden_w0_initial is not None
            else None
        )
        progressive = (
            ProgressiveDepthConfig(
                progressive_depth_initial, progressive_depth_interval, progressive_depth_growth
            )
            if progressive_depth_initial is not None
            else None
        )
        return ArchitectureConfig.for_wave(
            activation=canonical_activation,
            residual=residual or ResidualConfig(alpha_init=residual_alpha_init),
            wave=WaveConfig(
                first_layer_w0,
                hidden_w0,
                norm if norm != "rms" else "none",
                dropout,
                grad_clip_norm,
                warmup,
                progressive,
            ),
            attention=attention_value,
            spectral=(
                SpectralConfig(k_fft, gate_type, gate_groups, gate_init, gate_strength)
                if use_spectral_gate
                else None
            ),
            context=(
                ContextConfig(  # type: ignore[arg-type]
                    context_dim, context_builder, context_builder_params, use_film, use_phase_shift
                )
                if (context_dim is not None or context_builder is not None)
                else None
            ),
        )
    if preserve_shape:
        return ArchitectureConfig.convolutional(
            activation=canonical_activation,
            residual=residual,
            convolution=ConvolutionConfig(
                conv_channels, conv_kernel_size, data_format, per_element
            ),
            attention=attention_value,
        )
    state_value = None
    if stateful or state is not None:
        raw_state = (
            dict(state)
            if isinstance(state, Mapping)
            else {
                name: getattr(state, name)
                for name in ("rho", "beta", "init", "max_abs", "detach")
                if hasattr(state, name)
            }
        )
        raw_state.update({"reset": state_reset, "stream_lr": stream_lr})
        state_value = StateConfig(**raw_state)
    return ArchitectureConfig.dense(
        activation=canonical_activation,
        attention=attention_value,
        state=state_value,
        residual=residual,
    )


class PSANNRegressor(_Phase2Regressor):
    """One canonical estimator with registry-selected architecture builders."""

    def __init__(
        self,
        *,
        architecture: ArchitectureLike = _DEFAULT_ARCHITECTURE,
        hidden_layers: int = 2,
        hidden_units: int | None = None,
        epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        early_stopping: bool = False,
        patience: int = 20,
        num_workers: int = 0,
        warm_start: bool = False,
        loss: LossLike = "mse",
        loss_params: Optional[dict[str, Any]] = None,
        loss_reduction: str = "mean",
        scaler: Optional[ScalerSpec] = None,
        scaler_params: Optional[dict[str, Any]] = None,
        target_scaler: Optional[ScalerSpec] = None,
        target_scaler_params: Optional[dict[str, Any]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        device: str | torch.device = "auto",
        random_state: Optional[int] = None,
        amp: bool = False,
        amp_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        compile: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        compile_fullgraph: bool = False,
        compile_dynamic: bool = False,
        # Explicit 0.x compatibility inputs retained through this line.
        hidden_width: int | None = None,
        w0: float = 30.0,
        preprocessor: PreprocessorLike = None,
        lsm: object = _OMITTED,
        lsm_train: object = _OMITTED,
        lsm_pretrain_epochs: object = _OMITTED,
        lsm_lr: object = _OMITTED,
        activation: object = _OMITTED,
        activation_type: object = _OMITTED,
        preserve_shape: object = _OMITTED,
        data_format: object = _OMITTED,
        conv_kernel_size: object = _OMITTED,
        conv_channels: object = _OMITTED,
        per_element: object = _OMITTED,
        attention: object = _OMITTED,
        stateful: object = _OMITTED,
        state: object = _OMITTED,
        state_reset: object = _OMITTED,
        stream_lr: object = _OMITTED,
        context_builder: object = _OMITTED,
        context_builder_params: object = _OMITTED,
        w0_first: object = _OMITTED,
        w0_hidden: object = _OMITTED,
        norm: object = _OMITTED,
        drop_path_max: object = _OMITTED,
        residual_alpha_init: object = _OMITTED,
        first_layer_w0: object = _OMITTED,
        hidden_w0: object = _OMITTED,
        dropout: object = _OMITTED,
        grad_clip_norm: object = _OMITTED,
        first_layer_w0_initial: object = _OMITTED,
        hidden_w0_initial: object = _OMITTED,
        w0_warmup_epochs: object = _OMITTED,
        progressive_depth_initial: object = _OMITTED,
        progressive_depth_interval: object = _OMITTED,
        progressive_depth_growth: object = _OMITTED,
        context_dim: object = _OMITTED,
        use_film: object = _OMITTED,
        use_phase_shift: object = _OMITTED,
        use_spectral_gate: object = _OMITTED,
        k_fft: object = _OMITTED,
        gate_type: object = _OMITTED,
        gate_groups: object = _OMITTED,
        gate_init: object = _OMITTED,
        gate_strength: object = _OMITTED,
        phase_init: object = _OMITTED,
        phase_trainable: object = _OMITTED,
        pool: object = _OMITTED,
        shape: object = _OMITTED,
        k: object = _OMITTED,
        pattern: object = _OMITTED,
        radius: object = _OMITTED,
        offsets: object = _OMITTED,
        wrap_mode: object = _OMITTED,
        bias: object = _OMITTED,
        compute_mode: object = _OMITTED,
        geo_seed: object = _OMITTED,
    ) -> None:
        legacy_preprocessor = {
            name: value
            for name, value in {
                "lsm": lsm,
                "lsm_train": lsm_train,
                "lsm_pretrain_epochs": lsm_pretrain_epochs,
                "lsm_lr": lsm_lr,
            }.items()
            if value is not _OMITTED
        }
        if preprocessor is not None and legacy_preprocessor:
            raise ValueError(
                "preprocessor conflicts with legacy preprocessing keyword(s): "
                + ", ".join(legacy_preprocessor)
            )
        if legacy_preprocessor:
            warnings.warn(
                "lsm, lsm_train, lsm_pretrain_epochs, and lsm_lr are deprecated; "
                "use preprocessor=PreprocessorConfig(...).",
                DeprecationWarning,
                stacklevel=2,
            )
        legacy_lsm = legacy_preprocessor.get("lsm")
        legacy_lsm_train = bool(legacy_preprocessor.get("lsm_train", False))
        legacy_lsm_epochs = int(cast(int, legacy_preprocessor.get("lsm_pretrain_epochs", 0)))
        legacy_lsm_lr = legacy_preprocessor.get("lsm_lr")
        canonical_preprocessor = normalize_preprocessor(preprocessor)
        if canonical_preprocessor is None and "lsm" in legacy_preprocessor:
            adapted = normalize_legacy_lsm(
                legacy_lsm,
                trainable=legacy_lsm_train,
                pretrain_epochs=legacy_lsm_epochs,
                training_lr=cast(float | None, legacy_lsm_lr),
            )
            if adapted is not None:
                canonical_preprocessor = adapted
                legacy_lsm = None
        flat_supplied = {
            name: value
            for name, value in {
                "activation": activation,
                "activation_type": activation_type,
                "preserve_shape": preserve_shape,
                "data_format": data_format,
                "conv_kernel_size": conv_kernel_size,
                "conv_channels": conv_channels,
                "per_element": per_element,
                "attention": attention,
                "stateful": stateful,
                "state": state,
                "state_reset": state_reset,
                "stream_lr": stream_lr,
                "context_builder": context_builder,
                "context_builder_params": context_builder_params,
                "w0_first": w0_first,
                "w0_hidden": w0_hidden,
                "norm": norm,
                "drop_path_max": drop_path_max,
                "residual_alpha_init": residual_alpha_init,
                "first_layer_w0": first_layer_w0,
                "hidden_w0": hidden_w0,
                "dropout": dropout,
                "grad_clip_norm": grad_clip_norm,
                "first_layer_w0_initial": first_layer_w0_initial,
                "hidden_w0_initial": hidden_w0_initial,
                "w0_warmup_epochs": w0_warmup_epochs,
                "progressive_depth_initial": progressive_depth_initial,
                "progressive_depth_interval": progressive_depth_interval,
                "progressive_depth_growth": progressive_depth_growth,
                "context_dim": context_dim,
                "use_film": use_film,
                "use_phase_shift": use_phase_shift,
                "use_spectral_gate": use_spectral_gate,
                "k_fft": k_fft,
                "gate_type": gate_type,
                "gate_groups": gate_groups,
                "gate_init": gate_init,
                "gate_strength": gate_strength,
                "phase_init": phase_init,
                "phase_trainable": phase_trainable,
                "pool": pool,
                "shape": shape,
                "k": k,
                "pattern": pattern,
                "radius": radius,
                "offsets": offsets,
                "wrap_mode": wrap_mode,
                "bias": bias,
                "compute_mode": compute_mode,
                "geo_seed": geo_seed,
            }.items()
            if value is not _OMITTED
        }
        explicit_flat = bool(flat_supplied)
        defaults = {
            "activation": None,
            "activation_type": "psann",
            "preserve_shape": False,
            "data_format": "channels_first",
            "conv_kernel_size": 1,
            "conv_channels": None,
            "per_element": False,
            "attention": None,
            "stateful": False,
            "state": None,
            "state_reset": "batch",
            "stream_lr": None,
            "context_builder": None,
            "context_builder_params": None,
            "w0_first": 12.0,
            "w0_hidden": 1.0,
            "norm": "rms",
            "drop_path_max": 0.0,
            "residual_alpha_init": 0.0,
            "first_layer_w0": 30.0,
            "hidden_w0": 1.0,
            "dropout": 0.0,
            "grad_clip_norm": 5.0,
            "first_layer_w0_initial": 10.0,
            "hidden_w0_initial": 0.5,
            "w0_warmup_epochs": 10,
            "progressive_depth_initial": None,
            "progressive_depth_interval": 15,
            "progressive_depth_growth": 1,
            "context_dim": None,
            "use_film": True,
            "use_phase_shift": True,
            "use_spectral_gate": False,
            "k_fft": 64,
            "gate_type": "rfft",
            "gate_groups": "depthwise",
            "gate_init": 0.0,
            "gate_strength": 1.0,
            "phase_init": 0.0,
            "phase_trainable": True,
            "pool": "last",
            "shape": None,
            "k": 8,
            "pattern": "local",
            "radius": 1,
            "offsets": None,
            "wrap_mode": "clamp",
            "bias": True,
            "compute_mode": "gather",
            "geo_seed": None,
        }
        (
            activation,
            activation_type,
            preserve_shape,
            data_format,
            conv_kernel_size,
            conv_channels,
            per_element,
            attention,
            stateful,
            state,
            state_reset,
            stream_lr,
            context_builder,
            context_builder_params,
            w0_first,
            w0_hidden,
            norm,
            drop_path_max,
            residual_alpha_init,
            first_layer_w0,
            hidden_w0,
            dropout,
            grad_clip_norm,
            first_layer_w0_initial,
            hidden_w0_initial,
            w0_warmup_epochs,
            progressive_depth_initial,
            progressive_depth_interval,
            progressive_depth_growth,
            context_dim,
            use_film,
            use_phase_shift,
            use_spectral_gate,
            k_fft,
            gate_type,
            gate_groups,
            gate_init,
            gate_strength,
            phase_init,
            phase_trainable,
            pool,
            shape,
            k,
            pattern,
            radius,
            offsets,
            wrap_mode,
            bias,
            compute_mode,
            geo_seed,
        ) = cast(Any, [flat_supplied.get(name, default) for name, default in defaults.items()])
        if architecture is not _DEFAULT_ARCHITECTURE and explicit_flat:
            raise ValueError(
                "architecture conflicts with legacy architecture keyword(s): "
                + ", ".join(flat_supplied)
            )
        legacy_flat_adapter = architecture is _DEFAULT_ARCHITECTURE and explicit_flat
        if legacy_flat_adapter:
            warnings.warn(
                "Flat architecture keywords are deprecated; pass architecture=ArchitectureConfig(...).",
                DeprecationWarning,
                stacklevel=2,
            )
            architecture = _legacy_architecture(  # type: ignore[arg-type]
                activation=activation,
                activation_type=activation_type,
                preserve_shape=preserve_shape,
                data_format=data_format,
                conv_kernel_size=conv_kernel_size,
                conv_channels=conv_channels,
                per_element=per_element,
                attention=attention,
                stateful=stateful,
                state=state,
                state_reset=state_reset,
                stream_lr=stream_lr,
                context_builder=context_builder,
                context_builder_params=context_builder_params,
                w0_first=w0_first,
                w0_hidden=w0_hidden,
                norm=norm,
                drop_path_max=drop_path_max,
                residual_alpha_init=residual_alpha_init,
                first_layer_w0=first_layer_w0,
                hidden_w0=hidden_w0,
                dropout=dropout,
                grad_clip_norm=grad_clip_norm,
                first_layer_w0_initial=first_layer_w0_initial,
                hidden_w0_initial=hidden_w0_initial,
                w0_warmup_epochs=w0_warmup_epochs,
                progressive_depth_initial=progressive_depth_initial,
                progressive_depth_interval=progressive_depth_interval,
                progressive_depth_growth=progressive_depth_growth,
                context_dim=context_dim,
                use_film=use_film,
                use_phase_shift=use_phase_shift,
                use_spectral_gate=use_spectral_gate,
                k_fft=k_fft,
                gate_type=gate_type,
                gate_groups=gate_groups,
                gate_init=gate_init,
                gate_strength=gate_strength,
                phase_init=phase_init,
                phase_trainable=phase_trainable,
                pool=pool,
                shape=shape,
                k=k,
                pattern=pattern,
                radius=radius,
                offsets=offsets,
                wrap_mode=wrap_mode,
                bias=bias,
                compute_mode=compute_mode,
                geo_seed=geo_seed,
            )
        canonical = normalize_architecture(architecture)
        # ``super().__init__`` receives the resolved canonical value below, so
        # preserve the Phase 2 alias-only warning here rather than making the
        # inherited resolver believe both spellings were supplied.
        if hidden_width is not None and hidden_units is None:
            warnings.warn(
                "PSANNRegressor: `hidden_width` is deprecated; use `hidden_units` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        units = (
            hidden_width
            if hidden_units is None and hidden_width is not None
            else (64 if hidden_units is None else hidden_units)
        )
        validate_architecture(canonical, hidden_layers=int(hidden_layers))
        conv = canonical.convolution
        super().__init__(
            hidden_layers=hidden_layers,
            hidden_units=units,
            hidden_width=hidden_width,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            optimizer=optimizer,
            weight_decay=weight_decay,
            activation=_activation_mapping(canonical),  # type: ignore[arg-type]
            device=device,
            random_state=random_state,
            early_stopping=early_stopping,
            patience=patience,
            num_workers=num_workers,
            loss=loss,
            loss_params=loss_params,
            loss_reduction=loss_reduction,
            w0=w0,
            preserve_shape=conv is not None,
            data_format=conv.data_format if conv else data_format,
            conv_kernel_size=conv.kernel_size if conv else conv_kernel_size,
            conv_channels=conv.channels if conv and conv.channels is not None else conv_channels,
            per_element=conv.per_element if conv else False,
            activation_type=canonical.activation.kind,
            attention=_legacy_attention(canonical),
            stateful=canonical.state is not None,
            state=_legacy_state(canonical),
            state_reset=canonical.state.reset if canonical.state else state_reset,
            stream_lr=canonical.state.stream_lr if canonical.state else stream_lr,
            output_shape=output_shape,
            lsm=legacy_lsm,  # type: ignore[arg-type]
            lsm_train=legacy_lsm_train,
            lsm_pretrain_epochs=legacy_lsm_epochs,
            lsm_lr=legacy_lsm_lr,
            warm_start=warm_start,
            scaler=scaler,
            scaler_params=scaler_params,
            target_scaler=target_scaler,
            target_scaler_params=target_scaler_params,
            amp=amp,
            amp_dtype=amp_dtype,
            compile=compile,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
            compile_fullgraph=compile_fullgraph,
            compile_dynamic=compile_dynamic,
            context_builder=cast(
                Any, canonical.context.builder if canonical.context else context_builder
            ),
            context_builder_params=(
                _thaw(canonical.context.builder_params)
                if canonical.context and canonical.context.builder_params
                else context_builder_params  # type: ignore[arg-type]
            ),
        )
        self.architecture = canonical
        self.preprocessor = canonical_preprocessor
        if canonical_preprocessor is not None:
            # Retained fit hooks still consult these private compatibility values
            # while Phase 4 moves their construction ownership behind
            # ``preprocessor``. They are deliberately absent from canonical params.
            self.lsm_train = canonical_preprocessor.training.trainable
            self.lsm_lr = canonical_preprocessor.training.lr
        self.hidden_width = units
        # Flat ``preserve_shape`` retains its historical train/validation layout,
        # but its model is now a registry-built convolutional core wrapped only
        # at the input boundary.  This avoids advertising policies that a dense
        # fallback would silently discard.
        self._use_channel_first_train_inputs_ = (
            canonical.convolution is not None and not legacy_flat_adapter
        )
        self._legacy_flattened_preserve_shape_ = bool(
            legacy_flat_adapter and canonical.convolution is not None and not per_element
        )
        self._architecture_capabilities_: Any = None
        self._architecture_lifecycle_: Any = None
        self._architecture_structure_: dict[str, object] | None = None
        self._prepared_preprocessor_: nn.Module | None = None
        self.preprocessor_: nn.Module | None = None
        self.preprocessor_capabilities_: PreprocessorCapabilities | None = None
        self.preprocessor_diagnostics_: dict[str, object] | None = None
        self.preprocessor_controller_: object | None = None
        self._preprocessor_input_topology_: str | None = None
        self._preprocessor_input_shape_: tuple[int, ...] | None = None

    def _request(
        self,
        *,
        input_dim: int,
        output_dim: int,
        input_shape: tuple[int, ...],
        spatial_shape: tuple[int, ...] | None = None,
        spatial_ndim: int | None = None,
        in_channels: int | None = None,
    ) -> ArchitectureBuildRequest:
        seq_len = (
            int(np.prod(input_shape[:-1]))
            if self.architecture.kind == "sequence" and len(input_shape) > 1
            else 1
        )
        token_dim = (
            int(input_shape[-1])
            if self.architecture.kind == "sequence" and input_shape
            else int(input_dim)
        )
        capabilities = getattr(self, "preprocessor_capabilities_", None)
        effective_shape = tuple(input_shape)
        if (
            capabilities is not None
            and capabilities.output_topology == "tokens"
            and effective_shape
        ):
            effective_shape = effective_shape[:-1] + (capabilities.output_dim,)
            if self.architecture.kind == "sequence":
                token_dim = capabilities.output_dim
        return ArchitectureBuildRequest(
            self.architecture,
            self.hidden_layers,
            self.hidden_units,
            effective_shape,
            int(input_dim),
            int(output_dim),
            spatial_shape,
            spatial_ndim,
            in_channels,
            seq_len,
            token_dim,
            bool(self.per_element),
            self._device(),
            torch.float32,
            getattr(self, "_prepared_preprocessor_", None),
            capabilities.output_dim if capabilities is not None else None,
            getattr(self, "_architecture_structure_", None),
            self.w0,
        )

    def _receive_build(self, request: ArchitectureBuildRequest) -> nn.Module:
        result = build_architecture(request)
        self._architecture_capabilities_ = result.capabilities
        self._architecture_lifecycle_ = result.lifecycle
        if request.architecture.attention is not None:
            seq_len = (
                int(np.prod(request.spatial_shape))
                if request.spatial_shape is not None
                else int(np.prod(request.input_shape[:-1])) if len(request.input_shape) > 1 else 1
            )
            self._attention_shape_ = (seq_len, request.hidden_units)
        self._architecture_lifecycle_.on_model_built(model=result.model, runtime={})
        if request.preprocessor is not None:
            # Canonical builders own the one final composition.  Legacy hooks
            # recognize the returned wrapper and do not add another layer.
            if not isinstance(result.model, WithPreprocessor):
                raise RuntimeError(
                    "architecture builder did not compose the requested preprocessor."
                )
            return result.model
        return result.model

    def _resolve_lsm_module(
        self, data: object, *, preserve_shape: bool
    ) -> tuple[nn.Module | None, int | None]:
        """Prepare canonical preprocessing before architecture construction.

        The inherited method remains the isolated 0.x route for a supplied legacy
        ``lsm`` argument; canonical callers never enter that compatibility builder.
        """

        if self.preprocessor is None:
            return super()._resolve_lsm_module(data, preserve_shape=preserve_shape)
        array = np.asarray(data, dtype=np.float32)
        input_topology = {
            2: "flat",
            4: "spatial-2d",
            5: "spatial-3d",
        }.get(array.ndim)
        if array.ndim == 3:
            input_topology = "spatial-1d" if self.architecture.convolution is not None else "tokens"
        if input_topology is None:
            raise ValueError("preprocessor input must have a supported batch topology.")
        spatial_ndim = array.ndim - 2 if input_topology.startswith("spatial-") else None
        declared = declared_preprocessor_capabilities(self.preprocessor)
        geometry_size = None
        if self.architecture.geometry is not None and self.architecture.geometry.shape is not None:
            geometry_size = int(np.prod(self.architecture.geometry.shape))
        validate_preprocessor_capability(
            architecture_kind=self.architecture.kind,
            attention=self.architecture.attention is not None,
            convolutional=self.architecture.convolution is not None,
            spatial_ndim=spatial_ndim,
            capabilities=declared,
            geometry_size=geometry_size,
        )
        existing = getattr(self, "preprocessor_", None)
        current_shape = tuple(array.shape[1:])
        if (
            self.warm_start
            and isinstance(getattr(self, "model_", None), nn.Module)
            and isinstance(existing, nn.Module)
            and getattr(self, "_preprocessor_input_topology_", None) == input_topology
            and getattr(self, "_preprocessor_input_shape_", None) == current_shape
        ):
            self._prepared_preprocessor_ = existing
            return existing, declared.output_dim
        if self.warm_start and isinstance(getattr(self, "model_", None), nn.Module):
            self._clear_architecture_runtime()
        result = prepare_preprocessor(
            PreprocessorBuildRequest(
                self.preprocessor,
                input_topology,
                tuple(array.shape[1:]),
                array,
                self._device(),
                torch.float32,
            )
        )
        for parameter in result.module.parameters():
            parameter.requires_grad = self.preprocessor.training.trainable
        self._prepared_preprocessor_ = result.module
        self.preprocessor_ = result.module
        self.preprocessor_capabilities_ = result.capabilities
        self.preprocessor_diagnostics_ = dict(result.diagnostics)
        self.preprocessor_controller_ = result.controller
        self._preprocessor_input_topology_ = input_topology
        self._preprocessor_input_shape_ = current_shape
        self._lsm_module_ = result.module
        return result.module, result.capabilities.output_dim

    def _build_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[dict[str, Any]] = None,
        input_shape: Optional[tuple[int, ...]] = None,
    ) -> nn.Module:
        # Flat compatibility retains the old train/validation layout but routes
        # the actual model construction through the canonical convolution builder.
        if getattr(self, "_legacy_flattened_preserve_shape_", False):
            raw_shape = tuple(input_shape or (input_dim,))
            conv = self.architecture.convolution
            assert conv is not None
            internal = (
                (raw_shape[-1],) + raw_shape[:-1]
                if conv.data_format == "channels_last"
                else raw_shape
            )
            core = self._receive_build(
                self._request(
                    input_dim=int(np.prod(internal)),
                    output_dim=output_dim,
                    input_shape=internal,
                    spatial_shape=tuple(internal[1:]),
                    spatial_ndim=len(internal) - 1,
                    in_channels=int(internal[0]),
                )
            )
            return _FlattenedConvModel(core, input_shape=raw_shape, data_format=conv.data_format)
        return self._receive_build(
            self._request(
                input_dim=input_dim, output_dim=output_dim, input_shape=input_shape or (input_dim,)
            )
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray | None, *args: object, **kwargs: object
    ) -> "PSANNRegressor":
        warning = getattr(self, "_compat_runtime_warning_", None)
        if warning:
            warnings.warn(str(warning), RuntimeWarning, stacklevel=2)
        if getattr(self, "_compat_shaped_lsm_rejected_", False):
            raise ValueError(
                "WaveResNetRegressor does not support LSM preprocessors for preserve_shape inputs."
            )
        if self.architecture.kind == "wave" and (
            self.architecture.context is not None
            or getattr(self, "_legacy_context_requested_", False)
        ):
            context = kwargs.get("context")
            policy = self.architecture.context
            if policy is None and context is not None:
                array = np.asarray(context)
                dim = int(array.reshape(array.shape[0], -1).shape[1]) if array.ndim > 1 else 1
                policy = ContextConfig(dim=dim)
                self.architecture = replace(self.architecture, context=policy)
            assert policy is not None
            if context is None and (policy is None or policy.builder is None):
                raise ValueError(
                    f"WaveResNetRegressor expects a context array matching context_dim={policy.dim}; received context=None."
                )
            if context is not None and policy.dim is None:
                array = np.asarray(context)
                dim = int(array.reshape(array.shape[0], -1).shape[1]) if array.ndim > 1 else 1
                self.architecture = replace(self.architecture, context=replace(policy, dim=dim))
            elif context is None and policy.builder is not None and policy.dim is None:
                builder = self._get_context_builder()
                if builder is not None:
                    inferred = np.asarray(builder(np.asarray(X, dtype=np.float32)))
                    dim = int(inferred.reshape(inferred.shape[0], -1).shape[1])
                    self.architecture = replace(self.architecture, context=replace(policy, dim=dim))
        result = super().fit(X, y, *args, **kwargs)  # type: ignore[arg-type]
        if self._architecture_lifecycle_ is not None and self.model_ is not None:
            self._architecture_lifecycle_.on_fit_end(model=self.model_, runtime={})
        return result

    def score_reconstruction(self, X: np.ndarray) -> float:
        """Score the fitted canonical LSM reconstruction without mutating its module."""

        controller = getattr(self, "preprocessor_controller_", None)
        scorer = getattr(controller, "score_reconstruction", None)
        if not callable(scorer):
            raise RuntimeError("The fitted preprocessor has no reconstruction scoring controller.")
        prepared, _, _ = self._prepare_inference_inputs(X)
        return float(scorer(prepared))

    def _after_model_built(self) -> None:
        super()._after_model_built()
        if self._architecture_lifecycle_ is not None and self.model_ is not None:
            self._architecture_lifecycle_.on_fit_start(
                model=self.model_, optimizer=getattr(self, "_optimizer_", None), runtime={}
            )

    def _build_conv_core(
        self,
        spatial_ndim: int,
        in_channels: int,
        output_dim: int,
        *,
        segmentation_head: bool,
        spatial_shape: Optional[tuple[int, ...]] = None,
    ) -> nn.Module:
        shape = tuple(spatial_shape or ())
        return self._receive_build(
            self._request(
                input_dim=int(in_channels * (np.prod(shape) if shape else 1)),
                output_dim=output_dim,
                input_shape=(in_channels,) + shape,
                spatial_shape=shape,
                spatial_ndim=spatial_ndim,
                in_channels=in_channels,
            )
        )

    def gradient_hook(self, model: nn.Module) -> None:
        if self._architecture_lifecycle_ is not None and self._optimizer_ is not None:
            self._architecture_lifecycle_.before_optimizer_step(
                model=model, optimizer=self._optimizer_, runtime={}
            )

    def epoch_callback(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        improved: bool,
        patience_left: Optional[int],
    ) -> None:
        if self._architecture_lifecycle_ is not None:
            self._architecture_lifecycle_.on_epoch_end(
                model=self.model_,
                optimizer=self._optimizer_,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss if val_loss is not None else float("nan"),
                },
                runtime={},
            )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        names = (
            "architecture",
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
            "preprocessor",
        )
        params = {name: getattr(self, name) for name in names}
        if deep:
            if self.preprocessor is not None:
                params["preprocessor__component"] = self.preprocessor.component
                params["preprocessor__training"] = self.preprocessor.training
                for field in fields(self.preprocessor.training):
                    params[f"preprocessor__training__{field.name}"] = getattr(
                        self.preprocessor.training, field.name
                    )
                component = self.preprocessor.component
                for field in fields(component):
                    params[f"preprocessor__component__{field.name}"] = getattr(
                        component, field.name
                    )
                if hasattr(component, "pretraining"):
                    for field in fields(component.pretraining):
                        params[f"preprocessor__component__pretraining__{field.name}"] = getattr(
                            component.pretraining, field.name
                        )
            # ``conv_channels`` was historically visible from the public
            # deep parameter mapping.  Keeping it here (rather than in the
            # shallow constructor map) preserves that inspection surface
            # without feeding a flat architecture mirror back to sklearn
            # clone alongside ``architecture``.
            params["conv_channels"] = self.conv_channels
            for policy in (
                "activation",
                "residual",
                "convolution",
                "attention",
                "state",
                "context",
                "wave",
                "spectral",
                "sequence",
                "geometry",
            ):
                value = getattr(self.architecture, policy)
                params[f"architecture__{policy}"] = value
                if value is not None:
                    for field in fields(value):
                        params[f"architecture__{policy}__{field.name}"] = getattr(value, field.name)
        return params

    def _clear_architecture_runtime(self) -> None:
        for name in (
            "model_",
            "history_",
            "_optimizer_",
            "_lr_scheduler_",
            "_amp_scaler_",
            "_stream_opt_",
            "_stream_loss_",
            "_stream_model_token_",
            "_context_builder_callable_",
            "_hisso_cache_",
            "_hisso_trainer_",
            "_hisso_warmstart_optimizer_",
            "_architecture_lifecycle_",
            "_architecture_capabilities_",
            "_architecture_structure_",
            "_prepared_preprocessor_",
            "preprocessor_",
            "preprocessor_capabilities_",
            "preprocessor_diagnostics_",
            "preprocessor_controller_",
            "_preprocessor_input_topology_",
            "_preprocessor_input_shape_",
        ):
            self.__dict__.pop(name, None)

    def set_params(self, **params: object) -> "PSANNRegressor":
        if not params:
            return self
        # Keep the established sklearn aliases and the context-builder cache
        # semantics before applying canonical architecture updates.  The
        # context builder is fit plumbing retained until Phase 4; it is not a
        # dense-architecture policy.
        original = dict(params)
        normalised: dict[str, Any] = self._normalize_param_aliases(dict(params))
        reset_context_builder = (
            "context_builder" in normalised or "context_builder_params" in normalised
        )
        if "context_builder_params" in normalised:
            constructor_value = original.get("context_builder_params")
            self._context_builder_params_constructor_ = cast(Any, constructor_value)
            normalised["context_builder_params"] = deepcopy(
                {} if constructor_value is None else constructor_value
            )
        params = normalised
        if "architecture" in params and any(key.startswith("architecture__") for key in params):
            raise ValueError(
                "architecture and architecture__ parameters cannot be updated together."
            )
        if "preprocessor" in params and any(key.startswith("preprocessor__") for key in params):
            raise ValueError(
                "preprocessor and preprocessor__ parameters cannot be updated together."
            )
        candidate = self.architecture
        if "architecture" in params:
            candidate = normalize_architecture(cast(ArchitectureLike, params.pop("architecture")))
        nested = {key: params.pop(key) for key in list(params) if key.startswith("architecture__")}
        if nested:
            candidate = replace_architecture_paths(
                candidate,
                nested,
                hidden_layers=int(cast(Any, params.get("hidden_layers", self.hidden_layers))),
            )
        preprocessor_candidate = self.preprocessor
        if "preprocessor" in params:
            preprocessor_candidate = normalize_preprocessor(
                cast(PreprocessorLike, params.pop("preprocessor"))
            )
        preprocessor_nested = {
            key: params.pop(key) for key in list(params) if key.startswith("preprocessor__")
        }
        if preprocessor_nested:
            if preprocessor_candidate is None:
                raise ValueError("preprocessor__ updates require a configured preprocessor.")
            component_changes: dict[str, object] = {}
            training_changes: dict[str, object] = {}
            pretraining_changes: dict[str, object] = {}
            replacement_component: LSMConfig | ModulePreprocessorConfig | None = None
            for key, value in preprocessor_nested.items():
                path = key.split("__")[1:]
                if path == ["component"]:
                    if not isinstance(value, (LSMConfig, ModulePreprocessorConfig)):
                        raise TypeError("preprocessor__component must be a component config.")
                    replacement_component = value
                elif path == ["training"]:
                    if not isinstance(value, PreprocessorTrainingConfig):
                        raise TypeError(
                            "preprocessor__training must be a PreprocessorTrainingConfig."
                        )
                    training_changes = {
                        field.name: getattr(value, field.name) for field in fields(value)
                    }
                elif path[:2] == ["component", "pretraining"] and len(path) == 3:
                    pretraining_changes[path[2]] = value
                elif path[:1] == ["component"] and len(path) == 2:
                    component_changes[path[1]] = value
                elif path[:1] == ["training"] and len(path) == 2:
                    training_changes[path[1]] = value
                else:
                    raise ValueError(f"Invalid parameter {key!r} for PSANNRegressor.")
            component = replacement_component or preprocessor_candidate.component
            if pretraining_changes:
                if not hasattr(component, "pretraining"):
                    raise ValueError(
                        "preprocessor.component does not support pretraining parameters."
                    )
                component_changes["pretraining"] = replace(
                    component.pretraining, **cast(Any, pretraining_changes)
                )
            component = replace(component, **cast(Any, component_changes))
            training = replace(preprocessor_candidate.training, **cast(Any, training_changes))
            preprocessor_candidate = PreprocessorConfig(component=component, training=training)
        validate_architecture(
            candidate, hidden_layers=int(cast(Any, params.get("hidden_layers", self.hidden_layers)))
        )
        valid = set(self.get_params(deep=False)) | {
            "conv_channels",
            "context_builder",
            "context_builder_params",
        }
        unknown = set(params) - valid
        if unknown:
            raise ValueError(f"Invalid parameter {sorted(unknown)[0]!r} for PSANNRegressor.")
        changed_architecture = candidate != self.architecture
        changed_preprocessor = preprocessor_candidate != self.preprocessor
        for key, value in params.items():
            setattr(self, key, value)
        if reset_context_builder:
            self._context_builder_callable_ = None
            if self.context_builder is None and "context_dim" not in params:
                self._context_dim_ = None
                if hasattr(self, "context_dim"):
                    self.context_dim = None
        if changed_architecture:
            self.architecture = candidate
            conv = candidate.convolution
            self.preserve_shape = conv is not None
            self.per_element = bool(conv and conv.per_element)
            self._use_channel_first_train_inputs_ = conv is not None
            if conv is not None:
                self.data_format = conv.data_format
                self.conv_kernel_size = conv.kernel_size
                self.conv_channels = conv.channels or self.hidden_units
            self.activation = cast(Any, _activation_mapping(candidate))
            self.activation_type = candidate.activation.kind
            self.attention = cast(Any, _legacy_attention(candidate))
            self.state = _legacy_state(candidate)
            self.stateful = candidate.state is not None
            self._clear_architecture_runtime()
        if changed_preprocessor:
            self.preprocessor = preprocessor_candidate
            self._clear_architecture_runtime()
        return self

    def save(self, path: str) -> None:
        """Write a portable schema-v2 payload without serialising a final model."""

        self._ensure_fitted()
        model = self.model_
        if model is None:
            raise RuntimeError("Estimator is not fitted.")
        state_dict = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        # Facades retain legacy ``get_params`` for callers, but the canonical
        # checkpoint always stores the one canonical constructor surface.
        params = PSANNRegressor.get_params(self, deep=False)
        params["architecture"] = architecture_to_mapping(self.architecture)
        artifacts: dict[str, object] = {
            "hisso_reward_fn": getattr(self, "_hisso_reward_fn_", None),
        }
        configured_preprocessor = self.preprocessor
        attached_preprocessor = getattr(model, "preproc", None)
        if configured_preprocessor is not None and isinstance(
            configured_preprocessor.component, ModulePreprocessorConfig
        ):
            # The portable constructor metadata describes the boundary while
            # the prototype reconstructs user-defined module structure.  The
            # state dictionary below remains authoritative for its weights.
            component = configured_preprocessor.component
            params["preprocessor"] = {
                "kind": "module",
                "input_topology": component.input_topology,
                "output_topology": component.output_topology,
                "output_dim": component.output_dim,
                "training": {
                    "trainable": configured_preprocessor.training.trainable,
                    "lr": configured_preprocessor.training.lr,
                },
            }
            artifacts["preprocessor_module"] = deepcopy(component.module).cpu()
        elif configured_preprocessor is not None:
            params["preprocessor"] = preprocessor_to_mapping(configured_preprocessor)
        elif attached_preprocessor is not None:
            # Old ``lsm=`` objects remain a 0.x adapter.  Persist the actual
            # prepared module as a module-boundary artifact so that a new v2
            # save does not discard its graph structure or fitted weights.
            input_topology = (
                "spatial-2d" if hasattr(attached_preprocessor, "out_channels") else "flat"
            )
            output_dim = int(
                getattr(
                    attached_preprocessor,
                    "out_channels",
                    getattr(attached_preprocessor, "output_dim", 0),
                )
            )
            if output_dim <= 0:
                raise TypeError("legacy preprocessor module has no declared output width.")
            params["preprocessor"] = {
                "kind": "module",
                "input_topology": input_topology,
                "output_topology": input_topology,
                "output_dim": output_dim,
                "training": {
                    "trainable": bool(getattr(self, "lsm_train", False)),
                    "lr": getattr(self, "lsm_lr", None),
                },
            }
            artifacts["preprocessor_module"] = deepcopy(attached_preprocessor).cpu()
        fitted = {
            "input_shape": (
                tuple(self.input_shape_)
                if getattr(self, "input_shape_", None) is not None
                else None
            ),
            "internal_shape_cf": (
                tuple(self._internal_input_shape_cf_)
                if getattr(self, "_internal_input_shape_cf_", None) is not None
                else None
            ),
            "primary_dim": self._primary_dim_,
            "output_dim": self._output_dim_,
            "keep_column_output": bool(getattr(self, "_keep_column_output_", False)),
            "train_layout": self._train_inputs_layout_,
            "target_cf_shape": self._target_cf_shape_,
            "target_vector_dim": self._target_vector_dim_,
            "output_shape_tuple": self._output_shape_tuple_,
            "context_dim": self._context_dim_,
            "scaler_kind": self._scaler_kind_,
            "scaler_state": self._scaler_state_,
            "target_scaler_kind": self._target_scaler_kind_,
            "target_scaler_state": self._target_scaler_state_,
            "hisso_cfg": self._hisso_cfg_,
            "hisso_options": self._hisso_options_,
            "hisso_trained": self._hisso_trained_,
            "legacy_flattened_preserve_shape": getattr(
                self, "_legacy_flattened_preserve_shape_", False
            ),
        }
        capabilities = getattr(self, "preprocessor_capabilities_", None)
        if capabilities is not None:
            fitted["preprocessing"] = {
                "input_topology": capabilities.input_topology,
                "output_topology": capabilities.output_topology,
                "output_dim": capabilities.output_dim,
                "diagnostics": getattr(self, "preprocessor_diagnostics_", None) or {},
            }
        elif attached_preprocessor is not None:
            metadata = cast(Mapping[str, object], params["preprocessor"])
            fitted["preprocessing"] = {
                "input_topology": metadata["input_topology"],
                "output_topology": metadata["output_topology"],
                "output_dim": metadata["output_dim"],
                "diagnostics": {},
            }
        # A migrated checkpoint can carry a reconstruction discriminator which
        # the generic lifecycle does not own (for example the retained legacy
        # Wave attention wrapper).  Lifecycle counters are authoritative when
        # present, but must not discard those durable structure fields on a
        # later ordinary v1 save.
        structure = dict(getattr(self, "_architecture_structure_", {}) or {})
        if self._architecture_lifecycle_:
            structure.update(self._architecture_lifecycle_.structure_metadata())
        payload = {
            "schema": "psann.regressor",
            "schema_version": 2,
            "package_version": "0.12.4",
            "estimator_params": params,
            "fitted": fitted,
            "structure": structure,
            "model_state_dict": state_dict,
            "artifacts": artifacts,
        }
        try:
            torch.save(payload, path)
        except (AttributeError, TypeError) as exc:
            if "preprocessor_module" in artifacts:
                raise TypeError(
                    "Schema-v2 artifacts.preprocessor_module could not be serialized; "
                    "use an importable torch.nn.Module class."
                ) from exc
            raise

    @classmethod
    def load(
        cls, path: str, *, map_location: Optional[Union[str, torch.device]] = "cpu"
    ) -> "PSANNRegressor":
        # Historical pickles name ``psann.lsm`` classes. Their public import
        # path warns for users, but checkpoint deserialization must be silent.
        from .. import lsm as legacy_lsm_module

        legacy_lsm_module._set_deserialization_warning_suppressed(True)
        try:
            try:
                payload = torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location=map_location)
        finally:
            legacy_lsm_module._set_deserialization_warning_suppressed(False)
        if payload.get("schema") != "psann.regressor":
            # Preserve the strict Phase-2 reader, then attach its deserialised
            # module to a configuration-bearing canonical instance.  It remains a
            # migration bridge only: the next save writes a state-dict-only v1 file.
            from .._sklearn.geosparse import GeoSparseRegressor as LegacyGeo
            from .._sklearn.residual import (
                ResConvPSANNRegressor as LegacyResConv,
                ResPSANNRegressor as LegacyRes,
            )
            from .._sklearn.sgr import SGRPSANNRegressor as LegacySGR
            from .._sklearn.wave import WaveResNetRegressor as LegacyWave

            old_classes = {
                "PSANNRegressor": _Phase2Regressor,
                "ResPSANNRegressor": LegacyRes,
                "ResConvPSANNRegressor": LegacyResConv,
                "WaveResNetRegressor": LegacyWave,
                "SGRPSANNRegressor": LegacySGR,
                "GeoSparseRegressor": LegacyGeo,
            }
            old_name = payload.get("class")
            reader: Any = old_classes.get(old_name)
            if reader is None:
                raise ValueError(f"Unsupported unversioned estimator class {old_name!r}.")
            legacy = reader.load(path, map_location=map_location)
            activation: Any = normalize_architecture("dense").activation
            raw_activation = getattr(legacy, "activation", None)
            if isinstance(raw_activation, Mapping):
                activation = __import__(
                    "psann.architectures", fromlist=["ActivationConfig"]
                ).ActivationConfig(**raw_activation)
            if old_name in {"ResPSANNRegressor", "ResConvPSANNRegressor"}:
                residual = ResidualConfig(
                    getattr(legacy, "norm", "rms"),
                    getattr(legacy, "residual_alpha_init", 0.0),
                    getattr(legacy, "drop_path_max", 0.0),
                    getattr(legacy, "w0_first", 12.0),
                    getattr(legacy, "w0_hidden", 1.0),
                )
                if old_name == "ResConvPSANNRegressor" or getattr(legacy, "preserve_shape", False):
                    architecture = ArchitectureConfig.convolutional(
                        activation=activation,
                        residual=residual,
                        convolution=ConvolutionConfig(
                            getattr(legacy, "conv_channels", None),
                            getattr(legacy, "conv_kernel_size", 1),
                            getattr(legacy, "data_format", "channels_first"),
                            getattr(legacy, "per_element", False),
                        ),
                    )
                else:
                    architecture = ArchitectureConfig.dense(
                        activation=activation, residual=residual
                    )
            elif old_name == "WaveResNetRegressor":
                legacy_wave_model: Any = legacy.model_
                while hasattr(legacy_wave_model, "core"):
                    legacy_wave_model = legacy_wave_model.core
                # The 0.x shaped Wave options were accepted before all fit
                # routes used channel-first hooks.  Migrate what the retained
                # module actually executed, not merely an option that its old
                # fit route ignored; the latter cannot be reconstructed from a
                # state dict with a different topology.
                effective_convolution = hasattr(legacy_wave_model, "conv_core")
                effective_spectral = hasattr(legacy_wave_model, "spectral")
                effective_attention = hasattr(legacy_wave_model, "attention")
                raw_attention = getattr(legacy, "attention", None)
                migrated_attention = None
                if (
                    effective_attention
                    and raw_attention is not None
                    and getattr(raw_attention, "is_enabled", lambda: bool(raw_attention))()
                ):
                    migrated_attention = AttentionConfig(
                        **{
                            field.name: getattr(raw_attention, field.name)
                            for field in fields(AttentionConfig)
                            if hasattr(raw_attention, field.name)
                        }
                    )
                warmup = W0WarmupConfig(
                    getattr(legacy, "first_layer_w0_initial", 10.0),
                    getattr(legacy, "hidden_w0_initial", 0.5),
                    getattr(legacy, "w0_warmup_epochs", 10),
                )
                progressive_initial = getattr(legacy, "progressive_depth_initial", None)
                architecture = ArchitectureConfig.for_wave(
                    activation=activation,
                    residual=ResidualConfig(alpha_init=getattr(legacy, "residual_alpha_init", 0.0)),
                    wave=WaveConfig(
                        getattr(legacy, "first_layer_w0", 30.0),
                        getattr(legacy, "hidden_w0", 1.0),
                        getattr(legacy, "norm", "none"),
                        getattr(legacy, "dropout", 0.0),
                        getattr(legacy, "grad_clip_norm", 5.0),
                        warmup,
                        (
                            ProgressiveDepthConfig(
                                progressive_initial,
                                getattr(legacy, "progressive_depth_interval", 15),
                                getattr(legacy, "progressive_depth_growth", 1),
                            )
                            if progressive_initial is not None
                            else None
                        ),
                    ),
                    convolution=(
                        ConvolutionConfig(
                            getattr(legacy, "conv_channels", None),
                            getattr(legacy, "conv_kernel_size", 1),
                            getattr(legacy, "data_format", "channels_first"),
                            False,
                        )
                        if effective_convolution
                        else None
                    ),
                    # The legacy flat Wave builder explicitly ignored attention
                    # whenever spectral gating was enabled; preserve that effective
                    # behavior rather than creating an invalid canonical pair.
                    attention=(None if effective_spectral else migrated_attention),
                    context=(
                        ContextConfig(
                            getattr(legacy, "context_dim", None),
                            getattr(legacy, "context_builder", None),
                            getattr(legacy, "context_builder_params", None),
                            getattr(legacy, "use_film", True),
                            getattr(legacy, "use_phase_shift", True),
                        )
                        if getattr(legacy, "context_dim", None) is not None
                        or getattr(legacy, "context_builder", None) is not None
                        else None
                    ),
                    spectral=(
                        SpectralConfig(
                            getattr(legacy, "k_fft", 64),
                            getattr(legacy, "gate_type", "rfft"),
                            getattr(legacy, "gate_groups", "depthwise"),
                            getattr(legacy, "gate_init", 0.0),
                            getattr(legacy, "gate_strength", 1.0),
                        )
                        if effective_spectral
                        else None
                    ),
                )
            elif old_name == "SGRPSANNRegressor":
                architecture = ArchitectureConfig.for_sequence(
                    activation=activation,
                    sequence=SequenceConfig(
                        getattr(legacy, "phase_init", 0.0),
                        getattr(legacy, "phase_trainable", True),
                        getattr(legacy, "pool", "last"),
                    ),
                    spectral=(
                        SpectralConfig(
                            getattr(legacy, "k_fft", 64),
                            getattr(legacy, "gate_type", "rfft"),
                            getattr(legacy, "gate_groups", "depthwise"),
                            getattr(legacy, "gate_init", 0.0),
                            getattr(legacy, "gate_strength", 1.0),
                        )
                        if getattr(legacy, "use_spectral_gate", True)
                        else None
                    ),
                )
            elif old_name == "GeoSparseRegressor":
                architecture = ArchitectureConfig.geometric_sparse(
                    activation=activation,
                    residual=ResidualConfig(
                        getattr(legacy, "norm", "rms"),
                        getattr(legacy, "residual_alpha_init", 0.0),
                        getattr(legacy, "drop_path_max", 0.0),
                    ),
                    geometry=GeometryConfig(
                        getattr(legacy, "shape", None),
                        getattr(legacy, "k", 8),
                        getattr(legacy, "pattern", "local"),
                        getattr(legacy, "radius", 1),
                        getattr(legacy, "offsets", None),
                        getattr(legacy, "wrap_mode", "clamp"),
                        getattr(legacy, "bias", True),
                        getattr(legacy, "compute_mode", "gather"),
                        getattr(legacy, "geo_seed", None),
                    ),
                )
            elif getattr(legacy, "preserve_shape", False):
                architecture = ArchitectureConfig.convolutional(
                    activation=activation,
                    convolution=ConvolutionConfig(
                        getattr(legacy, "conv_channels", None),
                        getattr(legacy, "conv_kernel_size", 1),
                        getattr(legacy, "data_format", "channels_first"),
                        getattr(legacy, "per_element", False),
                    ),
                )
            else:
                architecture = ArchitectureConfig.dense(activation=activation)
            migrated = cls(
                architecture=architecture,
                hidden_layers=getattr(legacy, "hidden_layers", 2),
                hidden_units=getattr(legacy, "hidden_units", 64),
                epochs=getattr(legacy, "epochs", 200),
                batch_size=getattr(legacy, "batch_size", 128),
                lr=getattr(legacy, "lr", 1e-3),
                optimizer=getattr(legacy, "optimizer", "adam"),
                weight_decay=getattr(legacy, "weight_decay", 0.0),
                device=getattr(legacy, "device", "auto"),
                random_state=getattr(legacy, "random_state", None),
            )
            migrated.model_ = legacy.model_
            if old_name == "WaveResNetRegressor":
                # Legacy pickle payloads retain the module's *effective* W0
                # values but omit the schedule counters.  The schema-v1 state
                # dict deliberately contains no mutable W0 attributes, so
                # recover the counter from the retained module before a v1
                # re-save.  That lets the canonical lifecycle rebuild the same
                # values after a strict state-dict load.
                legacy_core: Any = migrated.model_
                while hasattr(legacy_core, "core"):
                    legacy_core = legacy_core.core
                legacy_core = getattr(legacy_core, "wave", legacy_core)
                warmup_step = int(getattr(legacy, "_w0_schedule_step", 0))
                wave = architecture.wave
                if wave is not None and wave.warmup is not None:
                    ratios: list[float] = []
                    effective = (
                        getattr(legacy_core, "stem_w0", None),
                        (
                            getattr(legacy_core.blocks[0], "w0", None)
                            if getattr(legacy_core, "blocks", None)
                            else None
                        ),
                    )
                    endpoints = (
                        (wave.warmup.first_initial, wave.first_w0),
                        (wave.warmup.hidden_initial, wave.hidden_w0),
                    )
                    for observed, (initial, endpoint) in zip(effective, endpoints):
                        if observed is not None and not math.isclose(initial, endpoint):
                            ratios.append((float(observed) - initial) / (endpoint - initial))
                    if ratios:
                        warmup_step = max(
                            0,
                            min(
                                wave.warmup.epochs,
                                round(sum(ratios) / len(ratios) * wave.warmup.epochs),
                            ),
                        )
                migrated._architecture_structure_ = {
                    "current_depth": int(
                        len(getattr(legacy_core, "blocks", ()))
                        or getattr(
                            legacy,
                            "_progressive_depth_current",
                            getattr(legacy, "hidden_layers", 1),
                        )
                    ),
                    "warmup_step": warmup_step,
                    "warmup_active": bool(
                        wave is not None
                        and wave.warmup is not None
                        and warmup_step < wave.warmup.epochs
                    ),
                    "next_expand_epoch": getattr(legacy, "_progressive_next_expand_epoch", None),
                    # Legacy flat attention used the generic token wrapper around
                    # a Wave token backbone.  Canonical Wave attention has a
                    # different (and intentional) composition, so retain this
                    # migration-only structural discriminator for strict v1 load.
                    "legacy_attention_wrapper": bool(
                        migrated.architecture.attention is not None
                        and hasattr(migrated.model_, "token_backbone")
                        and hasattr(migrated.model_, "readout")
                    ),
                }
            for name in (
                "input_shape_",
                "_internal_input_shape_cf_",
                "_primary_dim_",
                "_output_dim_",
                "_keep_column_output_",
                "_train_inputs_layout_",
                "_target_cf_shape_",
                "_target_vector_dim_",
                "_output_shape_tuple_",
                "_context_dim_",
                "_scaler_kind_",
                "_scaler_state_",
                "_target_scaler_kind_",
                "_target_scaler_state_",
            ):
                if hasattr(legacy, name):
                    setattr(migrated, name, getattr(legacy, name))
            # Migration deliberately retains the historical fitted module so
            # predictions remain bit-compatible until the caller saves a v1
            # checkpoint.  Still derive the canonical capabilities now: they
            # are part of the new object's public architecture contract and do
            # not require replacing the retained module.
            migrated_shape = tuple(getattr(migrated, "input_shape_", ()) or ())
            migrated_output = int(getattr(migrated, "_output_dim_", 1) or 1)
            if migrated.architecture.convolution is not None:
                internal = tuple(
                    getattr(migrated, "_internal_input_shape_cf_", ()) or migrated_shape
                )
                if not internal:
                    raise ValueError("Unversioned convolutional checkpoint is missing input shape.")
                capability_request = migrated._request(
                    input_dim=int(np.prod(internal)),
                    output_dim=migrated_output,
                    input_shape=internal,
                    spatial_shape=tuple(internal[1:]),
                    spatial_ndim=len(internal) - 1,
                    in_channels=int(internal[0]),
                )
            else:
                if not migrated_shape:
                    raise ValueError("Unversioned checkpoint is missing input shape.")
                capability_request = migrated._request(
                    input_dim=int(np.prod(migrated_shape)),
                    output_dim=migrated_output,
                    input_shape=migrated_shape,
                )
            migrated._architecture_capabilities_ = build_architecture(
                capability_request
            ).capabilities
            return migrated
        version = payload.get("schema_version")
        if version not in {1, 2}:
            raise ValueError(f"Unsupported psann.regressor schema version {version!r}.")
        raw_params = dict(payload.get("estimator_params", {}))
        # Device selection is a reconstruction input, not a post-load cosmetic
        # update.  Otherwise a CUDA-saved payload attempts CUDA construction on
        # CPU-only hosts despite map_location="cpu".
        if map_location is not None:
            raw_params["device"] = torch.device(map_location)
        raw_preprocessor = raw_params.get("preprocessor")
        artifacts = dict(payload.get("artifacts", {}))
        if version == 2:
            raw_params["preprocessor"] = _schema_v2_preprocessor_with_artifact(
                raw_preprocessor, artifacts
            )
        if "architecture" not in raw_params:
            raise ValueError("Schema-v1 checkpoint is missing estimator_params.architecture.")
        estimator = cls(**raw_params)
        fitted: dict[str, Any] = dict(payload.get("fitted", {}))
        estimator._legacy_flattened_preserve_shape_ = bool(
            fitted.get("legacy_flattened_preserve_shape", False)
        )
        if estimator._legacy_flattened_preserve_shape_:
            estimator._use_channel_first_train_inputs_ = False
        estimator._architecture_structure_ = cast(Any, payload.get("structure") or {})
        input_shape = tuple(fitted.get("input_shape") or ())
        output_dim = fitted.get("output_dim")
        primary_dim = fitted.get("primary_dim")
        if not input_shape or output_dim is None or primary_dim is None:
            raise ValueError("Schema-v1 checkpoint is missing fitted input/output metadata.")
        state_dict = payload["model_state_dict"]
        has_preprocessor = state_dict and any(str(key).startswith("preproc.") for key in state_dict)
        if estimator.preprocessor is not None:
            preprocessing = fitted.get("preprocessing")
            if version == 2 and not isinstance(preprocessing, Mapping):
                raise ValueError("Schema-v2 checkpoint is missing fitted.preprocessing metadata.")
            if not isinstance(preprocessing, Mapping):
                preprocessing = {}
            if version == 2:
                for key in ("input_topology", "output_topology", "output_dim"):
                    if key not in preprocessing:
                        raise ValueError(f"Schema-v2 fitted.preprocessing.{key} is missing.")
                if not isinstance(preprocessing["input_topology"], str):
                    raise TypeError(
                        "Schema-v2 fitted.preprocessing.input_topology must be a string."
                    )
                if not isinstance(preprocessing["output_topology"], str):
                    raise TypeError(
                        "Schema-v2 fitted.preprocessing.output_topology must be a string."
                    )
                if isinstance(preprocessing["output_dim"], bool) or not isinstance(
                    preprocessing["output_dim"], int
                ):
                    raise TypeError("Schema-v2 fitted.preprocessing.output_dim must be an integer.")
                declared = declared_preprocessor_capabilities(estimator.preprocessor)
                if preprocessing["input_topology"] != declared.input_topology:
                    raise ValueError(
                        "Schema-v2 fitted.preprocessing.input_topology conflicts with preprocessor."
                    )
                if preprocessing["output_topology"] != declared.output_topology:
                    raise ValueError(
                        "Schema-v2 fitted.preprocessing.output_topology conflicts with preprocessor."
                    )
                if preprocessing["output_dim"] != declared.output_dim:
                    raise ValueError(
                        "Schema-v2 fitted.preprocessing.output_dim conflicts with preprocessor."
                    )
            if estimator.architecture.convolution is not None:
                internal = tuple(fitted.get("internal_shape_cf") or input_shape)
                build_data = np.zeros((1,) + internal, dtype=np.float32)
                topology = f"spatial-{len(internal) - 1}d"
                shape = internal
            else:
                build_data = np.zeros((1,) + input_shape, dtype=np.float32)
                topology = "flat" if len(input_shape) == 1 else "tokens"
                shape = input_shape
            prepared = prepare_preprocessor(
                PreprocessorBuildRequest(
                    estimator.preprocessor,
                    topology,
                    shape,
                    build_data,
                    estimator._device(),
                    torch.float32,
                    reconstruction_only=True,
                )
            )
            estimator._prepared_preprocessor_ = prepared.module
            estimator.preprocessor_ = prepared.module
            estimator.preprocessor_capabilities_ = prepared.capabilities
            estimator.preprocessor_diagnostics_ = dict(preprocessing.get("diagnostics") or {})
            estimator.preprocessor_controller_ = prepared.controller
            estimator._preprocessor_input_topology_ = topology
            estimator._preprocessor_input_shape_ = tuple(shape)
            readout = estimator.preprocessor_diagnostics_.get("ols_readout")
            if isinstance(readout, torch.Tensor) and prepared.controller is not None:
                setattr(prepared.controller, "W_", readout.to(estimator._device()))
        if (
            estimator.architecture.convolution is not None
            and not estimator._legacy_flattened_preserve_shape_
        ):
            internal = tuple(fitted.get("internal_shape_cf") or input_shape)
            in_channels = int(internal[0])
            if has_preprocessor:
                preprocessor = estimator.preprocessor_
                if preprocessor is None:
                    preprocessor = getattr(estimator.lsm, "model", estimator.lsm)
                output_channels = getattr(preprocessor, "out_channels", None)
                if output_channels is None:
                    core_weight = next(
                        (
                            value
                            for key, value in state_dict.items()
                            if str(key).startswith("core.") and str(key).endswith("conv.weight")
                        ),
                        None,
                    )
                    if core_weight is None:
                        raise ValueError(
                            "Checkpoint preprocessor state is missing a convolutional core projection."
                        )
                    output_channels = int(cast(torch.Tensor, core_weight).shape[1])
                in_channels = int(output_channels)
            estimator.model_ = estimator._build_conv_core(
                len(internal) - 1,
                in_channels,
                int(output_dim),
                segmentation_head=bool(estimator.per_element),
                spatial_shape=tuple(internal[1:]),
            )
        else:
            estimator.model_ = estimator._build_dense_core(
                int(np.prod(input_shape)), int(output_dim), input_shape=input_shape
            )
        if has_preprocessor and estimator.architecture.convolution is None:
            # The core was trained on the preprocessor output, not raw X.  Its
            # saved first projection is the authoritative dimension for schema
            # reconstruction, including fitted LSMExpanders.
            core_weight = next(
                (
                    value
                    for key, value in state_dict.items()
                    if str(key).startswith("core.") and str(key).endswith("linear.weight")
                ),
                None,
            )
            if core_weight is None:
                raise ValueError(
                    "Checkpoint preprocessor state is missing a dense core projection."
                )
            estimator.model_ = estimator._build_dense_core(
                int(cast(torch.Tensor, core_weight).shape[1]),
                int(output_dim),
                input_shape=input_shape,
            )
        if has_preprocessor or (
            state_dict
            and all(str(key).startswith("core.") for key in state_dict)
            and not estimator._legacy_flattened_preserve_shape_
        ):
            # State-dict checkpoints retain constructor parameters, including the
            # fitted LSM module.  Recreate the same wrapper rather than a
            # placeholder whose empty prefix can never satisfy ``core.*`` keys.
            preprocessor = estimator.preprocessor_
            if preprocessor is None:
                preprocessor = getattr(estimator.lsm, "model", estimator.lsm)
            if not isinstance(estimator.model_, WithPreprocessor):
                estimator.model_ = WithPreprocessor(
                    cast(nn.Module | None, preprocessor), estimator.model_
                )
        estimator.model_.load_state_dict(state_dict, strict=True)
        if map_location is not None:
            estimator.device = torch.device(map_location)
        estimator.model_.to(estimator._device())
        estimator.model_.eval()
        for name, value in fitted.items():
            mapping = {
                "input_shape": "input_shape_",
                "internal_shape_cf": "_internal_input_shape_cf_",
                "primary_dim": "_primary_dim_",
                "output_dim": "_output_dim_",
                "keep_column_output": "_keep_column_output_",
                "train_layout": "_train_inputs_layout_",
                "target_cf_shape": "_target_cf_shape_",
                "target_vector_dim": "_target_vector_dim_",
                "output_shape_tuple": "_output_shape_tuple_",
                "context_dim": "_context_dim_",
                "scaler_kind": "_scaler_kind_",
                "scaler_state": "_scaler_state_",
                "target_scaler_kind": "_target_scaler_kind_",
                "target_scaler_state": "_target_scaler_state_",
                "hisso_cfg": "_hisso_cfg_",
                "hisso_options": "_hisso_options_",
                "hisso_trained": "_hisso_trained_",
            }
            target = mapping.get(name)
            if target:
                setattr(
                    estimator,
                    target,
                    (
                        tuple(value)
                        if name
                        in {
                            "input_shape",
                            "internal_shape_cf",
                            "target_cf_shape",
                            "output_shape_tuple",
                        }
                        and value is not None
                        else value
                    ),
                )
        estimator._optimizer_ = None
        estimator._hisso_trainer_ = None
        estimator._hisso_reward_fn_ = dict(payload.get("artifacts", {})).get("hisso_reward_fn")
        return estimator


__all__ = ["PSANNRegressor"]


# ``inspect.signature`` is part of sklearn's public estimator contract.  The
# implementation accepts the retained flat adapter, while introspection exposes the
# canonical constructor and therefore agrees with ``get_params(deep=False)``.
_CANONICAL_PARAM_NAMES = tuple(PSANNRegressor().get_params(deep=False))
_IMPLEMENTATION_SIGNATURE = signature(PSANNRegressor.__init__)
setattr(
    PSANNRegressor.__init__,
    "__signature__",
    Signature(
        [Parameter("self", Parameter.POSITIONAL_OR_KEYWORD)]
        + [_IMPLEMENTATION_SIGNATURE.parameters[name] for name in _CANONICAL_PARAM_NAMES]
    ),
)
