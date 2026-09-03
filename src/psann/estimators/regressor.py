"""The canonical registry-driven sklearn regressor.

The inherited mixins deliberately retain the Phase 2 training, scaling, HISSO, and
streaming implementation.  This class replaces only the former subclass-based model
selection with immutable architecture configuration and registry requests.
"""

from __future__ import annotations

import warnings
from dataclasses import fields
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .._sklearn.base import PSANNRegressor as _Phase2Regressor
from ..architectures import (
    ArchitectureBuildRequest,
    ArchitectureConfig,
    ArchitectureLike,
    AttentionConfig,
    ConvolutionConfig,
    StateConfig,
    build_architecture,
    architecture_to_mapping,
    normalize_architecture,
)
from ..architectures.config import replace_architecture_path, validate_architecture
from ..attention import AttentionConfig as LegacyAttentionConfig
from ..state import StateConfig as LegacyStateConfig
from ..types import LossLike, ScalerSpec

_DEFAULT_ARCHITECTURE = ArchitectureConfig.dense()


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
    activation_values.setdefault("kind", activation_type)
    if preserve_shape:
        convolution = ConvolutionConfig(conv_channels, conv_kernel_size, data_format, per_element)
        return ArchitectureConfig.convolutional(
            activation=(
                ArchitectureConfig.dense().activation
                if not activation_values
                else __import__(
                    "psann.architectures", fromlist=["ActivationConfig"]
                ).ActivationConfig(**activation_values)
            ),
            convolution=convolution,
        )
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
        activation=__import__(
            "psann.architectures", fromlist=["ActivationConfig"]
        ).ActivationConfig(**activation_values),
        attention=attention_value,
        state=state_value,
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
        lsm: object | None = None,
        lsm_train: bool = False,
        lsm_pretrain_epochs: int = 0,
        lsm_lr: float | None = None,
        activation: object | None = None,
        activation_type: str = "psann",
        preserve_shape: bool = False,
        data_format: str = "channels_first",
        conv_kernel_size: int = 1,
        conv_channels: int | None = None,
        per_element: bool = False,
        attention: object | None = None,
        stateful: bool = False,
        state: object | None = None,
        state_reset: str = "batch",
        stream_lr: float | None = None,
        context_builder: object | None = None,
        context_builder_params: object | None = None,
        w0_first: float = 12.0,
        w0_hidden: float = 1.0,
        norm: str = "rms",
        drop_path_max: float = 0.0,
        residual_alpha_init: float = 0.0,
        first_layer_w0: float = 30.0,
        hidden_w0: float = 1.0,
        dropout: float = 0.0,
        grad_clip_norm: float | None = 5.0,
        first_layer_w0_initial: float | None = 10.0,
        hidden_w0_initial: float | None = 0.5,
        w0_warmup_epochs: int = 10,
        progressive_depth_initial: int | None = None,
        progressive_depth_interval: int = 15,
        progressive_depth_growth: int = 1,
        context_dim: int | None = None,
        use_film: bool = True,
        use_phase_shift: bool = True,
        use_spectral_gate: bool = False,
        k_fft: int = 64,
        gate_type: str = "rfft",
        gate_groups: str = "depthwise",
        gate_init: float = 0.0,
        gate_strength: float = 1.0,
        phase_init: float = 0.0,
        phase_trainable: bool = True,
        pool: str = "last",
        shape: tuple[int, int] | None = None,
        k: int = 8,
        pattern: str = "local",
        radius: int = 1,
        offsets: object | None = None,
        wrap_mode: str = "clamp",
        bias: bool = True,
        compute_mode: str = "gather",
        geo_seed: int | None = None,
    ) -> None:
        explicit_flat = any(
            (
                activation is not None,
                activation_type != "psann",
                preserve_shape,
                conv_channels is not None,
                attention is not None,
                stateful,
                state is not None,
                context_builder is not None,
                context_dim is not None,
                use_spectral_gate,
                shape is not None,
                progressive_depth_initial is not None,
            )
        )
        if architecture is not _DEFAULT_ARCHITECTURE and explicit_flat:
            names = [
                name
                for name, enabled in (
                    ("activation", activation is not None),
                    ("preserve_shape", preserve_shape),
                    ("attention", attention is not None),
                    ("state", state is not None),
                    ("context_builder", context_builder is not None),
                )
                if enabled
            ]
            raise ValueError(
                "architecture conflicts with legacy architecture keyword(s): "
                + ", ".join(names or ["activation_type"])
            )
        if architecture is _DEFAULT_ARCHITECTURE and explicit_flat:
            warnings.warn(
                "Flat architecture keywords are deprecated; pass architecture=ArchitectureConfig(...).",
                DeprecationWarning,
                stacklevel=2,
            )
            architecture = _legacy_architecture(
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
        units = 64 if hidden_units is None else hidden_units
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
            activation=_activation_mapping(canonical),
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
            lsm=lsm,
            lsm_train=lsm_train,
            lsm_pretrain_epochs=lsm_pretrain_epochs,
            lsm_lr=lsm_lr,
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
            context_builder=canonical.context.builder if canonical.context else context_builder,
            context_builder_params=(
                dict(canonical.context.builder_params)
                if canonical.context and canonical.context.builder_params
                else context_builder_params
            ),
        )
        self.architecture = canonical
        self.hidden_width = hidden_width if hidden_width is not None else units
        self._use_channel_first_train_inputs_ = canonical.convolution is not None
        self._architecture_capabilities_ = None
        self._architecture_lifecycle_ = None
        self._architecture_structure_ = None

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
        return ArchitectureBuildRequest(
            self.architecture,
            self.hidden_layers,
            self.hidden_units,
            tuple(input_shape),
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
            None,
            None,
            getattr(self, "_architecture_structure_", None),
            self.w0,
        )

    def _receive_build(self, request: ArchitectureBuildRequest) -> nn.Module:
        result = build_architecture(request)
        self._architecture_capabilities_ = result.capabilities
        self._architecture_lifecycle_ = result.lifecycle
        self._architecture_lifecycle_.on_model_built(model=result.model, runtime={})
        return result.model

    def _build_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        *,
        state_cfg: Optional[dict[str, Any]] = None,
        input_shape: Optional[tuple[int, ...]] = None,
    ) -> nn.Module:
        return self._receive_build(
            self._request(
                input_dim=input_dim, output_dim=output_dim, input_shape=input_shape or (input_dim,)
            )
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
            "lsm",
            "lsm_train",
            "lsm_pretrain_epochs",
            "lsm_lr",
        )
        params = {name: getattr(self, name) for name in names}
        if deep:
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
            "_architecture_lifecycle_",
            "_architecture_capabilities_",
            "_architecture_structure_",
        ):
            self.__dict__.pop(name, None)

    def set_params(self, **params: object) -> "PSANNRegressor":
        if not params:
            return self
        if "architecture" in params and any(key.startswith("architecture__") for key in params):
            raise ValueError(
                "architecture and architecture__ parameters cannot be updated together."
            )
        candidate = self.architecture
        if "architecture" in params:
            candidate = normalize_architecture(params.pop("architecture"))
        for key in list(params):
            if key.startswith("architecture__"):
                candidate = replace_architecture_path(
                    candidate,
                    key,
                    params.pop(key),
                    hidden_layers=int(params.get("hidden_layers", self.hidden_layers)),
                )
        validate_architecture(
            candidate, hidden_layers=int(params.get("hidden_layers", self.hidden_layers))
        )
        valid = set(self.get_params(deep=False))
        unknown = set(params) - valid
        if unknown:
            raise ValueError(f"Invalid parameter {sorted(unknown)[0]!r} for PSANNRegressor.")
        changed_architecture = candidate != self.architecture
        for key, value in params.items():
            setattr(self, key, value)
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
            self.activation = _activation_mapping(candidate)
            self.activation_type = candidate.activation.kind
            self.attention = _legacy_attention(candidate)
            self.state = _legacy_state(candidate)
            self.stateful = candidate.state is not None
            self._clear_architecture_runtime()
        return self

    def save(self, path: str) -> None:
        """Write a portable schema-v1 payload without serialising a final module."""

        self._ensure_fitted()
        model = self.model_
        if model is None:
            raise RuntimeError("Estimator is not fitted.")
        state_dict = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        params = self.get_params(deep=False)
        params["architecture"] = architecture_to_mapping(self.architecture)
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
        }
        structure = (
            self._architecture_lifecycle_.structure_metadata()
            if self._architecture_lifecycle_
            else {}
        )
        torch.save(
            {
                "schema": "psann.regressor",
                "schema_version": 1,
                "package_version": "0.12.4",
                "estimator_params": params,
                "fitted": fitted,
                "structure": structure,
                "model_state_dict": state_dict,
                "artifacts": {},
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str, *, map_location: Optional[Union[str, torch.device]] = "cpu"
    ) -> "PSANNRegressor":
        try:
            payload = torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=map_location)
        if payload.get("schema") != "psann.regressor":
            # Preserve unversioned Phase-2 reading, then make the migrated object
            # explicitly canonical on its next save.  The old deserialised module is
            # intentionally retained only in memory; it is never emitted by v1.
            legacy = _Phase2Regressor.load(path, map_location=map_location)
            migrated = cls(
                architecture="dense",
                **{
                    key: value
                    for key, value in legacy.get_params(False).items()
                    if key in cls().get_params(False) and key != "architecture"
                },
            )
            migrated.model_ = legacy.model_
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
            return migrated
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported psann.regressor schema version {payload.get('schema_version')!r}."
            )
        raw_params = dict(payload.get("estimator_params", {}))
        if "architecture" not in raw_params:
            raise ValueError("Schema-v1 checkpoint is missing estimator_params.architecture.")
        estimator = cls(**raw_params)
        fitted = dict(payload.get("fitted", {}))
        estimator._architecture_structure_ = payload.get("structure") or {}
        input_shape = tuple(fitted.get("input_shape") or ())
        output_dim = fitted.get("output_dim")
        primary_dim = fitted.get("primary_dim")
        if not input_shape or output_dim is None or primary_dim is None:
            raise ValueError("Schema-v1 checkpoint is missing fitted input/output metadata.")
        if estimator.architecture.convolution is not None:
            internal = tuple(fitted.get("internal_shape_cf") or input_shape)
            estimator.model_ = estimator._build_conv_core(
                len(internal) - 1,
                int(internal[0]),
                int(output_dim),
                segmentation_head=bool(estimator.per_element),
                spatial_shape=tuple(internal[1:]),
            )
        else:
            estimator.model_ = estimator._build_dense_core(
                int(np.prod(input_shape)), int(output_dim), input_shape=input_shape
            )
        estimator.model_.load_state_dict(payload["model_state_dict"], strict=True)
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
        return estimator


__all__ = ["PSANNRegressor"]
