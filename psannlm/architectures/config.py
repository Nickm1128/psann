"""Immutable language-model configuration and strict capability validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass, replace
from numbers import Real
from typing import Any, Mapping, TypeVar, cast

from psann.architectures import (
    ActivationConfig,
    GeometryConfig,
    ResidualConfig,
    SpectralConfig,
    normalize_activation_config,
)

T = TypeVar("T")


def real(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    return result


def integer(value: object, path: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    if value < minimum:
        raise ValueError(f"{path} must be >= {minimum}.")
    return value


def choice(value: object, options: tuple[str, ...], path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{path} must be a string.")
    if value not in options:
        raise ValueError(f"{path} must be one of {options}.")
    return value


def probability(value: object, path: str) -> float:
    result = real(value, path)
    if not 0 <= result < 1:
        raise ValueError(f"{path} must be in [0, 1).")
    return result


def policy(cls: type[T], value: object, path: str) -> T:
    if isinstance(value, cls):
        raw = {f.name: getattr(value, f.name) for f in fields(cast(Any, cls))}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError(f"{path} must be {cls.__name__} or a mapping.")
    known = {f.name for f in fields(cast(Any, cls))}
    for key in raw:
        if not isinstance(key, str) or key not in known:
            raise ValueError(f"{path}.{key} is unknown.")
    # The shared core accepts some historical coercions. The LM boundary does not.
    numeric = {
        "amplitude_init",
        "frequency_init",
        "decay_init",
        "slope_init",
        "clip_max",
        "phase_init",
        "ratio_sum_tol",
        "alpha_init",
        "drop_path",
        "first_w0",
        "hidden_w0",
        "init",
        "strength",
    }
    for key in numeric & raw.keys():
        real(raw[key], f"{path}.{key}")
    if cls is ActivationConfig:
        if "kind" not in raw:
            raise ValueError(f"{path}.kind is required.")
        if "learnable" in raw:
            v = raw["learnable"]
            if not isinstance(v, (list, tuple)) or any(not isinstance(x, str) for x in v):
                raise TypeError(f"{path}.learnable must be a sequence of names.")
            if len(set(v)) != len(v):
                raise ValueError(f"{path}.learnable must contain unique names.")
        if raw.get("bounds") is not None:
            bounds = raw["bounds"]
            if not isinstance(bounds, Mapping):
                raise TypeError(f"{path}.bounds must be a mapping.")
            for key, pair in bounds.items():
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise TypeError(f"{path}.bounds.{key} must be a pair.")
                for x in pair:
                    if x is not None:
                        real(x, f"{path}.bounds.{key}")
    try:
        if cls is ActivationConfig:
            return normalize_activation_config(raw)  # type: ignore[return-value]
        return cls(**raw)
    except (ValueError, TypeError) as exc:
        raise type(exc)(f"{path}: {exc}") from exc


@dataclass(frozen=True)
class LMActivationInitializationConfig:
    amplitude_std: float = 0.0
    frequency_std: float = 0.0
    decay_std: float = 0.0
    amplitude_range: tuple[float, float] | None = None
    frequency_range: tuple[float, float] | None = None
    decay_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("amplitude", "frequency", "decay"):
            path = f"architecture.activation_initialization.{name}"
            std = real(getattr(self, name + "_std"), path + "_std")
            if std < 0:
                raise ValueError(f"{path}_std must be non-negative.")
            object.__setattr__(self, name + "_std", std)
            rng = getattr(self, name + "_range")
            if rng is not None:
                if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                    raise TypeError(f"{path}_range must be a pair.")
                low, high = (real(x, path + "_range") for x in rng)
                if low > high:
                    raise ValueError(f"{path}_range must be nondecreasing.")
                if std > 0:
                    raise ValueError(f"{path}_std conflicts with {path}_range.")
                object.__setattr__(self, name + "_range", (low, high))


@dataclass(frozen=True)
class LMTemporalConfig:
    mode: str = "disabled"
    kernel_size: int = 3
    dilation_growth: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        path = "architecture.temporal"
        choice(self.mode, ("disabled", "interleave", "replace", "attention-only"), path + ".mode")
        integer(self.kernel_size, path + ".kernel_size")
        if self.kernel_size % 2 == 0:
            raise ValueError(f"{path}.kernel_size must be odd.")
        integer(self.dilation_growth, path + ".dilation_growth")
        probability(self.dropout, path + ".dropout")
        if self.mode in {"disabled", "attention-only"}:
            for key, default in (("kernel_size", 3), ("dilation_growth", 1), ("dropout", 0.0)):
                if getattr(self, key) != default:
                    raise ValueError(f"{path}.{key} is inactive for mode={self.mode!r}.")


@dataclass(frozen=True)
class LMGeometryExecutionConfig:
    depth: int = 1
    chunk_size: int | None = 32

    def __post_init__(self) -> None:
        integer(self.depth, "architecture.geometry_execution.depth")
        if self.chunk_size is not None:
            integer(self.chunk_size, "architecture.geometry_execution.chunk_size")


_KINDS = ("transformer", "residual", "wave", "geometric-sparse")
_SINE_DEFAULT = ActivationConfig(decay_init=0.01)
_RESIDUAL_DEFAULT = ResidualConfig(alpha_init=1.0)


@dataclass(frozen=True)
class LMArchitectureConfig:
    kind: str
    activation: ActivationConfig
    activation_initialization: LMActivationInitializationConfig | None = None
    residual: ResidualConfig | None = None
    spectral: SpectralConfig | None = None
    temporal: LMTemporalConfig | None = None
    geometry: GeometryConfig | None = None
    geometry_execution: LMGeometryExecutionConfig | None = None

    def __post_init__(self) -> None:
        choice(self.kind, _KINDS, "architecture.kind")
        for key, cls in (
            ("activation", ActivationConfig),
            ("residual", ResidualConfig),
            ("spectral", SpectralConfig),
            ("temporal", LMTemporalConfig),
            ("geometry", GeometryConfig),
            ("geometry_execution", LMGeometryExecutionConfig),
            ("activation_initialization", LMActivationInitializationConfig),
        ):
            value = getattr(self, key)
            if value is not None:
                object.__setattr__(self, key, policy(cls, value, "architecture." + key))
        validate_capabilities(self)

    @classmethod
    def transformer(cls, **values: Any) -> LMArchitectureConfig:
        return cls(
            **dict({"kind": "transformer", "activation": ActivationConfig(kind="gelu")}, **values)
        )

    @classmethod
    def _residual(cls, **values: Any) -> LMArchitectureConfig:
        return cls(
            **dict(
                {"kind": "residual", "activation": _SINE_DEFAULT, "residual": _RESIDUAL_DEFAULT},
                **values,
            )
        )

    @classmethod
    def wave(cls, **values: Any) -> LMArchitectureConfig:
        return cls(
            **dict(
                {
                    "kind": "wave",
                    "activation": _SINE_DEFAULT,
                    "residual": _RESIDUAL_DEFAULT,
                    "temporal": LMTemporalConfig(),
                },
                **values,
            )
        )

    @classmethod
    def geometric_sparse(cls, **values: Any) -> LMArchitectureConfig:
        return cls(
            **dict(
                {
                    "kind": "geometric-sparse",
                    "activation": _SINE_DEFAULT,
                    "residual": _RESIDUAL_DEFAULT,
                    "geometry": GeometryConfig(),
                    "geometry_execution": LMGeometryExecutionConfig(),
                },
                **values,
            )
        )


def validate_capabilities(config: LMArchitectureConfig) -> None:
    allowed = {
        "transformer": set(),
        "residual": {"residual", "spectral"},
        "wave": {"residual", "temporal"},
        "geometric-sparse": {"residual", "geometry", "geometry_execution"},
    }[config.kind]
    required = allowed - {"spectral"}
    for name in ("residual", "spectral", "temporal", "geometry", "geometry_execution"):
        value = getattr(config, name)
        if name in required and value is None:
            raise ValueError(f"architecture.{name} is required for {config.kind}.")
        if name not in allowed and value is not None:
            raise ValueError(f"architecture.{name} is unsupported by {config.kind}.")
    if config.activation is None:
        raise ValueError("architecture.activation is required.")
    act = config.activation
    allowed_acts = {
        "transformer": {"gelu", "relu"},
        "residual": {"psann", "gelu"},
        "wave": {"psann", "gelu"},
        "geometric-sparse": {"psann", "gelu", "relu", "tanh", "mixed"},
    }[config.kind]
    if act.kind not in allowed_acts:
        raise ValueError(
            f"architecture.activation.kind={act.kind!r} is unsupported by {config.kind}."
        )
    if act.kind == "mixed" and any(
        x not in {"psann", "gelu", "relu", "tanh"} for x in act.activation_types or ()
    ):
        raise ValueError(
            "architecture.activation.activation_types contains an unsupported LM child."
        )
    psann = act.kind == "psann" or (act.kind == "mixed" and "psann" in (act.activation_types or ()))
    if act.feature_dim != -1:
        raise ValueError("architecture.activation.feature_dim must be -1 for LM tensors.")
    default = ActivationConfig()
    executed = {
        "kind",
        "activation_types",
        "activation_ratios",
        "mix_layout",
        "mix_seed",
        "ratio_sum_tol",
    }
    if psann:
        executed |= {
            "amplitude_init",
            "frequency_init",
            "decay_init",
            "learnable",
            "bounds",
            "decay_mode",
        }
    for field in fields(act):
        if field.name not in executed and getattr(act, field.name) != getattr(default, field.name):
            raise ValueError(f"architecture.activation.{field.name} is inactive for {act.kind}.")
    if config.activation_initialization is not None and not psann:
        raise ValueError(
            "architecture.activation_initialization requires an executing PSANN child."
        )
    if config.temporal is not None and config.temporal.mode == "attention-only":
        if act != _SINE_DEFAULT:
            raise ValueError("architecture.activation must be the default for attention-only.")
        if config.activation_initialization is not None:
            raise ValueError(
                "architecture.activation_initialization is inactive for attention-only."
            )
        if config.residual is not None and (
            config.residual.alpha_init != 1.0 or config.residual.drop_path != 0.0
        ):
            raise ValueError(
                "architecture.residual.alpha_init/drop_path are inactive for attention-only."
            )
    if config.residual is not None:
        if config.residual.first_w0 != 12.0 or config.residual.hidden_w0 != 1.0:
            raise ValueError(
                "architecture.residual.first_w0/hidden_w0 are unsupported by LM builders."
            )


def normalize_architecture(value: object) -> LMArchitectureConfig:
    if isinstance(value, LMArchitectureConfig):
        return value
    if isinstance(value, str):
        choice(value, _KINDS, "architecture.kind")
        return getattr(
            LMArchitectureConfig, "_residual" if value == "residual" else value.replace("-", "_")
        )()
    if not isinstance(value, Mapping):
        raise TypeError("architecture must be a typed policy, mapping, or canonical preset.")
    if "kind" not in value:
        raise ValueError("architecture.kind is required.")
    return policy(LMArchitectureConfig, value, "architecture")


@dataclass(frozen=True)
class LMConfig:
    architecture: LMArchitectureConfig
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_mlp: int | None = None
    vocab_size: int | None = None
    dropout: float = 0.0
    positional_encoding: str = "rope"
    attention_implementation: str = "math"

    def __post_init__(self) -> None:
        object.__setattr__(self, "architecture", normalize_architecture(self.architecture))
        for name in ("d_model", "n_layers", "n_heads"):
            integer(getattr(self, name), "config." + name)
        for name in ("d_mlp", "vocab_size"):
            if getattr(self, name) is not None:
                integer(getattr(self, name), name)
        probability(self.dropout, "dropout")
        choice(self.positional_encoding, ("rope", "alibi", "sinusoidal"), "positional_encoding")
        choice(self.attention_implementation, ("math", "sdpa", "auto"), "attention_implementation")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.positional_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("n_heads must give an even RoPE head dimension.")
        if self.positional_encoding == "sinusoidal" and self.d_model % 2:
            raise ValueError("d_model must be even for sinusoidal positions.")
        geo = self.architecture.geometry
        if (
            geo is not None
            and geo.shape is not None
            and math.prod(geo.shape) != (self.d_mlp or 4 * self.d_model)
        ):
            raise ValueError("architecture.geometry.shape must multiply to d_mlp.")
        activation = self.architecture.activation
        if activation.kind == "mixed":
            from psann.architectures.components import activation_feature_counts

            counts = activation_feature_counts(activation, features=self.d_mlp or 4 * self.d_model)
            if counts.get("psann", 0) == 0:
                if self.architecture.activation_initialization is not None:
                    raise ValueError(
                        "architecture.activation_initialization requires an executing PSANN child with positive width."
                    )
                defaults = ActivationConfig()
                for key in (
                    "amplitude_init",
                    "frequency_init",
                    "decay_init",
                    "learnable",
                    "bounds",
                    "decay_mode",
                ):
                    if getattr(activation, key) != getattr(defaults, key):
                        raise ValueError(
                            f"architecture.activation.{key} is inactive without a positive-width PSANN child."
                        )


def normalize_lm_config(value: object, *, for_build: bool = False, **flat: Any) -> LMConfig:
    if isinstance(value, LMConfig):
        result = value
    elif isinstance(value, Mapping):
        raw = dict(value)
        if "kind" in raw:
            choice(raw.pop("kind"), ("lm",), "kind")
        if "architecture" not in raw:
            raise ValueError("architecture is required.")
        result = policy(LMConfig, raw, "config")
    else:
        raise TypeError("config must be LMConfig or a mapping.")
    if flat:
        from .compat import check_flat_duplicates

        check_flat_duplicates(result, flat)
    if for_build:
        if result.vocab_size is None:
            raise ValueError("vocab_size must be resolved before construction.")
        if result.d_mlp is None:
            result = replace(result, d_mlp=4 * result.d_model)
    return result


def to_mapping(value: Any) -> Any:
    """Serialize immutable configuration to fresh JSON-compatible containers."""
    if is_dataclass(value):
        result = {f.name: to_mapping(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, LMConfig):
            result = dict(kind="lm", **result)
        return result
    if isinstance(value, Mapping):
        return {key: to_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_mapping(item) for item in value]
    return value


# Install the named constructor after dataclass captures the instance field's
# None default. Instances carry their residual policy; the class offers the preset.
setattr(LMArchitectureConfig, "residual", LMArchitectureConfig.__dict__["_residual"])
