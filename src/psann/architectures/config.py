"""Immutable public architecture policies for :class:`PSANNRegressor`.

The objects in this module deliberately describe configuration only.  Builders own
the translation to the legacy numerical backbones, which keeps configuration useful
for sklearn cloning, persistence, and validation without making model modules depend
on estimators.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, fields, replace
from collections.abc import Iterator
from typing import Any, Callable, Mapping, TypeAlias, cast


class FrozenMapping(Mapping[str, object]):
    """Pickle-safe immutable mapping with deterministic equality and iteration."""

    def __init__(self, values: Mapping[str, object]) -> None:
        self._items = tuple((str(key), _freeze(item)) for key, item in sorted(values.items()))
        self._values = dict(self._items)

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __hash__(self) -> int:
        return hash(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())


def _canonical_name(value: str) -> str:
    """Canonicalise documented separator variants, and nothing else."""

    if not isinstance(value, str):
        raise TypeError("architecture.kind must be a string.")
    return value.strip().lower().replace("_", "-")


def _finite(value: float, path: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    return result


def _positive(value: float, path: str) -> float:
    result = _finite(value, path)
    if result <= 0:
        raise ValueError(f"{path} must be positive.")
    return result


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path} must be at least {minimum}.")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean.")
    return value


def _frozen_mapping(value: Mapping[str, object] | None, path: str) -> FrozenMapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping or None.")
    return FrozenMapping(value)


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ActivationConfig:
    kind: str = "psann"
    amplitude_init: float = 1.0
    frequency_init: float = 1.0
    decay_init: float = 0.1
    learnable: tuple[str, ...] = ("amplitude", "frequency", "decay")
    decay_mode: str = "abs"
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None
    slope_init: float = 1.0
    slope_trainable: bool = True
    clip_max: float = 1.0
    activation_types: tuple[str, ...] | None = None
    activation_ratios: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        kind = _canonical_name(self.kind)
        if kind not in {"psann", "relu", "tanh", "relu-sigmoid-psann"}:
            raise ValueError("activation.kind must be psann, relu, tanh, or relu-sigmoid-psann.")
        object.__setattr__(self, "kind", kind)
        if _positive(self.amplitude_init, "activation.amplitude_init") <= 0:
            raise ValueError("activation.amplitude_init must be positive.")
        _positive(self.frequency_init, "activation.frequency_init")
        if _finite(self.decay_init, "activation.decay_init") < 0:
            raise ValueError("activation.decay_init must be non-negative.")
        learnable = (self.learnable,) if isinstance(self.learnable, str) else tuple(self.learnable)
        allowed = {"amplitude", "frequency", "decay"}
        if any(item not in allowed for item in learnable):
            raise ValueError("activation.learnable contains an unknown value.")
        object.__setattr__(self, "learnable", learnable)
        mode = _canonical_name(self.decay_mode)
        if mode not in {"abs", "relu", "none"}:
            raise ValueError("activation.decay_mode must be abs, relu, or none.")
        object.__setattr__(self, "decay_mode", mode)
        bounds = _frozen_mapping(
            dict(self.bounds) if self.bounds is not None else None, "activation.bounds"
        )
        if bounds is not None:
            for key, pair in bounds.items():
                if key not in allowed or not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError("activation.bounds must map activation fields to bound pairs.")
                low, high = pair
                if low is not None:
                    _finite(low, f"activation.bounds.{key}[0]")
                if high is not None:
                    _finite(high, f"activation.bounds.{key}[1]")
                if low is not None and high is not None and low > high:
                    raise ValueError(f"activation.bounds.{key} has reversed bounds.")
        object.__setattr__(self, "bounds", bounds)
        _finite(self.slope_init, "activation.slope_init")
        _boolean(self.slope_trainable, "activation.slope_trainable")
        _positive(self.clip_max, "activation.clip_max")
        if self.activation_types is not None:
            types = tuple(_canonical_name(item) for item in self.activation_types)
            ratios = tuple(float(item) for item in self.activation_ratios or ())
            if not types or len(types) != len(ratios) or any(item < 0 for item in ratios):
                raise ValueError(
                    "activation.activation_types and activation_ratios must be matching non-empty tuples."
                )
            object.__setattr__(self, "activation_types", types)
            object.__setattr__(self, "activation_ratios", ratios)
        elif self.activation_ratios is not None:
            raise ValueError("activation.activation_ratios requires activation_types.")


@dataclass(frozen=True)
class ResidualConfig:
    norm: str = "rms"
    alpha_init: float = 0.0
    drop_path: float = 0.0
    first_w0: float = 12.0
    hidden_w0: float = 1.0

    def __post_init__(self) -> None:
        norm = _canonical_name(self.norm)
        if norm not in {"rms", "layer", "none"}:
            raise ValueError("residual.norm must be rms, layer, or none.")
        object.__setattr__(self, "norm", norm)
        _finite(self.alpha_init, "residual.alpha_init")
        drop = _finite(self.drop_path, "residual.drop_path")
        if not 0 <= drop < 1:
            raise ValueError("residual.drop_path must satisfy 0 <= value < 1.")
        _positive(self.first_w0, "residual.first_w0")
        _positive(self.hidden_w0, "residual.hidden_w0")


@dataclass(frozen=True)
class ConvolutionConfig:
    channels: int | None = None
    kernel_size: int = 1
    data_format: str = "channels_first"
    per_element: bool = False

    def __post_init__(self) -> None:
        if self.channels is not None:
            object.__setattr__(
                self, "channels", _integer(self.channels, "convolution.channels", minimum=1)
            )
        object.__setattr__(
            self, "kernel_size", _integer(self.kernel_size, "convolution.kernel_size", minimum=1)
        )
        fmt = _canonical_name(self.data_format)
        if fmt not in {"channels-first", "channels-last"}:
            raise ValueError("convolution.data_format must be channels_first or channels_last.")
        object.__setattr__(self, "data_format", fmt.replace("-", "_"))
        _boolean(self.per_element, "convolution.per_element")


@dataclass(frozen=True)
class AttentionConfig:
    kind: str = "mha"
    num_heads: int = 4
    dropout: float = 0.0
    bias: bool = True
    batch_first: bool = True
    add_bias_kv: bool = False
    add_zero_attn: bool = False

    def __post_init__(self) -> None:
        if _canonical_name(self.kind) != "mha":
            raise ValueError("attention.kind must be mha.")
        object.__setattr__(
            self, "num_heads", _integer(self.num_heads, "attention.num_heads", minimum=1)
        )
        drop = _finite(self.dropout, "attention.dropout")
        if not 0 <= drop < 1:
            raise ValueError("attention.dropout must satisfy 0 <= value < 1.")
        for name in ("bias", "batch_first", "add_bias_kv", "add_zero_attn"):
            _boolean(getattr(self, name), f"attention.{name}")


@dataclass(frozen=True)
class StateConfig:
    rho: float = 0.95
    beta: float = 1.0
    init: float = 1.0
    max_abs: float = 5.0
    detach: bool = True
    reset: str = "batch"
    stream_lr: float | None = None

    def __post_init__(self) -> None:
        rho = _finite(self.rho, "state.rho")
        if not 0 <= rho < 1:
            raise ValueError("state.rho must satisfy 0 <= value < 1.")
        if _finite(self.beta, "state.beta") < 0:
            raise ValueError("state.beta must be non-negative.")
        _finite(self.init, "state.init")
        _positive(self.max_abs, "state.max_abs")
        reset = _canonical_name(self.reset)
        if reset not in {"batch", "epoch", "none"}:
            raise ValueError("state.reset must be batch, epoch, or none.")
        object.__setattr__(self, "reset", reset)
        if self.stream_lr is not None:
            _positive(self.stream_lr, "state.stream_lr")
        _boolean(self.detach, "state.detach")


@dataclass(frozen=True)
class ContextConfig:
    dim: int | None = None
    builder: str | Callable[..., object] | None = None
    builder_params: Mapping[str, object] | None = None
    film: bool = True
    phase_shift: bool = True

    def __post_init__(self) -> None:
        if self.dim is not None:
            object.__setattr__(self, "dim", _integer(self.dim, "context.dim", minimum=1))
        if (
            self.builder is not None
            and not isinstance(self.builder, str)
            and not callable(self.builder)
        ):
            raise TypeError("context.builder must be a string, callable, or None.")
        object.__setattr__(
            self,
            "builder_params",
            _frozen_mapping(
                dict(self.builder_params) if self.builder_params is not None else None,
                "context.builder_params",
            ),
        )
        _boolean(self.film, "context.film")
        _boolean(self.phase_shift, "context.phase_shift")


@dataclass(frozen=True)
class W0WarmupConfig:
    first_initial: float = 10.0
    hidden_initial: float = 0.5
    epochs: int = 10

    def __post_init__(self) -> None:
        _positive(self.first_initial, "wave.warmup.first_initial")
        _positive(self.hidden_initial, "wave.warmup.hidden_initial")
        object.__setattr__(self, "epochs", _integer(self.epochs, "wave.warmup.epochs", minimum=0))


@dataclass(frozen=True)
class ProgressiveDepthConfig:
    initial_layers: int
    interval: int = 15
    growth: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "initial_layers",
            _integer(self.initial_layers, "wave.progressive_depth.initial_layers", minimum=1),
        )
        object.__setattr__(
            self, "interval", _integer(self.interval, "wave.progressive_depth.interval", minimum=1)
        )
        object.__setattr__(
            self, "growth", _integer(self.growth, "wave.progressive_depth.growth", minimum=1)
        )


@dataclass(frozen=True)
class WaveConfig:
    first_w0: float = 30.0
    hidden_w0: float = 1.0
    norm: str = "none"
    dropout: float = 0.0
    grad_clip_norm: float | None = 5.0
    warmup: W0WarmupConfig | None = None
    progressive_depth: ProgressiveDepthConfig | None = None

    def __post_init__(self) -> None:
        _positive(self.first_w0, "wave.first_w0")
        _positive(self.hidden_w0, "wave.hidden_w0")
        norm = _canonical_name(self.norm)
        if norm not in {"none", "weight", "rms"}:
            raise ValueError("wave.norm must be none, weight, or rms.")
        object.__setattr__(self, "norm", norm)
        drop = _finite(self.dropout, "wave.dropout")
        if not 0 <= drop < 1:
            raise ValueError("wave.dropout must satisfy 0 <= value < 1.")
        if self.grad_clip_norm is not None:
            _positive(self.grad_clip_norm, "wave.grad_clip_norm")


@dataclass(frozen=True)
class SpectralConfig:
    k_fft: int = 64
    gate_type: str = "rfft"
    groups: str = "depthwise"
    init: float = 0.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_fft", _integer(self.k_fft, "spectral.k_fft", minimum=1))
        kind = _canonical_name(self.gate_type)
        if kind not in {"rfft", "fourier-features"}:
            raise ValueError("spectral.gate_type must be rfft or fourier-features.")
        object.__setattr__(self, "gate_type", kind)
        groups = _canonical_name(self.groups)
        if groups not in {"depthwise", "full"}:
            raise ValueError("spectral.groups must be depthwise or full.")
        object.__setattr__(self, "groups", groups)
        _finite(self.init, "spectral.init")
        if _finite(self.strength, "spectral.strength") < 0:
            raise ValueError("spectral.strength must be non-negative.")


@dataclass(frozen=True)
class SequenceConfig:
    phase_init: float = 0.0
    phase_trainable: bool = True
    pool: str = "last"

    def __post_init__(self) -> None:
        _finite(self.phase_init, "sequence.phase_init")
        _boolean(self.phase_trainable, "sequence.phase_trainable")
        pool = _canonical_name(self.pool)
        if pool not in {"last", "mean"}:
            raise ValueError("sequence.pool must be last or mean.")
        object.__setattr__(self, "pool", pool)


@dataclass(frozen=True)
class GeometryConfig:
    shape: tuple[int, int] | None = None
    k: int = 8
    pattern: str = "local"
    radius: int = 1
    offsets: tuple[tuple[int, int], ...] | None = None
    wrap_mode: str = "clamp"
    bias: bool = True
    compute_mode: str = "gather"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.shape is not None:
            shape = tuple(_integer(item, "geometry.shape", minimum=1) for item in self.shape)
            if len(shape) != 2 or any(item <= 0 for item in shape):
                raise ValueError("geometry.shape must contain two positive dimensions.")
            object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "k", _integer(self.k, "geometry.k", minimum=1))
        object.__setattr__(self, "radius", _integer(self.radius, "geometry.radius", minimum=0))
        pattern = _canonical_name(self.pattern)
        if pattern not in {"local", "random", "hash"}:
            raise ValueError("geometry.pattern must be local, random, or hash.")
        object.__setattr__(self, "pattern", pattern)
        wrap = _canonical_name(self.wrap_mode)
        if wrap not in {"clamp", "wrap"}:
            raise ValueError("geometry.wrap_mode must be clamp or wrap.")
        object.__setattr__(self, "wrap_mode", wrap)
        compute = _canonical_name(self.compute_mode)
        if compute not in {"gather", "scatter"}:
            raise ValueError("geometry.compute_mode must be gather or scatter.")
        object.__setattr__(self, "compute_mode", compute)
        if self.offsets is not None:
            offsets = tuple(
                tuple(_integer(value, "geometry.offsets") for value in pair)
                for pair in self.offsets
            )
            if not offsets or any(len(pair) != 2 for pair in offsets):
                raise ValueError("geometry.offsets must be non-empty pairs of integers.")
            object.__setattr__(self, "offsets", offsets)
        _boolean(self.bias, "geometry.bias")
        if self.seed is not None:
            object.__setattr__(self, "seed", _integer(self.seed, "geometry.seed"))


@dataclass(frozen=True)
class ArchitectureConfig:
    kind: str = "dense"
    activation: ActivationConfig = ActivationConfig()
    residual: ResidualConfig | None = None
    convolution: ConvolutionConfig | None = None
    attention: AttentionConfig | None = None
    state: StateConfig | None = None
    context: ContextConfig | None = None
    wave: WaveConfig | None = None
    spectral: SpectralConfig | None = None
    sequence: SequenceConfig | None = None
    geometry: GeometryConfig | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _canonical_name(self.kind))
        # Typed construction has the same policy normalization as a tagged
        # mapping.  Estimator-dependent progressive-depth validation occurs
        # later, when the actual hidden layer count is available.
        for name in _POLICIES:
            object.__setattr__(self, name, _policy_from_mapping(name, getattr(self, name)))
        validate_architecture(self, hidden_layers=None)

    @classmethod
    def dense(cls, **kwargs: object) -> "ArchitectureConfig":
        return cls(kind="dense", **cast(Any, kwargs))

    @classmethod
    def convolutional(cls, **kwargs: object) -> "ArchitectureConfig":
        kwargs.setdefault("convolution", ConvolutionConfig())
        return cls(kind="convolutional", **cast(Any, kwargs))

    @classmethod
    def for_wave(cls, **kwargs: object) -> "ArchitectureConfig":
        kwargs.setdefault("residual", ResidualConfig())
        kwargs.setdefault("wave", WaveConfig())
        return cls(kind="wave", **cast(Any, kwargs))

    @classmethod
    def for_sequence(cls, **kwargs: object) -> "ArchitectureConfig":
        kwargs.setdefault("sequence", SequenceConfig())
        kwargs.setdefault("spectral", SpectralConfig())
        return cls(kind="sequence", **cast(Any, kwargs))

    @classmethod
    def geometric_sparse(cls, **kwargs: object) -> "ArchitectureConfig":
        kwargs.setdefault("residual", ResidualConfig())
        kwargs.setdefault("geometry", GeometryConfig())
        return cls(kind="geometric-sparse", **cast(Any, kwargs))


ArchitectureLike: TypeAlias = ArchitectureConfig | Mapping[str, object] | str

_POLICIES: dict[str, type[Any]] = {
    "activation": ActivationConfig,
    "residual": ResidualConfig,
    "convolution": ConvolutionConfig,
    "attention": AttentionConfig,
    "state": StateConfig,
    "context": ContextConfig,
    "wave": WaveConfig,
    "spectral": SpectralConfig,
    "sequence": SequenceConfig,
    "geometry": GeometryConfig,
}


def _policy_from_mapping(name: str, value: object) -> object | None:
    if value is None:
        return None
    cls = _POLICIES[name]
    if isinstance(value, cls):
        return value
    if name == "attention" and getattr(value, "kind", None) in {"none", "off", ""}:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"architecture.{name} must be a mapping, {cls.__name__}, or None.")
    raw = dict(value)
    if name == "attention" and _canonical_name(str(raw.get("kind", "mha"))) in {"none", "off"}:
        return None
    known = {field.name for field in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"architecture.{name}.{sorted(unknown)[0]} is unknown.")
    if name == "context" and "builder_params" in raw:
        raw["builder_params"] = _frozen_mapping(
            raw["builder_params"], "architecture.context.builder_params"
        )
    if name == "activation":
        if "learnable" in raw and isinstance(raw["learnable"], list):
            raw["learnable"] = tuple(raw["learnable"])
        if "activation_types" in raw and isinstance(raw["activation_types"], list):
            raw["activation_types"] = tuple(raw["activation_types"])
        if "activation_ratios" in raw and isinstance(raw["activation_ratios"], list):
            raw["activation_ratios"] = tuple(raw["activation_ratios"])
        if "bounds" in raw:
            raw["bounds"] = _frozen_mapping(raw["bounds"], "architecture.activation.bounds")
    if name == "geometry":
        for field_name in ("shape", "offsets"):
            if field_name in raw and raw[field_name] is not None:
                raw[field_name] = tuple(
                    tuple(item) if isinstance(item, (list, tuple)) else item
                    for item in raw[field_name]
                )
    if name == "wave":
        if isinstance(raw.get("warmup"), Mapping):
            raw["warmup"] = W0WarmupConfig(**raw["warmup"])
        if isinstance(raw.get("progressive_depth"), Mapping):
            raw["progressive_depth"] = ProgressiveDepthConfig(**raw["progressive_depth"])
    return cls(**raw)  # type: ignore[call-arg]


def _from_mapping(value: Mapping[str, object]) -> ArchitectureConfig:
    raw = dict(value)
    unknown = set(raw) - ({"kind"} | set(_POLICIES))
    if unknown:
        raise ValueError(f"architecture.{sorted(unknown)[0]} is unknown.")
    if "kind" not in raw:
        raise ValueError("architecture.kind is required.")
    policies = {name: _policy_from_mapping(name, raw.get(name)) for name in _POLICIES}
    return ArchitectureConfig(kind=str(raw["kind"]), **policies)  # type: ignore[arg-type]


_PRESETS: dict[str, tuple[str, bool]] = {
    "dense": ("dense", False),
    "residual": ("residual", False),
    "convolutional": ("convolutional", False),
    "residual-convolutional": ("residual-convolutional", False),
    "wave": ("wave", False),
    "sequence": ("sequence", False),
    "geometric-sparse": ("geometric-sparse", False),
    "psann": ("dense", True),
    "respsann": ("residual", True),
    "res-psann": ("residual", True),
    "resconvpsann": ("residual-convolutional", True),
    "res-conv-psann": ("residual-convolutional", True),
    "waveresnet": ("wave", True),
    "wave-resnet": ("wave", True),
    "sgrpsann": ("sequence", True),
    "sgr-psann": ("sequence", True),
    "geosparse": ("geometric-sparse", True),
    "geo-sparse": ("geometric-sparse", True),
}


def normalize_architecture(value: ArchitectureLike) -> ArchitectureConfig:
    """Return one validated architecture object, preserving canonical identity."""

    if isinstance(value, ArchitectureConfig):
        return value
    if isinstance(value, Mapping):
        return _from_mapping(value)
    if not isinstance(value, str):
        raise TypeError("architecture must be an ArchitectureConfig, mapping, or string.")
    key = _canonical_name(value)
    target = _PRESETS.get(key)
    if target is None:
        raise ValueError(f"Unknown architecture {value!r}.")
    name, deprecated = target
    if deprecated:
        warnings.warn(
            f"architecture={value!r} is deprecated; use {name!r}.", DeprecationWarning, stacklevel=2
        )
    if name == "dense":
        return ArchitectureConfig.dense()
    if name == "residual":
        return ArchitectureConfig.dense(residual=ResidualConfig())
    if name == "convolutional":
        return ArchitectureConfig.convolutional()
    if name == "residual-convolutional":
        return ArchitectureConfig.convolutional(residual=ResidualConfig())
    if name == "wave":
        return ArchitectureConfig.for_wave()
    if name == "sequence":
        return ArchitectureConfig.for_sequence()
    return ArchitectureConfig.geometric_sparse()


def validate_architecture(value: ArchitectureConfig, *, hidden_layers: int | None) -> None:
    """Validate cross-policy capability constraints before any model is built."""

    kind = value.kind
    if kind not in {"dense", "convolutional", "wave", "sequence", "geometric-sparse"}:
        raise ValueError(
            "architecture.kind must be dense, convolutional, wave, sequence, or geometric-sparse."
        )
    if kind == "dense":
        if (
            value.convolution
            or value.context
            or value.spectral
            or value.wave
            or value.sequence
            or value.geometry
        ):
            raise ValueError("architecture.dense contains a policy unsupported by dense.")
        if value.residual is not None and value.state is not None:
            raise ValueError("architecture.dense cannot combine residual and state.")
    elif kind == "convolutional":
        if value.convolution is None:
            raise ValueError("architecture.convolution is required for convolutional.")
        if (
            value.state
            or value.context
            or value.spectral
            or value.wave
            or value.sequence
            or value.geometry
        ):
            raise ValueError("architecture.convolutional contains an unsupported policy.")
    elif kind == "wave":
        if value.residual is None or value.wave is None:
            raise ValueError("architecture.wave requires residual and wave policies.")
        if value.state or value.sequence or value.geometry:
            raise ValueError("architecture.wave contains an unsupported policy.")
        if value.attention and value.spectral:
            raise ValueError("architecture.wave cannot combine attention and spectral.")
        residual = value.residual
        if (
            residual.drop_path != 0
            or residual.norm != "rms"
            or residual.first_w0 != 12
            or residual.hidden_w0 != 1
        ):
            raise ValueError("architecture.wave residual policy may only customize alpha_init.")
    elif kind == "sequence":
        if value.sequence is None:
            raise ValueError("architecture.sequence requires sequence policy.")
        if value.activation.kind != "psann":
            raise ValueError("architecture.sequence requires activation.kind='psann'.")
        if any(
            (
                value.residual,
                value.convolution,
                value.attention,
                value.state,
                value.context,
                value.wave,
                value.geometry,
            )
        ):
            raise ValueError("architecture.sequence contains an unsupported policy.")
    else:
        if value.residual is None or value.geometry is None:
            raise ValueError(
                "architecture.geometric-sparse requires residual and geometry policies."
            )
        if any(
            (
                value.convolution,
                value.attention,
                value.state,
                value.context,
                value.wave,
                value.spectral,
                value.sequence,
            )
        ):
            raise ValueError("architecture.geometric-sparse contains an unsupported policy.")
        residual = value.residual
        if residual.first_w0 != 12 or residual.hidden_w0 != 1:
            raise ValueError("architecture.geometric-sparse does not support residual W0 values.")
    if (
        value.wave
        and value.wave.progressive_depth
        and hidden_layers is not None
        and value.wave.progressive_depth.initial_layers > hidden_layers
    ):
        raise ValueError("wave.progressive_depth.initial_layers cannot exceed hidden_layers.")


def _mapping_value(value: object) -> object:
    if hasattr(value, "__dataclass_fields__"):
        return {key: _mapping_value(item) for key, item in asdict(cast(Any, value)).items()}
    return _thaw(value)


def architecture_to_mapping(value: ArchitectureConfig) -> dict[str, object]:
    value = normalize_architecture(value)
    result: dict[str, object] = {"kind": value.kind, "activation": _mapping_value(value.activation)}
    for name in _POLICIES:
        if name == "activation":
            continue
        policy = getattr(value, name)
        if policy is not None:
            result[name] = _mapping_value(policy)
    return result


def replace_architecture_path(
    value: ArchitectureConfig, path: str, replacement: object, *, hidden_layers: int
) -> ArchitectureConfig:
    """Rebuild one frozen nested policy for sklearn ``set_params``."""

    pieces = path.split("__")
    if len(pieces) < 2 or pieces[0] != "architecture":
        raise ValueError(f"Unknown architecture parameter path {path!r}.")
    policy_name = pieces[1]
    if policy_name not in _POLICIES:
        raise ValueError(f"Unknown architecture policy {policy_name!r}.")
    if len(pieces) == 2:
        candidate = replace(
            value, **cast(Any, {policy_name: _policy_from_mapping(policy_name, replacement)})
        )
    else:
        policy = getattr(value, policy_name)
        if policy is None:
            raise ValueError(f"architecture.{policy_name} is absent; set that policy object first.")
        field_name = pieces[2]
        if len(pieces) != 3 or field_name not in {field.name for field in fields(policy)}:
            raise ValueError(f"Unknown architecture parameter path {path!r}.")
        candidate = replace(
            value,
            **cast(Any, {policy_name: replace(policy, **cast(Any, {field_name: replacement}))}),
        )
    validate_architecture(candidate, hidden_layers=hidden_layers)
    return candidate
