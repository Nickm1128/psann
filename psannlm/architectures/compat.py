"""Explicit legacy translations; canonical configuration never filters keys."""

from __future__ import annotations


import inspect
import warnings
from dataclasses import fields, is_dataclass
from typing import Any, Mapping

from psann.architectures import ActivationConfig, GeometryConfig, ResidualConfig, SpectralConfig

from .config import (
    LMActivationInitializationConfig,
    LMArchitectureConfig,
    LMConfig,
    LMGeometryExecutionConfig,
    LMTemporalConfig,
)

BASE_KINDS = {
    "transformer": "transformer",
    "respsann": "residual",
    "sgrpsann": "residual",
    "waveresnet": "wave",
    "geosparse": "geometric-sparse",
}


def legacy_api_config(**values: Any) -> LMConfig:
    """Translate stored legacy benchmark/API options into the canonical model policy."""
    base = values.pop("base", "waveresnet")
    return legacy_lm_config(base, values, high_level=True, warn=False)


def check_flat_duplicates(config: LMConfig, flat: Mapping[str, Any]) -> None:
    """Compare only supplied legacy values, preserving omission/default distinctions."""
    from .config import normalize_architecture

    basic = {
        "d_model",
        "n_layers",
        "n_heads",
        "d_mlp",
        "vocab_size",
        "dropout",
        "positional_encoding",
    }
    paths = {name: name for name in basic}
    paths.update(
        attn_impl="attention_implementation", mlp_activation="architecture.activation.kind"
    )
    paths.update(
        {
            "geosparse_" + key: "architecture.geometry." + key
            for key in (
                "shape",
                "k",
                "pattern",
                "radius",
                "offsets",
                "wrap_mode",
                "bias",
                "compute_mode",
                "seed",
            )
        }
    )
    paths.update(
        {
            "geosparse_" + old: "architecture." + new
            for old, new in (
                ("depth", "geometry_execution.depth"),
                ("chunk_size", "geometry_execution.chunk_size"),
                ("norm", "residual.norm"),
                ("residual_alpha_init", "residual.alpha_init"),
                ("drop_path_max", "residual.drop_path"),
                ("activation", "activation.kind"),
                ("activation_types", "activation.activation_types"),
                ("activation_ratios", "activation.activation_ratios"),
                ("activation_layout", "activation.mix_layout"),
                ("activation_ratio_sum_tol", "activation.ratio_sum_tol"),
            )
        }
    )
    paths.update(
        {
            old: "architecture.spectral." + new
            for old, new in (
                ("k_fft", "k_fft"),
                ("gate_type", "gate_type"),
                ("gate_groups", "groups"),
                ("gate_init", "init"),
                ("gate_strength", "strength"),
            )
        }
    )
    paths.update(
        {
            "wave_" + old: "architecture.temporal." + new
            for old, new in (
                ("kernel_size", "kernel_size"),
                ("dilation_growth", "dilation_growth"),
                ("dropout", "dropout"),
            )
        }
    )
    known = set(paths) | {
        "base",
        "architecture",
        "sine",
        "sine_params",
        "rope",
        "wave_interleave",
        "wave_replace",
        "use_spectral_gate",
    }
    for key in flat:
        if key not in known:
            raise ValueError(f"flat.{key} is unknown.")
    base = {
        "transformer": "transformer",
        "residual": "sgrpsann" if config.architecture.spectral is not None else "respsann",
        "wave": "waveresnet",
        "geometric-sparse": "geosparse",
    }[config.architecture.kind]
    if "base" in flat and flat["base"] != base:
        raise ValueError("flat.base conflicts with config.architecture.kind/spectral.")
    if (
        "architecture" in flat
        and normalize_architecture(flat["architecture"]) != config.architecture
    ):
        raise ValueError("flat.architecture conflicts with config.architecture.")
    raw = {name: getattr(config, name) for name in basic}
    raw.update({k: v for k, v in flat.items() if k not in {"base", "architecture"}})
    candidate = legacy_lm_config(base, raw, warn=False)

    def get(value: object, path: str) -> Any:
        for name in path.split("."):
            value = getattr(value, name, None)
        return value

    compare_paths = {key: paths[key] for key in flat if key in paths}
    if "rope" in flat:
        compare_paths["rope"] = "positional_encoding"
    if {"wave_interleave", "wave_replace"} & flat.keys():
        compare_paths["wave_interleave/wave_replace"] = "architecture.temporal.mode"
    if "use_spectral_gate" in flat:
        if bool(flat["use_spectral_gate"]) != (config.architecture.spectral is not None):
            raise ValueError("flat.use_spectral_gate conflicts with config.architecture.spectral.")
    for key in ("sine", "sine_params"):
        if key not in flat:
            continue
        sine = flat[key]
        supplied = dict(sine) if isinstance(sine, Mapping) else vars(sine)
        for old, name in (("amp", "amplitude"), ("freq", "frequency"), ("damp", "decay")):
            for suffix, path in (
                ("_init", "activation." + name + "_init"),
                ("_init_std", "activation_initialization." + name + "_std"),
                ("_range", "activation_initialization." + name + "_range"),
                ("_bounds", "activation.bounds"),
            ):
                if old + suffix in supplied:
                    compare_paths[key + "." + old + suffix] = "architecture." + path
        for name, path in (
            ("trainable", "learnable"),
            ("learnable", "learnable"),
            ("decay_mode", "decay_mode"),
        ):
            if name in supplied:
                compare_paths[key + "." + name] = "architecture.activation." + path
    for key, path in compare_paths.items():
        left, right = get(candidate, path), get(config, path)
        if path.endswith("_std"):
            left, right = left or 0.0, right or 0.0
        if left != right:
            raise ValueError(f"flat.{key} conflicts with config.{path}.")
    compatibility_warning(
        "Flat LM configuration is deprecated; matching explicit values were normalized once."
    )


def compatibility_warning(message: str) -> None:
    frame = inspect.currentframe()
    level = 1
    while frame is not None and str(frame.f_globals.get("__name__", "")).startswith("psannlm"):
        level += 1
        frame = frame.f_back
    warnings.warn(message, DeprecationWarning, stacklevel=level)


def sine_policies(
    value: object = None,
) -> tuple[ActivationConfig, LMActivationInitializationConfig | None, list[str]]:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = dict(value)
    elif is_dataclass(value):
        raw = {f.name: getattr(value, f.name) for f in fields(value)}
    else:
        raise TypeError("sine must be a legacy configuration or mapping.")
    notes = []
    samples: dict[str, Any] = {}
    means: dict[str, Any] = {}
    bounds = {}
    for old, name, default in (
        ("amp", "amplitude", 1.0),
        ("freq", "frequency", 1.0),
        ("damp", "decay", 0.01),
    ):
        means[name + "_init"] = raw.get(old + "_init", default)
        std = raw.get(old + "_init_std", 0.0)
        rng = raw.get(old + "_range")
        if std <= 0:
            if std < 0:
                notes.append(f"{old}_init_std disabled as 0.0")
            std = 0.0
        if std > 0 and rng is not None:
            notes.append(f"{old}_range ignored because positive {old}_init_std wins")
            rng = None
        if rng is not None:
            rng = tuple(sorted(rng))
            if tuple(raw[old + "_range"]) != rng:
                notes.append(f"{old}_range endpoints sorted")
        samples[name + "_std"] = std
        samples[name + "_range"] = rng
        if raw.get(old + "_bounds") is not None:
            bounds[name] = raw[old + "_bounds"]
    learnable = raw.get("learnable")
    if learnable is None:
        learnable = ("amplitude", "frequency", "decay") if raw.get("trainable", True) else ()
    activation = ActivationConfig(
        **means,
        learnable=tuple(learnable),
        decay_mode=raw.get("decay_mode", "abs"),
        bounds=bounds or None,
    )
    init = LMActivationInitializationConfig(**samples)
    return activation, init if any(samples.values()) else None, notes


def legacy_lm_config(
    base: str,
    values: Mapping[str, Any] | None = None,
    *,
    high_level: bool = False,
    warn: bool = True,
) -> LMConfig:
    """Translate the exact historical factory or high-level argument surface."""
    key = base.strip().lower()
    if key not in BASE_KINDS:
        raise KeyError(f"Unknown base {base!r}. Available: {tuple(BASE_KINDS)}")
    raw = dict(values or {})
    notes: list[str] = []
    dims: dict[str, Any] = {}
    default: Any
    for name, default in (("d_model", 512), ("n_layers", 8), ("n_heads", 8), ("vocab_size", None)):
        dims[name] = raw.pop(name, default)
    dims["d_mlp"] = raw.pop("d_mlp", None if high_level else 2048)
    dims["dropout"] = raw.pop("dropout", 0.0)
    position = raw.pop("positional_encoding", None)
    rope = raw.pop("rope", None)
    if position is None or (rope is not None and not high_level):
        position = "sinusoidal" if rope is False else "rope"
    dims["positional_encoding"] = str(position).strip().lower()
    dims["attention_implementation"] = raw.pop("attn_impl", "math") if not high_level else "math"
    act_name = raw.pop("mlp_activation", "sine")
    sine = raw.pop("sine", raw.pop("sine_params", None))
    if high_level:
        # This surface historically forwarded only dimensions, dropout and MLP activation.
        if raw:
            notes.append("inactive high-level options ignored: " + ", ".join(sorted(raw)))
        raw.clear()
        if sine is not None:
            src = dict(sine) if isinstance(sine, Mapping) else vars(sine)
            active = {
                "amp_init",
                "freq_init",
                "damp_init",
                "amp_init_std",
                "freq_init_std",
                "damp_init_std",
                "trainable",
            }
            if set(src) - active:
                notes.append("inactive high-level sine options ignored")
            sine = {k: v for k, v in src.items() if k in active}
    kind = BASE_KINDS[key]
    inactive_wave = (
        kind == "wave" and not raw.get("wave_interleave", False) and raw.get("wave_replace", False)
    )
    fixed = (
        kind == "transformer"
        or (kind in {"residual", "wave"} and act_name == "gelu")
        or (
            kind == "geometric-sparse"
            and raw.get("geosparse_activation", "psann") in {"gelu", "relu", "tanh"}
        )
    )
    if fixed or inactive_wave:
        if sine is not None:
            notes.append("inactive sine policy ignored")
        sine = None
    if inactive_wave:
        act_name = "sine"
    act, initialization, sine_notes = sine_policies(sine)
    notes.extend(sine_notes)
    architecture: dict[str, Any] = {"kind": kind}
    if kind == "transformer":
        if act_name not in {"gelu", "relu"}:
            notes.append(f"mlp_activation={act_name!r} coerced to GELU")
            act_name = "gelu"
        act, initialization = ActivationConfig(kind=act_name), None
        ignored = {
            "use_spectral_gate",
            "k_fft",
            "gate_type",
            "gate_groups",
            "gate_init",
            "gate_strength",
            "wave_interleave",
            "wave_replace",
            "wave_kernel_size",
            "wave_dilation_growth",
            "wave_dropout",
        }
        for name in sorted(ignored & raw.keys()):
            raw.pop(name)
            notes.append(f"{name} ignored by transformer")
    elif kind in {"residual", "wave"}:
        if act_name in {"sine", "psann"}:
            pass
        elif act_name == "gelu":
            act, initialization = ActivationConfig(kind="gelu"), None
        else:
            raise ValueError(f"mlp_activation={act_name!r} is unsupported by {key}.")
        architecture["residual"] = ResidualConfig(alpha_init=1.0)
        if kind == "residual":
            spectral = raw.pop("use_spectral_gate", key == "sgrpsann")
            gate = {}
            for old, new, default in (
                ("k_fft", "k_fft", 64),
                ("gate_type", "gate_type", "rfft"),
                ("gate_groups", "groups", "depthwise"),
                ("gate_init", "init", 0.0),
                ("gate_strength", "strength", 1.0),
            ):
                gate[new] = raw.pop(old, default)
            if spectral:
                architecture["spectral"] = SpectralConfig(**gate)
            elif gate != {
                "k_fft": 64,
                "gate_type": "rfft",
                "groups": "depthwise",
                "init": 0.0,
                "strength": 1.0,
            }:
                notes.append("inactive spectral options ignored")
        else:
            interleave, replace = raw.pop("wave_interleave", False), raw.pop("wave_replace", False)
            mode = {
                (False, False): "disabled",
                (True, False): "interleave",
                (True, True): "replace",
                (False, True): "attention-only",
            }[(interleave, replace)]
            temporal = {
                "kernel_size": raw.pop("wave_kernel_size", 3),
                "dilation_growth": raw.pop("wave_dilation_growth", 1),
                "dropout": raw.pop("wave_dropout", 0.0),
            }
            if mode in {"disabled", "attention-only"}:
                if temporal != {"kernel_size": 3, "dilation_growth": 1, "dropout": 0.0}:
                    notes.append("inactive temporal kernel/dilation/dropout ignored")
                temporal = {}
            if mode == "attention-only":
                act, initialization = ActivationConfig(decay_init=0.01), None
                notes.append(
                    "attention-only retains no MLP or temporal activation; inactive activation options ignored"
                )
            architecture["temporal"] = LMTemporalConfig(mode=mode, **temporal)
    else:
        import re

        shape = raw.pop("geosparse_shape", None)
        if isinstance(shape, str):
            match = re.fullmatch(r"\s*(\d+)\s*[x,]\s*(\d+)\s*", shape)
            if match is None:
                raise ValueError("geosparse_shape must be a pair or HxW.")
            shape = tuple(map(int, match.groups()))
        geometry: dict[str, Any] = {"shape": shape}
        for name in (
            "k",
            "pattern",
            "radius",
            "offsets",
            "wrap_mode",
            "bias",
            "compute_mode",
            "seed",
        ):
            if "geosparse_" + name in raw:
                geometry[name] = raw.pop("geosparse_" + name)
        architecture["geometry"] = GeometryConfig(**geometry)
        chunk = raw.pop("geosparse_chunk_size", 32)
        architecture["geometry_execution"] = LMGeometryExecutionConfig(
            depth=raw.pop("geosparse_depth", 1), chunk_size=None if chunk == 0 else chunk
        )
        drop_path = raw.pop("geosparse_drop_path_max", 0.0)
        if architecture["geometry_execution"].depth == 1 and drop_path != 0:
            notes.append("geosparse_drop_path_max ignored by legacy depth=1")
            drop_path = 0.0
        architecture["residual"] = ResidualConfig(
            norm=raw.pop("geosparse_norm", "rms"),
            alpha_init=raw.pop("geosparse_residual_alpha_init", 1.0),
            drop_path=drop_path,
        )
        name = raw.pop("geosparse_activation", "psann")
        if name == "sine":
            name = "psann"
        extra = {
            "activation_types": raw.pop("geosparse_activation_types", None),
            "activation_ratios": raw.pop("geosparse_activation_ratios", None),
            "ratio_sum_tol": raw.pop("geosparse_activation_ratio_sum_tol", 1e-3),
            "mix_layout": raw.pop("geosparse_activation_layout", "random"),
            "mix_seed": geometry.get("seed"),
        }
        if name == "mixed":
            from dataclasses import replace

            act = replace(act, kind="mixed", **extra)
        elif name != "psann":
            act, initialization = ActivationConfig(kind=name), None
        # The old geosparse factory accepted arbitrary shared-harness keys.
        if raw:
            notes.append("inactive geosparse options ignored: " + ", ".join(sorted(raw)))
            raw.clear()
    if raw:
        raise ValueError(f"legacy.{sorted(raw)[0]} is unknown for {key}.")
    if act.kind == "mixed":
        from dataclasses import replace
        from psann.architectures.components import activation_feature_counts

        if (
            activation_feature_counts(act, features=dims["d_mlp"] or 4 * dims["d_model"]).get(
                "psann", 0
            )
            == 0
        ):
            defaults = ActivationConfig()
            inactive = (
                "amplitude_init",
                "frequency_init",
                "decay_init",
                "learnable",
                "bounds",
                "decay_mode",
            )
            act = replace(act, **{name: getattr(defaults, name) for name in inactive})
            initialization = None
            notes.append("inactive zero-width PSANN child policy ignored")
    architecture.update(activation=act, activation_initialization=initialization)
    result = LMConfig(architecture=LMArchitectureConfig(**architecture), **dims)
    if warn:
        compatibility_warning(
            f"{base} is a legacy compatibility base; use canonical LMConfig. " + "; ".join(notes)
        )
    return result
