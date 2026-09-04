"""Strict configuration equivalence and rejection at the real build boundary."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import pytest

from psann.architectures import ActivationConfig, GeometryConfig, ResidualConfig, SpectralConfig
from psannlm.architectures import (
    LMActivationInitializationConfig,
    LMArchitectureConfig,
    LMConfig,
    LMGeometryExecutionConfig,
    LMTemporalConfig,
    build_lm_model,
    normalize_architecture,
    normalize_lm_config,
    to_mapping,
)
from psannlm.architectures.compat import legacy_lm_config

KINDS = ("transformer", "residual", "wave", "geometric-sparse")


@pytest.mark.parametrize("kind", KINDS)
def test_typed_nested_tagged_preset_equivalence_and_immutability(kind):
    arch = getattr(LMArchitectureConfig, kind.replace("-", "_"))()
    expected = LMConfig(arch, d_model=24, n_heads=3, n_layers=2, d_mlp=36, vocab_size=29)
    mapping = to_mapping(expected)
    before = deepcopy(mapping)
    assert normalize_lm_config(mapping) == expected
    assert normalize_lm_config(dict(mapping, architecture=arch)) == expected
    assert normalize_lm_config(dict(mapping, architecture=kind)) == expected
    assert normalize_architecture(to_mapping(arch)) == arch
    assert normalize_architecture(kind) == arch
    assert mapping == before
    with pytest.raises(FrozenInstanceError):
        expected.n_layers = 3
    with pytest.raises(FrozenInstanceError):
        arch.kind = "residual"


@pytest.mark.parametrize("name", ["amplitude", "frequency", "decay"])
@pytest.mark.parametrize("value", [0.0, 0.23])
def test_initialization_spread_typed_mapping(name, value):
    init = LMActivationInitializationConfig(**{name + "_std": value})
    arch = LMArchitectureConfig.residual(activation_initialization=to_mapping(init))
    assert arch.activation_initialization == init
    assert normalize_architecture(to_mapping(arch)) == arch
    with pytest.raises(FrozenInstanceError):
        init.amplitude_std = 0.2


@pytest.mark.parametrize("name", ["amplitude", "frequency", "decay"])
@pytest.mark.parametrize("pair", [(0.03, 0.8), (0.23, 0.23)])
def test_initialization_range_detaches_caller_lists(name, pair):
    raw = {name + "_range": list(pair)}
    config = LMArchitectureConfig.residual(activation_initialization=raw)
    raw[name + "_range"][0] = 99
    assert getattr(config.activation_initialization, name + "_range") == pair
    assert normalize_architecture(to_mapping(config)) == config


INVALID = []
for key in ("d_model", "n_heads", "n_layers", "d_mlp", "vocab_size"):
    INVALID += [(key, x) for x in (True, "24", 0, -1, 1.5)]
for key in ("dropout",):
    INVALID += [(key, x) for x in (True, "0.1", -0.1, 1.0, float("nan"), float("inf"))]
INVALID += [
    ("unknown", 1),
    ("kind", "wrong"),
    ("architecture.kind", "unknown"),
    ("architecture.unknown", 1),
    ("architecture.activation.unknown", 1),
    ("architecture.activation.kind", True),
]
INVALID += [("positional_encoding", x) for x in (False, "unknown")]
INVALID += [("attention_implementation", x) for x in (False, "unknown")]
for name in ("amplitude", "frequency", "decay"):
    INVALID += [
        (f"architecture.activation_initialization.{name}_std", x)
        for x in (True, "0.2", -0.2, float("nan"), float("inf"))
    ]
    INVALID += [
        (f"architecture.activation_initialization.{name}_range", x)
        for x in (
            True,
            "0,1",
            [1],
            [1, 2, 3],
            [True, 1],
            ["0", 1],
            [2, 1],
            [0, float("nan")],
            [0, float("inf")],
        )
    ]
for name in (
    "amplitude_init",
    "frequency_init",
    "decay_init",
    "slope_init",
    "clip_max",
    "phase_init",
    "ratio_sum_tol",
):
    INVALID += [(f"architecture.activation.{name}", x) for x in (True, "0.2", float("nan"))]
for name in ("alpha_init", "drop_path", "first_w0", "hidden_w0"):
    INVALID += [(f"architecture.residual.{name}", x) for x in (True, "0.2", float("inf"))]


def set_path(mapping, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        if mapping.get(key) is None:
            mapping[key] = {}
        mapping = mapping[key]
    mapping[keys[-1]] = value


@pytest.mark.parametrize("path,value", INVALID, ids=[p + "-" + repr(v) for p, v in INVALID])
def test_invalid_values_fail_before_builder(path, value, monkeypatch):
    from psannlm.architectures import registry

    reached = []
    monkeypatch.setitem(registry._BUILDERS, "residual", lambda r: reached.append(r))
    raw = to_mapping(
        LMConfig(LMArchitectureConfig.residual(), d_model=24, n_heads=3, vocab_size=29)
    )
    set_path(raw, path, value)
    before = deepcopy(raw)
    with pytest.raises((TypeError, ValueError), match=path.split(".")[-1]):
        build_lm_model(raw)
    assert reached == []
    # NaN is deliberately not compared by equality.
    assert repr(raw) == repr(before)


@pytest.mark.parametrize("name", ["amplitude", "frequency", "decay"])
def test_initialization_spread_range_conflict(name):
    with pytest.raises(ValueError, match=name + "_std conflicts"):
        LMActivationInitializationConfig(**{name + "_std": 0.2, name + "_range": (0.1, 0.3)})


@pytest.mark.parametrize(
    "kind,field,value",
    [
        (kind, name, val)
        for kind, forbidden in (
            ("transformer", ("residual", "spectral", "temporal", "geometry", "geometry_execution")),
            ("residual", ("temporal", "geometry", "geometry_execution")),
            ("wave", ("spectral", "geometry", "geometry_execution")),
            ("geometric-sparse", ("spectral", "temporal")),
        )
        for name, val in (
            ("residual", ResidualConfig()),
            ("spectral", SpectralConfig()),
            ("temporal", LMTemporalConfig()),
            ("geometry", GeometryConfig()),
            ("geometry_execution", LMGeometryExecutionConfig()),
        )
        if name in forbidden
    ],
)
def test_unsupported_policy_matrix(kind, field, value):
    raw = to_mapping(normalize_architecture(kind))
    raw[field] = value
    with pytest.raises(ValueError, match="architecture." + field):
        build_lm_model(dict(architecture=raw, d_model=24, n_heads=3, vocab_size=29))


@pytest.mark.parametrize(
    "kind,field",
    [
        (k, f)
        for k, fs in (
            ("transformer", ("activation",)),
            ("residual", ("activation", "residual")),
            ("wave", ("activation", "residual", "temporal")),
            ("geometric-sparse", ("activation", "residual", "geometry", "geometry_execution")),
        )
        for f in fs
    ],
)
def test_required_policy_explicit_none(kind, field):
    raw = to_mapping(normalize_architecture(kind))
    raw[field] = None
    with pytest.raises(ValueError, match="architecture." + field):
        normalize_architecture(raw)


@pytest.mark.parametrize("mode", ["disabled", "attention-only"])
@pytest.mark.parametrize(
    "field,value", [("kernel_size", 5), ("dilation_growth", 2), ("dropout", 0.2)]
)
def test_inactive_temporal_fields_rejected(mode, field, value):
    with pytest.raises(ValueError, match="temporal." + field):
        LMTemporalConfig(mode=mode, **{field: value})


@pytest.mark.parametrize("activation", ["gelu", "relu", "tanh"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("frequency_init", 1.7),
        ("learnable", ()),
        ("decay_mode", "none"),
        ("bounds", {"decay": (0.01, 0.2)}),
    ],
)
def test_fixed_activation_irrelevant_fields_rejected(activation, field, value):
    with pytest.raises(ValueError, match="activation." + field):
        LMArchitectureConfig.geometric_sparse(
            activation=ActivationConfig(kind=activation, **{field: value})
        )


@pytest.mark.parametrize(
    "kind,activation",
    [
        ("transformer", "psann"),
        ("residual", "relu"),
        ("wave", "relu"),
        ("geometric-sparse", "phase-psann"),
    ],
)
def test_activation_kind_rejections(kind, activation):
    with pytest.raises(ValueError, match="activation.kind"):
        replace(normalize_architecture(kind), activation=ActivationConfig(kind=activation))


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation_initialization", LMActivationInitializationConfig()),
        ("activation", ActivationConfig(decay_init=0.2)),
    ],
)
def test_attention_only_rejects_inactive_activation(field, value):
    with pytest.raises(ValueError, match="architecture." + field):
        LMArchitectureConfig.wave(
            temporal=LMTemporalConfig(mode="attention-only"), **{field: value}
        )


@pytest.mark.parametrize(
    "interleave,replace,mode",
    [
        (False, False, "disabled"),
        (True, False, "interleave"),
        (True, True, "replace"),
        (False, True, "attention-only"),
    ],
)
def test_legacy_wave_exact_normalized_truth_table(interleave, replace, mode):
    values = dict(
        vocab_size=29,
        d_model=24,
        n_heads=3,
        d_mlp=36,
        wave_interleave=interleave,
        wave_replace=replace,
    )
    with pytest.warns(DeprecationWarning) as found:
        config = legacy_lm_config("waveresnet", values)
    assert len(found) == 1
    assert found[0].filename == __file__
    assert config.architecture.temporal.mode == mode
    assert normalize_lm_config(to_mapping(config)) == config


@pytest.mark.parametrize("base", ["transformer", "respsann", "sgrpsann", "waveresnet", "geosparse"])
def test_legacy_dimensions_and_alias_warning(base):
    with pytest.warns(DeprecationWarning) as found:
        config = legacy_lm_config(
            base,
            dict(
                d_model=24,
                n_heads=3,
                n_layers=2,
                d_mlp=36,
                vocab_size=29,
                dropout=0.13,
                positional_encoding="alibi",
                attn_impl="sdpa",
            ),
        )
    assert len(found) == 1
    assert (
        config.d_model,
        config.n_heads,
        config.n_layers,
        config.d_mlp,
        config.vocab_size,
        config.dropout,
        config.positional_encoding,
        config.attention_implementation,
    ) == (24, 3, 2, 36, 29, 0.13, "alibi", "sdpa")
    assert (config.architecture.spectral is not None) == (base == "sgrpsann")


def test_missing_discriminators_and_unresolved_vocabulary():
    for raw, path in (
        ({}, "architecture"),
        ({"architecture": {}}, "architecture.kind"),
        ({"architecture": {"kind": "residual", "activation": {}}}, "activation.kind"),
    ):
        with pytest.raises((TypeError, ValueError), match=path):
            build_lm_model(raw)
    with pytest.raises(ValueError, match="vocab_size"):
        build_lm_model(LMConfig(LMArchitectureConfig.residual()))


@pytest.mark.parametrize(
    "values,path",
    [
        (dict(d_model=25), "d_model"),
        (dict(d_model=27), "n_heads"),
        (
            dict(
                d_mlp=35,
                architecture=LMArchitectureConfig.geometric_sparse(
                    geometry=GeometryConfig(shape=(4, 9))
                ),
            ),
            "geometry.shape",
        ),
    ],
)
def test_cross_dimension_rejections(values, path):
    raw = dict(architecture="residual", d_model=24, n_heads=3, vocab_size=29)
    raw.update(values)
    with pytest.raises(ValueError, match=path):
        build_lm_model(raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("d_model", 24),
        ("dropout", 0.0),
        ("attn_impl", "math"),
        ("sine_params", {"freq_init": 0.7}),
        ("wave_interleave", True),
    ],
)
def test_omitted_matching_conflicting_flat_values(field, value):
    config = LMConfig(
        LMArchitectureConfig.wave(
            activation=ActivationConfig(frequency_init=0.7, decay_init=0.01),
            temporal=LMTemporalConfig(mode="interleave"),
        ),
        d_model=24,
        n_heads=3,
        vocab_size=29,
    )
    assert normalize_lm_config(config) is config
    with pytest.warns(DeprecationWarning) as found:
        assert normalize_lm_config(config, **{field: value}) is config
    assert len(found) == 1
    conflicting = {
        "d_model": 48,
        "dropout": 0.2,
        "attn_impl": "sdpa",
        "sine_params": {"freq_init": 0.8},
        "wave_interleave": False,
    }[field]
    with pytest.raises(ValueError, match="flat." + field + ".*config."):
        normalize_lm_config(config, **{field: conflicting})


@pytest.mark.parametrize(
    "policy,field,value",
    [
        ("spectral", "k_fft", True),
        ("spectral", "k_fft", 0),
        ("spectral", "gate_type", "invalid"),
        ("spectral", "groups", "invalid"),
        ("spectral", "init", True),
        ("spectral", "strength", "0.2"),
        ("spectral", "strength", -0.2),
        ("temporal", "mode", "invalid"),
        ("temporal", "kernel_size", 4),
        ("temporal", "kernel_size", True),
        ("temporal", "dilation_growth", "2"),
        ("temporal", "dilation_growth", 0),
        ("temporal", "dropout", 1.0),
        ("geometry", "shape", [4, True]),
        ("geometry", "shape", [4, 9, 1]),
        ("geometry", "shape", "4x9"),
        ("geometry", "k", True),
        ("geometry", "k", "8"),
        ("geometry", "k", 0),
        ("geometry", "radius", -1),
        ("geometry", "pattern", "invalid"),
        ("geometry", "wrap_mode", "invalid"),
        ("geometry", "compute_mode", "invalid"),
        ("geometry", "bias", 1),
        ("geometry", "offsets", [[1]]),
        ("geometry", "offsets", [[True, 0]]),
        ("geometry", "offsets", []),
        ("geometry", "seed", True),
        ("geometry_execution", "depth", 0),
        ("geometry_execution", "depth", True),
        ("geometry_execution", "chunk_size", 0),
        ("geometry_execution", "chunk_size", "7"),
        ("activation_initialization", "unknown", 0.2),
    ],
)
def test_nested_policy_rejections_at_build(policy, field, value):
    kind = (
        "residual"
        if policy in {"spectral", "activation_initialization"}
        else "wave" if policy == "temporal" else "geometric-sparse"
    )
    config = to_mapping(
        LMConfig(normalize_architecture(kind), d_model=24, n_heads=3, vocab_size=29)
    )
    if config["architecture"][policy] is None:
        config["architecture"][policy] = {}
    config["architecture"][policy][field] = value
    with pytest.raises((TypeError, ValueError), match=policy + ".*" + field):
        build_lm_model(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("activation_types", ["psann", "psann"]),
        ("activation_types", ["psann", True]),
        ("activation_types", []),
        ("activation_ratios", [0.2, 0.2]),
        ("activation_ratios", [True, 0]),
        ("activation_ratios", ["0.4", 0.6]),
        ("activation_ratios", [-0.2, 1.2]),
        ("activation_ratios", [1]),
        ("mix_layout", "invalid"),
        ("mix_seed", True),
        ("feature_dim", 0),
    ],
)
def test_mixed_configuration_invalid_neighbors(field, value):
    activation = dict(
        kind="mixed", activation_types=["psann", "gelu"], activation_ratios=[0.4, 0.6]
    )
    activation[field] = value
    with pytest.raises((TypeError, ValueError), match="activation.*" + field):
        LMArchitectureConfig.geometric_sparse(activation=activation)
