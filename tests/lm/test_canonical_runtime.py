"""Executed canonical policy effects and seeded legacy numerical parity."""

from dataclasses import replace
import random

import pytest
import torch
from torch import nn

from psann.architectures import ActivationConfig, GeometryConfig, ResidualConfig, SpectralConfig
from psannlm.architectures import (
    LMArchitectureConfig,
    LMConfig,
    LMGeometryExecutionConfig,
    build_lm_model,
    to_mapping,
)
from psannlm.architectures.compat import legacy_lm_config
from psannlm.lm.models.sine import SineConfig
from psannlm.lm.models.transformer_respsann import ResPSANNTransformer, ResPSANNTransformerConfig
from psannlm.lm.models.transformer_vanilla import VanillaTransformer, VanillaTransformerConfig
from psannlm.lm.models.transformer_waveresnet import (
    WaveResNetTransformer,
    WaveResNetTransformerConfig,
)
from psannlm.lm.models.transformer_geosparse import GeoSparseTransformer, GeoSparseTransformerConfig
from test_legacy_routes import TOKENS, options


def build(architecture, **values):
    return build_lm_model(
        LMConfig(
            architecture,
            **dict(dict(d_model=24, n_heads=3, n_layers=2, d_mlp=36, vocab_size=29), **values),
        )
    ).model


def seed():
    torch.manual_seed(197)
    random.seed(197)


def compare(actual, expected):
    assert actual.state_dict().keys() == expected.state_dict().keys()
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)
    actual.eval()
    expected.eval()
    a, b = actual(TOKENS), expected(TOKENS)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    for (_, p), (_, q) in zip(actual.named_parameters(), expected.named_parameters()):
        assert p.requires_grad == q.requires_grad
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "base,cls,config_cls",
    [
        ("transformer", VanillaTransformer, VanillaTransformerConfig),
        ("respsann", ResPSANNTransformer, ResPSANNTransformerConfig),
        ("sgrpsann", ResPSANNTransformer, ResPSANNTransformerConfig),
        ("waveresnet", WaveResNetTransformer, WaveResNetTransformerConfig),
        ("geosparse", GeoSparseTransformer, GeoSparseTransformerConfig),
    ],
)
def test_all_base_canonical_mapping_matches_direct_low_level_runtime(base, cls, config_cls):
    raw = options(base)
    if base == "sgrpsann":
        raw["use_spectral_gate"] = True
    normalized = legacy_lm_config(base, raw, warn=False)
    seed()
    expected = cls(config_cls(**raw))
    expected_rng = torch.get_rng_state().clone(), random.getstate()
    seed()
    actual = build_lm_model(to_mapping(normalized)).model
    assert torch.equal(torch.get_rng_state(), expected_rng[0])
    assert random.getstate() == expected_rng[1]
    compare(actual, expected)
    assert actual.lm_config == normalized


@pytest.mark.parametrize(
    "interleave,replace_mode,mode",
    [
        (False, False, "disabled"),
        (True, False, "interleave"),
        (True, True, "replace"),
        (False, True, "attention-only"),
    ],
)
@pytest.mark.parametrize("mapping", [False, True])
def test_wave_four_structures_real_builder_forward_backward(
    interleave, replace_mode, mode, mapping
):
    raw = dict(
        vocab_size=29,
        d_model=24,
        n_heads=3,
        n_layers=2,
        d_mlp=36,
        wave_interleave=interleave,
        wave_replace=replace_mode,
    )
    config = legacy_lm_config("waveresnet", raw, warn=False)
    seed()
    expected = WaveResNetTransformer(WaveResNetTransformerConfig(**raw))
    seed()
    actual = build_lm_model(to_mapping(config) if mapping else config).model
    compare(actual, expected)
    assert actual.lm_config.architecture.temporal.mode == mode
    keys = actual.state_dict()
    assert any(".mlp." in key for key in keys) == (not replace_mode)
    assert any(".wave." in key for key in keys) == interleave
    block = actual.blocks[0]
    x = torch.linspace(-1.7, 2.3, 336).reshape(2, 7, 24).requires_grad_()
    attention = x + block.attn(block.norm1(x))
    if mode == "attention-only":
        equation = attention
    elif mode == "replace":
        equation = block.wave(block.norm2(attention))
    else:
        equation = attention + block.alpha * block.mlp(block.norm2(attention))
        if interleave:
            equation = block.wave(equation)
    torch.testing.assert_close(block(x), equation, rtol=0, atol=0)
    block(x).square().sum().backward()
    assert torch.count_nonzero(x.grad) > 1


@pytest.mark.parametrize("field", ["amp", "freq", "damp"])
@pytest.mark.parametrize(
    "sampling,value",
    [
        ("init_std", 0.23),
        ("init_std", 0.0),
        ("init_std", -0.23),
        ("range", (0.13, 0.8)),
        ("range", (0.8, 0.13)),
        ("range", (0.31, 0.31)),
    ],
)
def test_sampling_seeded_parameters_logits_gradients_and_rng_order(field, sampling, value):
    sine = SineConfig(
        amp_init=1.3, freq_init=0.8, damp_init=0.04, **{field + "_" + sampling: value}
    )
    raw = dict(vocab_size=29, d_model=24, n_heads=3, n_layers=2, d_mlp=36, sine=sine)
    config = legacy_lm_config("respsann", raw, warn=False)
    seed()
    expected = ResPSANNTransformer(ResPSANNTransformerConfig(**raw))
    rng = torch.get_rng_state().clone(), random.getstate()
    seed()
    actual = build_lm_model(config).model
    assert torch.equal(torch.get_rng_state(), rng[0])
    assert random.getstate() == rng[1]
    compare(actual, expected)


@pytest.mark.parametrize(
    "kind,activation",
    [
        ("transformer", "gelu"),
        ("transformer", "relu"),
        ("residual", "psann"),
        ("residual", "gelu"),
        ("wave", "psann"),
        ("wave", "gelu"),
        ("geometric-sparse", "psann"),
        ("geometric-sparse", "gelu"),
        ("geometric-sparse", "relu"),
        ("geometric-sparse", "tanh"),
        ("geometric-sparse", "mixed"),
    ],
)
def test_activation_capability_reaches_forward_and_optimizer(kind, activation):
    if activation == "mixed":
        policy = ActivationConfig(
            kind="mixed",
            activation_types=("psann", "gelu"),
            activation_ratios=(0.4, 0.6),
            frequency_init=0.7,
            mix_seed=31,
        )
    else:
        policy = ActivationConfig(kind=activation)
    architecture = getattr(LMArchitectureConfig, kind.replace("-", "_"))(activation=policy)
    seed()
    model = build(architecture)
    before = model.embed.weight.detach().clone()
    logits = model(TOKENS)
    assert logits.shape == (2, 7, 29)
    nn.functional.cross_entropy(logits.flatten(0, 1), TOKENS.flatten()).backward()
    assert torch.count_nonzero(model.embed.weight.grad) > 10
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    torch.optim.AdamW(model.parameters(), lr=0.013).step()
    assert not torch.equal(before, model.embed.weight)
    activations = [m for m in model.modules() if isinstance(m, (nn.GELU, nn.ReLU, nn.Tanh))]
    if activation in {"gelu", "relu", "tanh", "mixed"}:
        expected_type = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh, "mixed": nn.GELU}[
            activation
        ]
        assert any(isinstance(m, expected_type) for m in activations)


@pytest.mark.parametrize("pattern", ["local", "random", "hash"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("radius", 2),
        ("offsets", ((0, 0), (2, -1), (-2, 2))),
        ("wrap_mode", "wrap"),
        ("seed", 67),
        ("k", 11),
    ],
)
def test_geometry_candidate_policies_change_executed_sparse_path(pattern, field, value):
    geometry = GeometryConfig(shape=(4, 9), pattern=pattern, k=13, seed=31)
    architecture = LMArchitectureConfig.geometric_sparse(
        geometry=geometry, geometry_execution=LMGeometryExecutionConfig(depth=2, chunk_size=7)
    )
    seed()
    expected = build(architecture)
    seed()
    actual = build(replace(architecture, geometry=replace(geometry, **{field: value})))
    p, q = expected.blocks[0].mlp.blocks[0].fc1, actual.blocks[0].mlp.blocks[0].fc1
    if pattern == "local" and field == "seed":
        # Local connectivity has deterministic candidate order for every seed.
        torch.testing.assert_close(p.in_index_per_out, q.in_index_per_out, rtol=0, atol=0)
        compare(actual, expected)
        return
    assert not torch.equal(p.in_index_per_out, q.in_index_per_out)
    x, y = expected(TOKENS), actual(TOKENS)
    assert not torch.allclose(x, y)
    x.square().mean().backward()
    y.square().mean().backward()
    assert torch.count_nonzero(q.weight.grad) > 1


@pytest.mark.parametrize("kind", ["residual", "wave", "geometric-sparse"])
@pytest.mark.parametrize("norm", ["rms", "layer", "none"])
def test_nondefault_residual_alpha_norm_and_drop_path_execute(kind, norm):
    architecture = getattr(LMArchitectureConfig, kind.replace("-", "_"))(
        residual=ResidualConfig(norm=norm, alpha_init=0.37, drop_path=0.4)
    )
    if kind == "geometric-sparse":
        architecture = replace(architecture, geometry_execution=LMGeometryExecutionConfig(depth=2))
    seed()
    actual = build(architecture)
    seed()
    zero = build(replace(architecture, residual=replace(architecture.residual, alpha_init=0.0)))
    seed()
    a = actual(TOKENS)
    seed()
    b = zero(TOKENS)
    assert not torch.allclose(a, b)
    a.square().mean().backward()
    alpha_params = [p for name, p in actual.named_parameters() if name.endswith("alpha")]
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in alpha_params)
    observed = []
    from psann.architectures.components import DropPath

    handles = [
        m.register_forward_hook(
            lambda m, args, out: observed.append((args[0].detach().clone(), out.detach().clone()))
        )
        for m in actual.modules()
        if isinstance(m, DropPath)
    ]
    torch.manual_seed(31)
    actual(TOKENS)
    assert observed and any(not torch.equal(x, y) for x, y in observed)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("gate_type", ["rfft", "fourier-features"])
@pytest.mark.parametrize("groups", ["depthwise", "full"])
def test_spectral_is_residual_policy_with_observable_gate_gradients(gate_type, groups):
    architecture = LMArchitectureConfig.residual(
        spectral=SpectralConfig(
            k_fft=5, gate_type=gate_type, groups=groups, init=0.23, strength=0.4
        )
    )
    seed()
    actual = build(architecture)
    seed()
    disabled = build(replace(architecture, spectral=replace(architecture.spectral, strength=0.0)))
    a, b = actual(TOKENS), disabled(TOKENS)
    assert not torch.allclose(a, b)
    a.square().mean().backward()
    assert actual.lm_capabilities.kind == "residual"
    assert any(
        "spectral" in name and p.grad is not None and torch.count_nonzero(p.grad)
        for name, p in actual.named_parameters()
    )


def test_registry_duplicate_and_visible_builder_failure(monkeypatch):
    from psannlm.architectures import registry

    original = registry._BUILDERS["residual"]
    with pytest.raises(ValueError, match="already registered"):
        registry.register_lm_builder("residual", original)

    def broken(request):
        raise ImportError("visible built-in failure")

    monkeypatch.setitem(registry._BUILDERS, "residual", original)
    registry.register_lm_builder("residual", broken, replace=True)
    with pytest.raises(ImportError, match="visible built-in failure"):
        build(LMArchitectureConfig.residual())


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.04])
def test_legacy_zero_width_mixed_child_preserves_state_logits_gradients_and_rng(ratio):
    raw = dict(
        d_model=24,
        n_heads=3,
        n_layers=2,
        d_mlp=36,
        vocab_size=29,
        geosparse_activation="mixed",
        geosparse_activation_types=("psann", "gelu"),
        geosparse_activation_ratios=(ratio, 1 - ratio),
        sine=SineConfig(freq_init=0.73, amp_init_std=0.23),
    )
    seed()
    reference = GeoSparseTransformer(GeoSparseTransformerConfig(**raw))
    torch_rng, python_rng = torch.get_rng_state(), random.getstate()
    seed()
    with pytest.warns(DeprecationWarning) as warnings:
        normalized = legacy_lm_config("geosparse", raw)
    actual = build_lm_model(normalized).model
    assert len(warnings) == 1
    assert torch.equal(torch.get_rng_state(), torch_rng) and random.getstate() == python_rng
    compare(actual, reference)
    assert (normalized.architecture.activation_initialization is None) == (ratio < 0.04)


@pytest.mark.parametrize("drop_path", [0.0, 0.37])
def test_single_depth_geometric_drop_path_legacy_identity_and_canonical_runtime(drop_path):
    raw = dict(
        d_model=24,
        n_heads=3,
        n_layers=2,
        d_mlp=36,
        vocab_size=29,
        geosparse_depth=1,
        geosparse_drop_path_max=drop_path,
    )
    seed()
    reference = GeoSparseTransformer(GeoSparseTransformerConfig(**raw))
    seed()
    normalized = legacy_lm_config("geosparse", raw, warn=False)
    actual = build_lm_model(normalized).model
    compare(actual, reference)
    assert normalized.architecture.residual.drop_path == 0.0
    architecture = replace(
        normalized.architecture,
        residual=replace(normalized.architecture.residual, drop_path=drop_path),
    )
    seed()
    active = build_lm_model(replace(normalized, architecture=architecture)).model.train()
    actual.train()
    seed()
    a = active(TOKENS)
    seed()
    b = actual(TOKENS)
    if drop_path:
        assert not torch.equal(a, b)
    else:
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().mean().backward()
    assert torch.count_nonzero(active.lm_head.weight.grad) > 10


@pytest.mark.parametrize("field", ["amp", "freq", "damp"])
@pytest.mark.parametrize("spread", [-0.23, 0.0, 0.23])
def test_legacy_spread_range_precedence_preserves_parameters_logits_gradients_and_rng(
    field, spread
):
    sine = SineConfig(
        amp_init=1.3,
        freq_init=0.8,
        damp_init=0.04,
        **{field + "_init_std": spread, field + "_range": (0.8, 0.13)},
    )
    raw = dict(vocab_size=29, d_model=24, n_heads=3, n_layers=2, d_mlp=36, sine=sine)
    seed()
    expected = ResPSANNTransformer(ResPSANNTransformerConfig(**raw))
    rng = torch.get_rng_state().clone(), random.getstate()
    seed()
    with pytest.warns(DeprecationWarning) as warnings:
        config = legacy_lm_config("respsann", raw)
    assert len(warnings) == 1 and warnings[0].filename == __file__
    assert ("wins" in str(warnings[0].message)) == (spread > 0)
    actual = build_lm_model(config).model
    assert torch.equal(torch.get_rng_state(), rng[0]) and random.getstate() == rng[1]
    compare(actual, expected)
    name = {"amp": "amplitude", "freq": "frequency", "damp": "decay"}[field]
    init = config.architecture.activation_initialization
    assert getattr(init, name + "_std") == max(spread, 0)
    assert getattr(init, name + "_range") == (None if spread > 0 else (0.13, 0.8))
