"""Executable references for retained LM factories and checkpoint formats."""

from dataclasses import asdict, replace
from pathlib import Path
import random

import pytest
import torch
from torch import nn

from psannlm.lm import psannLM, psannLMDataPrep
from psannlm.lm.config import TrainConfig
from psannlm.lm.models.registry import get_base
from psannlm.lm.models.sine import SineConfig
from psannlm.lm.train.trainer import Trainer

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_models"
BASES = ("transformer", "respsann", "sgrpsann", "waveresnet", "geosparse")
TOKENS = torch.tensor([[1, 4, 7, 10, 13, 16, 19], [2, 5, 8, 11, 14, 17, 20]])


def options(base):
    common = dict(vocab_size=29, d_model=24, n_layers=2, n_heads=3, d_mlp=36)
    common.update(positional_encoding="alibi", attn_impl="math")
    if base != "transformer":
        common["sine"] = SineConfig(
            amp_init=1.3, freq_init=0.7, damp_init=0.04, learnable=("frequency",)
        )
    if base == "sgrpsann":
        common.update(
            k_fft=5,
            gate_type="fourier_features",
            gate_groups="full",
            gate_init=0.3,
            gate_strength=0.4,
        )
    if base == "waveresnet":
        common.update(wave_interleave=True, wave_kernel_size=5, wave_dilation_growth=2)
    if base == "geosparse":
        common.update(
            geosparse_shape=(4, 9),
            geosparse_depth=2,
            geosparse_k=5,
            geosparse_pattern="random",
            geosparse_wrap_mode="wrap",
            geosparse_norm="layer",
            geosparse_residual_alpha_init=0.6,
            geosparse_bias=False,
            geosparse_compute_mode="scatter",
            geosparse_seed=43,
            geosparse_chunk_size=7,
            geosparse_activation="mixed",
            geosparse_activation_types=["psann", "relu"],
            geosparse_activation_ratios=[0.4, 0.6],
        )
    return common


def seeded_model(base, **overrides):
    torch.manual_seed(137)
    random.seed(137)
    return get_base(base)(**(options(base) | overrides))


@pytest.mark.parametrize("base", BASES)
def test_factory_reference_state_logits_and_backward(base):
    reference = torch.load(FIXTURES / f"{base}.pt", weights_only=True)
    model = seeded_model(base).eval()
    assert model.state_dict().keys() == reference["state"].keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference["state"][key], rtol=0, atol=0)
    logits = model(TOKENS)
    torch.testing.assert_close(logits, reference["logits"], rtol=1e-5, atol=1e-6)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), TOKENS.roll(-1, 1).flatten())
    loss.backward()
    torch.testing.assert_close(loss, reference["loss"], rtol=1e-6, atol=1e-6)
    assert dict(model.named_parameters()).keys() == reference["gradients"].keys()
    for key, parameter in model.named_parameters():
        expected = reference["gradients"][key]
        if expected is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("base", BASES)
def test_factory_reference_cached_prefill_and_next_token(base):
    reference = torch.load(FIXTURES / f"{base}.pt", weights_only=True)
    model = seeded_model(base).eval()
    with torch.no_grad():
        prefill, kvs = model(TOKENS, use_cache=True)
        next_logits, extended = model(TOKENS[:, :1], use_cache=True, past_kvs=kvs)
    torch.testing.assert_close(prefill, reference["prefill"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(next_logits, reference["next_logits"], rtol=1e-5, atol=1e-6)
    for (k, v), (expected_k, expected_v) in zip(extended, reference["extended"]):
        assert k.shape == v.shape == (2, 3, 8, 8)
        torch.testing.assert_close(k, expected_k, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(v, expected_v, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("position", ["rope", "alibi", "sinusoidal"])
@pytest.mark.parametrize("attention", ["math", "sdpa", "auto"])
def test_attention_and_checkpointed_backward_reference(base, position, attention):
    reference = seeded_model(base, positional_encoding=position, attn_impl="math")
    actual = seeded_model(base, positional_encoding=position, attn_impl=attention)
    actual.enable_gradient_checkpointing(True)
    expected_logits = reference(TOKENS)
    actual_logits = actual(TOKENS)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-4, atol=2e-6)
    expected_logits.square().mean().backward()
    actual_logits.square().mean().backward()
    for p, q in zip(reference.parameters(), actual.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("base", BASES)
def test_high_level_old_payload_exact_state_and_generation(base, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data = psannLMDataPrep(["abc def ghij", "klmn opq rst uvw xyz"], tokenizer="simple")
    # The reference preserves metadata, including options ignored by this old API.
    model = psannLM.load(str(FIXTURES / f"{base}_high.pt"))
    model._tokenizer = data.tokenizer
    old = torch.load(FIXTURES / f"{base}_high.pt", weights_only=True)
    for key, value in model._model.state_dict().items():
        torch.testing.assert_close(value, old["state_dict"][key], rtol=0, atol=0)
    out = tmp_path / "copy.pt"
    model.save(str(out))
    restored = psannLM.load(str(out))
    restored._tokenizer = data.tokenizer
    torch.manual_seed(73)
    expected = model.generate("abc", max_new_tokens=9, temperature=0.7)
    torch.manual_seed(73)
    assert restored.generate("abc", max_new_tokens=9, temperature=0.7) == expected
    torch.testing.assert_close(restored._model(TOKENS), model._model(TOKENS), rtol=0, atol=0)


@pytest.mark.parametrize(
    "interleave,replace", [(False, False), (True, False), (False, True), (True, True)]
)
def test_wave_boolean_truth_table_at_block_boundary(interleave, replace):
    model = seeded_model("waveresnet", wave_interleave=interleave, wave_replace=replace)
    model.eval()
    block = model.blocks[0]
    x = torch.linspace(-1.7, 2.3, 2 * 7 * 24).reshape(2, 7, 24).requires_grad_()
    after_attention = x + block.attn(block.norm1(x))
    if replace:
        expected = block.wave(block.norm2(after_attention)) if interleave else after_attention
    else:
        expected = after_attention + block.alpha * block.mlp(block.norm2(after_attention))
        if interleave:
            expected = block.wave(expected)
    actual = block(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    assert torch.count_nonzero(x.grad) == x.numel()
    state = block.state_dict()
    assert any(key.startswith("mlp.") for key in state) == (not replace)
    assert any(key.startswith("wave.") for key in state) == interleave
    if replace and not interleave:
        assert block.alpha.grad is None
        torch.testing.assert_close(actual, after_attention, rtol=0, atol=0)
    else:
        assert not torch.allclose(actual, after_attention)


@pytest.mark.parametrize("chunk", [0, 1, 7, 36, 80])
@pytest.mark.parametrize("compute", ["gather", "scatter"])
def test_geosparse_chunk_disable_and_partition_parity(chunk, compute):
    reference = seeded_model(
        "geosparse", geosparse_chunk_size=0, geosparse_compute_mode=compute
    ).eval()
    chunked = seeded_model(
        "geosparse", geosparse_chunk_size=chunk, geosparse_compute_mode=compute
    ).eval()
    a, b = reference(TOKENS), chunked(TOKENS)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    a.square().mean().backward()
    b.square().mean().backward()
    for p, q in zip(reference.parameters(), chunked.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "base,ignored",
    [
        (
            "transformer",
            {"sine": SineConfig(amp_init=3.0), "wave_interleave": True, "use_spectral_gate": True},
        ),
        ("geosparse", {"unrecognized_option": 7, "wave_replace": True}),
    ],
)
def test_legacy_factory_ignored_options_have_exact_runtime_parity(base, ignored):
    expected = seeded_model(base).eval()
    actual = seeded_model(base, **ignored).eval()
    torch.testing.assert_close(actual(TOKENS), expected(TOKENS), rtol=0, atol=0)
    assert actual.state_dict().keys() == expected.state_dict().keys()


@pytest.mark.parametrize("base", BASES)
def test_high_level_ignored_variant_options_reach_default_runtime(base):
    kwargs = dict(base=base, vocab_size=29, d_model=24, n_layers=2, n_heads=3, d_mlp=36)
    torch.manual_seed(91)
    expected = psannLM(**kwargs)._ensure_model(29).eval()
    torch.manual_seed(91)
    actual = (
        psannLM(
            **kwargs,
            wave_interleave=True,
            wave_replace=True,
            k_fft=3,
            geosparse_depth=3,
            attn_impl="sdpa",
        )
        ._ensure_model(29)
        .eval()
    )
    torch.testing.assert_close(actual(TOKENS), expected(TOKENS), rtol=0, atol=0)
    assert actual.state_dict().keys() == expected.state_dict().keys()


@pytest.mark.parametrize("field", ["amp_init_std", "freq_init_std", "damp_init_std"])
def test_sine_initialization_spread_changes_parameters_logits_and_gradients(field):
    standard = seeded_model("respsann", sine=SineConfig()).eval()
    varied = seeded_model("respsann", sine=SineConfig(**{field: 0.2})).eval()
    assert not torch.equal(standard(TOKENS), varied(TOKENS))
    parameter_name = {"amp_init_std": "_A", "freq_init_std": "_f", "damp_init_std": "_d"}[field]
    p = getattr(varied.blocks[0].mlp.act, parameter_name)
    assert torch.unique(p).numel() > 1
    varied(TOKENS).square().sum().backward()
    assert torch.count_nonzero(p.grad) > 1


@pytest.mark.parametrize("activation", ["gelu", "relu"])
def test_vanilla_activation_is_observable(activation):
    model = seeded_model("transformer", mlp_activation=activation).eval()
    x = torch.linspace(-2.1, 1.9, 36).reshape(1, 1, 36).requires_grad_()
    expected = nn.functional.gelu(x) if activation == "gelu" else nn.functional.relu(x)
    actual = model.blocks[0].mlp[1](x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert torch.count_nonzero(x.grad) > 0
    other = seeded_model("transformer", mlp_activation="relu" if activation == "gelu" else "gelu")
    assert not torch.allclose(model(TOKENS), other(TOKENS))


@pytest.mark.parametrize(
    "field,parameter", [("amp_range", "_A"), ("freq_range", "_f"), ("damp_range", "_d")]
)
@pytest.mark.parametrize("bounds", [(0.2, 1.4), (1.4, 0.2), (0.6, 0.6)])
def test_sine_range_sampling_order_and_reversed_equal_neighbors(field, parameter, bounds):
    model = seeded_model("respsann", sine=SineConfig(**{field: bounds})).eval()
    reference_rng = random.Random(137)
    lo, hi = sorted(bounds)
    for block in model.blocks:
        expected = reference_rng.uniform(lo, hi)
        actual = nn.functional.softplus(getattr(block.mlp.act, parameter))
        torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-6, atol=1e-7)
    standard = seeded_model("respsann", sine=SineConfig()).eval()
    assert not torch.allclose(model(TOKENS), standard(TOKENS))
    model(TOKENS).square().sum().backward()
    assert torch.count_nonzero(getattr(model.blocks[0].mlp.act, parameter).grad) > 1


@pytest.mark.parametrize("field", ["amp_init_std", "freq_init_std", "damp_init_std"])
@pytest.mark.parametrize("spread", [-0.2, 0.0])
def test_sine_nonpositive_spread_retains_scalar_runtime(field, spread):
    model = seeded_model("respsann", sine=SineConfig(**{field: spread})).eval()
    reference = seeded_model("respsann", sine=SineConfig()).eval()
    torch.testing.assert_close(model(TOKENS), reference(TOKENS), rtol=0, atol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)


def test_trainer_checkpoint_real_optimizer_step_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    data = psannLMDataPrep(
        ["one two three four five six seven eight nine ten " * 2], tokenizer="simple", max_length=8
    )
    model = seeded_model("respsann", vocab_size=data.vocab_size)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    cfg = TrainConfig(
        batch_tokens=16,
        lr=0.003,
        warmup_steps=0,
        amp="fp32",
        ddp="off",
        checkpoint_dir=str(tmp_path),
        steps_per_epoch=2,
        grad_accum_steps=2,
        dataloader_num_workers=0,
    )
    trainer = Trainer(cfg)
    trainer.train(model, data.dataset, max_length=8)
    payload = torch.load(tmp_path / "final.pt", weights_only=True)
    assert set(payload) == {"state", "model", "optim", "cfg", "data_state"}
    assert payload["state"]["step"] == 2
    assert payload["cfg"] == asdict(cfg)
    assert payload["optim"]["state"]
    assert not torch.equal(before["lm_head.weight"], payload["model"]["lm_head.weight"])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, payload["model"][key], rtol=0, atol=0)
    model2 = seeded_model("respsann", vocab_size=data.vocab_size)
    trainer2 = Trainer(replace(cfg, steps_per_epoch=4))
    trainer2.train(model2, data.dataset, max_length=8, resume_checkpoint=str(tmp_path / "final.pt"))
    resumed = torch.load(tmp_path / "final.pt", weights_only=True)
    assert resumed["state"]["step"] == 4
    assert not torch.equal(resumed["model"]["lm_head.weight"], payload["model"]["lm_head.weight"])
    first = next(iter(payload["optim"]["state"].values()))
    second = next(iter(resumed["optim"]["state"].values()))
    assert second["step"] > first["step"]
