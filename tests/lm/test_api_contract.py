"""Public compatibility, strict training policies and inactive-policy neighbors."""

from copy import deepcopy
from dataclasses import fields, replace
import random

import pytest
import torch

from psann.architectures import ActivationConfig, ResidualConfig
from psannlm import LMArchitectureConfig, LMConfig, LMTrainer, PSANNLM, PSANNLMDataPrep, TrainConfig
from psannlm.architectures import build_lm_model, to_mapping
from psannlm.architectures.compat import legacy_lm_config
from psannlm.lm.config import normalize_train_config
from psannlm.lm.train.trainer import Trainer
from psannlm import psannLM, psannLMDataPrep
from psannlm.persistence import load_lm_checkpoint


@pytest.mark.parametrize("base", ["transformer", "respsann", "sgrpsann", "waveresnet", "geosparse"])
def test_alias_warns_once_at_caller_and_matches_config_state_logits_and_saved_metadata(
    base, tmp_path
):
    options = dict(
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=29,
        positional_encoding="alibi",
        sine_params={"freq_init": 0.73, "amp_init_std": 0.17},
    )
    original = deepcopy(options)
    torch.manual_seed(271)
    with pytest.warns(DeprecationWarning) as warnings:
        legacy = psannLM(base=base, **options)
    assert len(warnings) == 1 and warnings[0].filename == __file__
    old_model = legacy._ensure_model(29).eval()
    config = legacy_lm_config(base, options, high_level=True, warn=False)
    mapping = to_mapping(config)
    expected_mapping = deepcopy(mapping)
    torch.manual_seed(271)
    canonical = PSANNLM(config=mapping, device="cpu")
    new_model = canonical._ensure_model(29).eval()
    tokens = torch.arange(14).reshape(2, 7) + 4
    torch.testing.assert_close(old_model(tokens), new_model(tokens), rtol=0, atol=0)
    old_model(tokens).square().mean().backward()
    new_model(tokens).square().mean().backward()
    for (name, p), (other, q) in zip(old_model.named_parameters(), new_model.named_parameters()):
        assert name == other and p.requires_grad == q.requires_grad
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    for obj, name in [(legacy, "legacy"), (canonical, "canonical")]:
        obj.save(tmp_path / f"{name}.pt")
    assert (
        torch.load(tmp_path / "legacy.pt", weights_only=True)["config"]
        == torch.load(tmp_path / "canonical.pt", weights_only=True)["config"]
    )
    with pytest.warns(DeprecationWarning) as warnings:
        loaded = psannLM.load(tmp_path / "legacy.pt", map_location="cpu")
    assert len(warnings) == 1 and warnings[0].filename == __file__
    assert isinstance(loaded, psannLM)
    torch.testing.assert_close(loaded._model.eval()(tokens), old_model(tokens), rtol=0, atol=0)
    assert options == original and mapping == expected_mapping


def test_legacy_data_and_trainer_aliases_warn_once_and_execute_same_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.warns(DeprecationWarning) as warnings:
        data = psannLMDataPrep(
            ["abc def ghij klmn opq rst uvw xyz " * 8], tokenizer="simple", max_length=7
        )
    assert len(warnings) == 1 and warnings[0].filename == __file__
    cfg = TrainConfig(
        steps_per_epoch=3,
        batch_tokens=14,
        lr=0.003,
        warmup_steps=0,
        amp="fp32",
        ddp="off",
        checkpoint_dir=str(tmp_path),
    )
    with pytest.warns(DeprecationWarning) as warnings:
        trainer = Trainer(cfg)
    assert len(warnings) == 1 and warnings[0].filename == __file__
    model_cfg = LMConfig(
        architecture="residual",
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=data.vocab_size,
    )
    torch.manual_seed(733)
    trainer.train(model_cfg, data.dataset, max_length=7, device="cpu")
    first = torch.load(tmp_path / "final.pt", weights_only=True)
    torch.manual_seed(733)
    LMTrainer(cfg).train(to_mapping(model_cfg), data.dataset, max_length=7, device="cpu")
    second = torch.load(tmp_path / "final.pt", weights_only=True)
    for key in first["model"]:
        torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
    assert first["config"] == second["config"] and first["state"] == second["state"]


@pytest.mark.parametrize(
    "field",
    [
        f.name
        for f in fields(TrainConfig)
        if f.type in {"int", "float", "int | None", "float | None"}
    ],
)
@pytest.mark.parametrize("value", [True, "2"])
def test_train_numeric_fields_reject_bool_and_string_neighbors(field, value):
    with pytest.raises((ValueError, TypeError), match="train." + field):
        normalize_train_config({field: value})


@pytest.mark.parametrize("field", [f.name for f in fields(TrainConfig) if f.type == "bool"])
@pytest.mark.parametrize("value", [0, "false"])
def test_train_boolean_fields_reject_numeric_and_string_neighbors(field, value):
    with pytest.raises((ValueError, TypeError), match="train." + field):
        normalize_train_config({field: value})


@pytest.mark.parametrize("ratio", [0.0, 0.01, 0.04])
def test_mixed_sampling_requires_executing_positive_width_child(ratio):
    activation = ActivationConfig(
        kind="mixed",
        activation_types=("psann", "gelu"),
        activation_ratios=(ratio, 1 - ratio),
        mix_layout="contiguous",
    )
    architecture = LMArchitectureConfig.geometric_sparse(
        activation=activation, activation_initialization={"amplitude_std": 0.23}
    )
    values = dict(d_model=24, n_layers=2, n_heads=3, d_mlp=36, vocab_size=29)
    if ratio < 0.04:
        with pytest.raises(ValueError, match="activation_initialization.*positive width"):
            LMConfig(architecture, **values)
        model = build_lm_model(
            LMConfig(replace(architecture, activation_initialization=None), **values)
        ).model
    else:
        model = build_lm_model(LMConfig(architecture, **values)).model
    logits = model(torch.arange(14).reshape(2, 7))
    logits.square().mean().backward()
    assert torch.count_nonzero(model.lm_head.weight.grad) > 10
    children = [m for name, m in model.named_modules() if name.endswith("acts.psann")]
    assert bool(children) == (ratio >= 0.04)
    if children:
        assert any(
            p.grad is not None and torch.count_nonzero(p.grad) for p in children[0].parameters()
        )


@pytest.mark.parametrize("field,value", [("alpha_init", 0.37), ("drop_path", 0.4)])
def test_attention_only_rejects_inactive_residual_policy_but_executes_norm(field, value):
    with pytest.raises(ValueError, match="residual.alpha_init/drop_path"):
        LMArchitectureConfig.wave(
            temporal={"mode": "attention-only"},
            residual=replace(ResidualConfig(alpha_init=1.0), **{field: value}),
        )
    config = LMConfig(
        LMArchitectureConfig.wave(
            temporal={"mode": "attention-only"},
            residual=ResidualConfig(alpha_init=1.0, norm="layer"),
        ),
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=29,
    )
    model = build_lm_model(config).model
    x = torch.randn(2, 7, 24, requires_grad=True)
    block = model.blocks[0]
    torch.testing.assert_close(block(x), x + block.attn(block.norm1(x)), rtol=0, atol=0)
    block(x).square().sum().backward()
    assert torch.count_nonzero(block.norm1.weight.grad) > 1


@pytest.mark.parametrize("target", [2, 3])
def test_completed_resume_preserves_model_counters_rng_and_future_random_draws(target, tmp_path):
    config = LMConfig(
        architecture="residual", d_model=24, n_layers=2, n_heads=3, d_mlp=36, vocab_size=29
    )
    cfg = TrainConfig(
        steps_per_epoch=3,
        batch_tokens=14,
        lr=0.003,
        warmup_steps=0,
        amp="fp32",
        ddp="off",
        checkpoint_dir=str(tmp_path),
    )
    dataset = [
        {"input_ids": torch.arange(7) + i, "labels": torch.arange(7) + i + 1} for i in range(12)
    ]
    LMTrainer(cfg).train(config, dataset, max_length=7, device="cpu")
    source = tmp_path / "final.pt"
    original = torch.load(source, weights_only=True)
    generator = torch.Generator().set_state(original["rng"]["torch"])
    expected_draw = torch.rand(11, generator=generator)
    python = random.Random()
    python.setstate(original["rng"]["python"])
    expected_python = [python.random() for _ in range(7)]
    trainer = LMTrainer(replace(cfg, steps_per_epoch=target))
    trainer.train(config, dataset, max_length=7, device="cpu", resume_checkpoint=str(source))
    restored = torch.load(source, weights_only=True)
    assert restored["state"] == original["state"]
    for key in original["model"]:
        torch.testing.assert_close(restored["model"][key], original["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(11), expected_draw, rtol=0, atol=0)
    assert [random.random() for _ in range(7)] == expected_python
    assert torch.equal(restored["rng"]["torch"], original["rng"]["torch"])
    assert restored["rng"]["numpy"] == original["rng"]["numpy"]
    assert len(restored["rng"]["cuda"]) == len(original["rng"]["cuda"])
    for key in ("torch", "cuda", "python", "numpy"):
        malformed = deepcopy(restored)
        malformed["rng"][key] = "bad"
        torch.save(malformed, tmp_path / "bad.pt")
        with pytest.raises(ValueError, match="checkpoint.rng"):
            load_lm_checkpoint(tmp_path / "bad.pt")


def test_external_legacy_factory_retains_kwargs_and_duplicate_rules(monkeypatch):
    from psannlm.architectures import registry
    from psannlm.lm.models.registry import get_base, list_bases, register_base

    monkeypatch.setattr(registry, "_BUILDERS", dict(registry._BUILDERS))

    def factory(*, width, scale):
        model = torch.nn.Linear(width, 5, bias=False)
        with torch.no_grad():
            model.weight.fill_(scale)
        return model

    with pytest.warns(DeprecationWarning):
        register_base("custom", factory)
    with pytest.raises(ValueError, match="already registered"):
        register_base("custom", factory)
    with pytest.warns(DeprecationWarning):
        assert "custom" in list_bases()
    with pytest.warns(DeprecationWarning) as warnings:
        model = get_base("custom")(width=7, scale=0.37)
    assert len(warnings) == 1 and warnings[0].filename == __file__
    x = torch.linspace(-1, 2, 21).reshape(3, 7).requires_grad_()
    torch.testing.assert_close(model(x), (x.sum(-1, keepdim=True) * 0.37).expand(3, 5))
    model(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 5 * 0.37))
    with pytest.warns(DeprecationWarning):
        register_base("custom", factory, replace=True)


@pytest.mark.parametrize("base", ["transformer", "respsann", "sgrpsann", "waveresnet", "geosparse"])
def test_legacy_model_config_converts_to_executing_model_without_second_warning(base):
    from psannlm.lm.config import ModelConfig

    with pytest.warns(DeprecationWarning) as warnings:
        shell = ModelConfig(
            base=base,
            d_model=24,
            n_layers=2,
            n_heads=3,
            d_mlp=36,
            vocab_size=29,
            sine_freq_init=0.73,
        )
    assert len(warnings) == 1 and warnings[0].filename == __file__
    config = shell.to_lm_config()
    torch.manual_seed(719)
    actual = build_lm_model(config).model
    torch.manual_seed(719)
    expected = build_lm_model(
        legacy_lm_config(
            base,
            dict(
                d_model=24, n_layers=2, n_heads=3, d_mlp=36, vocab_size=29, sine={"freq_init": 0.73}
            ),
            warn=False,
        )
    ).model
    tokens = torch.arange(14).reshape(2, 7)
    torch.testing.assert_close(actual(tokens), expected(tokens), rtol=0, atol=0)
    actual(tokens).square().mean().backward()
    expected(tokens).square().mean().backward()
    torch.testing.assert_close(
        actual.lm_head.weight.grad, expected.lm_head.weight.grad, rtol=0, atol=0
    )


def test_canonical_vocab_resolution_and_conflict_do_not_mutate_caller(tmp_path):
    config = LMConfig(architecture="residual", d_model=24, n_layers=2, n_heads=3)
    lm = PSANNLM(config=config, device="cpu")
    with pytest.raises(ValueError, match="vocab_size"):
        lm.save(tmp_path / "unresolved.pt")
    assert lm._model is None and config.vocab_size is None
    model = lm._ensure_model(29)
    assert model(torch.arange(14).reshape(2, 7)).shape == (2, 7, 29)
    assert config.vocab_size is None and config.d_mlp is None
    with pytest.raises(ValueError, match="vocab_size conflicts"):
        lm._ensure_model(31)
    before = model.lm_head.weight.detach().clone()
    lm.save(tmp_path / "resolved.pt")
    restored = PSANNLM.load(tmp_path / "resolved.pt", map_location="cpu")
    torch.testing.assert_close(restored._model.lm_head.weight, before, rtol=0, atol=0)
    assert restored.config.vocab_size == 29 and restored.config.d_mlp == 96


def test_flat_fit_repeated_calls_preserve_legacy_training_policy_and_exact_updates(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    data = PSANNLMDataPrep(
        ["abc def ghij klmn opq rst uvw xyz " * 4], tokenizer="simple", max_length=7
    )
    config = LMConfig(
        architecture="residual",
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=data.vocab_size,
    )
    train = TrainConfig(epochs=1, batch_tokens=14, lr=0.003, amp="fp32", ddp="off")
    results = []
    for legacy in (False, True):
        torch.manual_seed(163)
        lm = PSANNLM(config=config, device="cpu")
        if legacy:
            with pytest.warns(DeprecationWarning) as warnings:
                lm.fit(
                    data,
                    epochs=1,
                    batch_tokens=14,
                    lr=0.003,
                    amp="fp32",
                    ddp="off",
                    steps_per_epoch=1,
                )
            assert len(warnings) == 1 and "steps_per_epoch" in str(warnings[0].message)
            with pytest.warns(DeprecationWarning):
                lm.fit(data, epochs=4, batch_tokens=700, lr=0.9, grad_checkpoint=True)
        else:
            lm.fit(data, train=train)
            lm.fit(data, train=replace(train, grad_checkpoint=True))
        assert lm._trainer.cfg == replace(train, grad_checkpoint=True)
        results.append(lm._model.state_dict())
    assert train.grad_checkpoint is False
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["sentencepiece", "tokenizers"])
def test_corrupt_serialized_tokenizer_names_model_path(backend):
    from psannlm.lm.data.tokenizer import Tokenizer

    state = {"backend": backend, "model": b"bad" if backend == "sentencepiece" else "bad"}
    if backend == "tokenizers":
        state.update(passthrough_ids=True, special_ids={"pad": 0, "bos": 1, "eos": 2, "unk": 3})
    with pytest.raises(ValueError, match="tokenizer.model"):
        Tokenizer.from_state(state)


@pytest.mark.parametrize(
    "mode", ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]
)
def test_compile_mode_neighbors_are_explicit(mode):
    assert normalize_train_config({"torch_compile_mode": mode}).torch_compile_mode == mode
    with pytest.raises(ValueError, match="train.torch_compile_mode"):
        normalize_train_config({"torch_compile_mode": mode + "-unknown"})
