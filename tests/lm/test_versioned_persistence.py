"""Artifact closure is proved by executed models, exact state and sampled tokens."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random

import pytest
import torch

from psannlm import LMConfig, PSANNLM, PSANNLMDataPrep, TrainConfig, LMTrainer
from psannlm.architectures import build_lm_model, normalize_lm_config, to_mapping
from psannlm.architectures.compat import legacy_lm_config
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.persistence import load_lm_checkpoint, model_payload

FIXTURES = Path(__file__).parent / "fixtures/legacy_models"
BASES = ("transformer", "respsann", "sgrpsann", "waveresnet", "geosparse")
TEXTS = ["abc def ghij", "klmn opq rst uvw xyz"]
TOKENS = torch.tensor([[1, 4, 7, 10, 13, 16, 19], [2, 5, 8, 11, 14, 17, 20]])


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def closure(lm, tmp_path, *, mode=None):
    original_config = to_mapping(lm.config)
    original_state = {k: v.clone() for k, v in lm._model.state_dict().items()}
    lm._model.eval()
    original_logits = lm._model(TOKENS)
    torch.manual_seed(991)
    original_text = lm.generate("abc", max_new_tokens=11, top_p=0.87, temperature=0.73)
    original_ids = lm._tokenizer.encode("abc", add_specials=True)
    for generation in (1, 2):
        path = tmp_path / f"generation{generation}.pt"
        lm.save(path)
        payload = torch.load(path, weights_only=True)
        assert payload["schema"] == "psannlm.model" and payload["schema_version"] == 1
        assert payload["config"] == original_config
        if mode is not None:
            assert payload["config"]["architecture"]["temporal"]["mode"] == mode
        lm = PSANNLM.load(path, map_location="cpu")
        assert to_mapping(lm.config) == original_config
        assert_state(lm._model.state_dict(), original_state)
        lm._model.eval()
        torch.testing.assert_close(lm._model(TOKENS), original_logits, rtol=0, atol=0)
        assert lm._tokenizer.encode("abc", add_specials=True) == original_ids
        torch.manual_seed(991)
        assert lm.generate("abc", max_new_tokens=11, top_p=0.87, temperature=0.73) == original_text
    return lm


@pytest.mark.parametrize("base", BASES)
def test_legacy_high_model_two_generations_exact_state_tokens_and_discriminators(base, tmp_path):
    lm = PSANNLM.load(FIXTURES / f"{base}_high.pt", map_location="cpu")
    lm.attach_tokenizer(PSANNLMDataPrep(TEXTS, tokenizer="simple").tokenizer)
    restored = closure(lm, tmp_path)
    assert restored.config.architecture.kind == {
        "respsann": "residual",
        "sgrpsann": "residual",
        "waveresnet": "wave",
        "geosparse": "geometric-sparse",
    }.get(base, base)
    assert (restored.config.architecture.spectral is not None) == (base == "sgrpsann")


@pytest.mark.parametrize("base", BASES)
def test_nondefault_factory_weights_migrate_through_two_model_generations(base, tmp_path):
    from test_legacy_routes import options

    reference = torch.load(FIXTURES / f"{base}.pt", weights_only=True)
    raw = tmp_path / "old_weights.pt"
    torch.save(reference["state"], raw)
    loaded = load_lm_checkpoint(
        raw, legacy_config=legacy_lm_config(base, options(base), warn=False)
    )
    assert loaded.artifact_kind == "psannlm.weights"
    lm = PSANNLM(config=loaded.config, device="cpu")
    lm._model = loaded.model
    lm.attach_tokenizer(PSANNLMDataPrep(TEXTS, tokenizer="simple").tokenizer)
    closure(lm, tmp_path)
    torch.testing.assert_close(
        loaded.model.eval()(TOKENS), reference["logits"], rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize(
    "interleave,replace_wave,mode",
    [
        (False, False, "disabled"),
        (True, False, "interleave"),
        (True, True, "replace"),
        (False, True, "attention-only"),
    ],
)
@pytest.mark.parametrize("activation", ["sine", "gelu"])
def test_wave_all_legacy_structures_activation_and_rng_survive_two_generations(
    interleave, replace_wave, mode, activation, tmp_path
):
    options = dict(
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=29,
        wave_interleave=interleave,
        wave_replace=replace_wave,
        mlp_activation=activation,
    )
    if interleave:
        options.update(wave_kernel_size=5, wave_dilation_growth=2)
    if activation == "sine" and mode != "attention-only":
        options["sine"] = {"amp_init_std": 0.23, "freq_range": [0.6, 1.4], "damp_init_std": 0.017}
    torch.manual_seed(359)
    random.seed(359)
    config = legacy_lm_config("waveresnet", options, warn=False)
    lm = PSANNLM(config=config, device="cpu")
    lm._ensure_model(29)
    lm.attach_tokenizer(PSANNLMDataPrep(TEXTS, tokenizer="simple").tokenizer)
    restored = closure(lm, tmp_path, mode=mode)
    block = restored._model.blocks[0]
    states = block.state_dict()
    assert any(k.startswith("mlp.") for k in states) == (not replace_wave)
    assert any(k.startswith("wave.") for k in states) == interleave
    if mode == "attention-only":
        x = torch.randn(2, 7, 24, requires_grad=True)
        expected = x + block.attn(block.norm1(x))
        actual = block(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        assert torch.count_nonzero(x.grad) == x.numel()


@pytest.mark.parametrize(
    "backend,passthrough",
    [("simple", False), ("sentencepiece", False), ("tokenizers", False), ("tokenizers", True)],
)
def test_tokenizer_fitted_state_generation_survives_two_generations_without_files(
    backend, passthrough, tmp_path
):
    tokenizer = Tokenizer(
        TokenizerConfig(
            backend=backend, vocab_size=128, min_frequency=1, hf_passthrough_ids=passthrough
        )
    )
    tokenizer.fit(TEXTS * 3)
    cfg = LMConfig(
        architecture="residual",
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=tokenizer.vocab_size,
    )
    lm = PSANNLM(config=cfg, device="cpu")
    lm._ensure_model(tokenizer.vocab_size)
    lm.attach_tokenizer(tokenizer)
    ids = tokenizer.encode("abc def", add_specials=True)
    decoded = tokenizer.decode(ids)
    restored = closure(lm, tmp_path)
    assert restored._tokenizer.decode(ids) == decoded
    assert restored._tokenizer.to_state() == tokenizer.to_state()


@pytest.fixture
def payload():
    model = build_lm_model(
        LMConfig(
            architecture="residual", d_model=24, n_layers=2, n_heads=3, d_mlp=36, vocab_size=29
        )
    ).model
    return model_payload(model, None)


@pytest.mark.parametrize(
    "path,value,error",
    [
        (("schema",), "psann.model", "checkpoint.schema"),
        (("schema",), [], "checkpoint.schema"),
        (("schema_version",), True, "checkpoint.schema_version"),
        (("schema_version",), 1.0, "checkpoint.schema_version"),
        (("schema_version",), 2, "checkpoint.schema_version"),
        (("schema_version",), "1", "checkpoint.schema_version"),
        (("package_version",), "", "checkpoint.package_version"),
        (("extra",), 0, "checkpoint.extra"),
        (("config", "kind"), "model", "checkpoint.config.kind"),
        (("config", "d_model"), True, "config.d_model"),
        (("config", "architecture", "kind"), "sgrpsann", "architecture.kind"),
        (("config", "architecture", "typo"), 0, "architecture.typo"),
        (("config", "architecture", "residual"), None, "architecture.residual"),
        (("state_dict",), [], "checkpoint.state_dict"),
        (("state_dict", "embed.weight"), 4, "checkpoint.state_dict.embed.weight"),
        (
            ("state_dict", "embed.weight"),
            torch.ones(28, 24),
            "checkpoint.state_dict.embed.weight.shape",
        ),
        (
            ("state_dict", "embed.weight"),
            torch.ones(29, 24, dtype=torch.int64),
            "checkpoint.state_dict.embed.weight.dtype",
        ),
        (("state_dict", "unknown"), torch.tensor(0), "checkpoint.state_dict.unknown"),
        (("device",), "garbage", "checkpoint.device"),
        (("device",), "meta", "checkpoint.device"),
        (("tokenizer",), {"backend": "unknown"}, "tokenizer.backend"),
        (("tokenizer",), {"backend": "simple", "vocabulary": ["a", "a"]}, "tokenizer.vocabulary"),
    ],
)
def test_model_artifact_rejects_adjacent_malformed_values_at_named_path(
    payload, path, value, error, tmp_path
):
    raw = deepcopy(payload)
    target = raw
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    file = tmp_path / "bad.pt"
    torch.save(raw, file)
    with pytest.raises((ValueError, TypeError), match=error):
        PSANNLM.load(file, map_location="cpu")


@pytest.mark.parametrize(
    "missing",
    ["schema", "schema_version", "package_version", "config", "state_dict", "device", "tokenizer"],
)
def test_model_artifact_requires_exact_top_level_fields(payload, missing, tmp_path):
    del payload[missing]
    file = tmp_path / "missing.pt"
    torch.save(payload, file)
    with pytest.raises((ValueError, TypeError), match="checkpoint." + missing):
        PSANNLM.load(file)


def test_legacy_trainer_is_explicit_resume_artifact_and_restarts_optimizer(tmp_path):
    config = legacy_lm_config(
        "respsann",
        dict(
            vocab_size=29,
            d_model=24,
            n_layers=2,
            n_heads=3,
            d_mlp=36,
            positional_encoding="alibi",
            sine={"freq_init": 0.73, "amp_init_std": 0.17},
        ),
        warn=False,
    )
    file = FIXTURES / "trainer_residual.pt"
    with pytest.raises(ValueError, match="checkpoint.config"):
        load_lm_checkpoint(file)
    loaded = load_lm_checkpoint(file, legacy_config=config)
    assert loaded.artifact_kind == "psannlm.trainer"
    original = loaded.payload
    assert_state(loaded.model.state_dict(), original["model"])
    assert original["state"]["step"] == 2 and original["optim"]["state"]
    cfg = TrainConfig(
        batch_tokens=14,
        lr=0.003,
        warmup_steps=0,
        amp="fp32",
        ddp="off",
        steps_per_epoch=5,
        checkpoint_dir=str(tmp_path),
        dataloader_num_workers=0,
    )
    dataset = [{"input_ids": TOKENS[i % 2], "labels": TOKENS[i % 2].roll(-1)} for i in range(12)]
    trainer = LMTrainer(cfg)
    trainer.train(loaded.model, dataset, max_length=7, resume_checkpoint=str(file), device="cpu")
    saved = torch.load(tmp_path / "final.pt", weights_only=True)
    assert saved["state"]["step"] == 5
    assert saved["config"] == to_mapping(normalize_lm_config(config, for_build=True))
    assert not torch.equal(saved["model"]["lm_head.weight"], original["model"]["lm_head.weight"])
    assert all(v["step"].item() == 5 for v in saved["optim"]["state"].values())
    with pytest.raises(ValueError, match="checkpoint.schema is psannlm.trainer"):
        PSANNLM.load(tmp_path / "final.pt")
    restored = load_lm_checkpoint(tmp_path / "final.pt")
    torch.testing.assert_close(
        restored.model.eval()(TOKENS), loaded.model.eval()(TOKENS), rtol=0, atol=0
    )


@pytest.mark.parametrize("architecture", ["transformer", "residual", "wave", "geometric-sparse"])
def test_canonical_fit_uses_immutable_nondefault_policy_and_continues_after_extended_resume(
    architecture, tmp_path
):
    data = PSANNLMDataPrep(TEXTS * 8, tokenizer="simple", max_length=7)
    source = LMConfig(architecture=architecture, d_model=24, n_layers=2, n_heads=3, d_mlp=36)
    lm = PSANNLM(config=source, device="cpu")
    training = TrainConfig(
        steps_per_epoch=3,
        batch_tokens=14,
        lr=0.004,
        warmup_steps=0,
        amp="fp32",
        ddp="off",
        grad_accum_steps=2,
        checkpoint_dir=str(tmp_path),
    )
    lm.fit(data, train=training)
    assert source.vocab_size is None and lm.config.vocab_size == data.vocab_size
    first = torch.load(tmp_path / "final.pt", weights_only=True)
    assert first["state"]["step"] == 3
    second = replace(training, steps_per_epoch=6)
    lm.fit(data, train=to_mapping(second), resume_checkpoint=str(tmp_path / "final.pt"))
    final = torch.load(tmp_path / "final.pt", weights_only=True)
    assert training.steps_per_epoch == 3 and second.steps_per_epoch == 6
    assert final["state"]["step"] == 6
    assert not torch.equal(final["model"]["lm_head.weight"], first["model"]["lm_head.weight"])
    assert all(v["step"].item() == 6 for v in final["optim"]["state"].values())
