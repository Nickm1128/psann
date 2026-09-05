"""Public option spellings, builder replacement and sampling contracts execute end to end."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import re
import warnings

import pytest
import torch

from psannlm import LMConfig, PSANNLM, psannLM
from psannlm.architectures import build_lm_model, normalize_lm_config, to_mapping
from psannlm.architectures import registry
from psannlm.cli import main
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig


def small_config(kind="residual"):
    return LMConfig(
        architecture=kind,
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=29,
        positional_encoding="alibi",
    )


def option(name, value, equals):
    return [f"{name}={value}"] if equals else [name, str(value)]


@pytest.mark.parametrize("equals", [False, True], ids=["spaced", "equals"])
@pytest.mark.parametrize("route", ["train", "resume"])
def test_yaml_option_spellings_select_yaml_and_report_missing_file(equals, route, tmp_path):
    flags = option("--config", tmp_path / "absent.yaml", equals)
    if route == "resume":
        flags += option("--resume-ckpt", tmp_path / "absent.pt", equals)
    with pytest.raises(FileNotFoundError, match="absent.yaml"):
        main([route, *flags])


@pytest.mark.parametrize("flags", [[], ["--config", "missing.yaml"], ["--config=missing.yaml"]])
def test_resume_requires_checkpoint_before_training(flags):
    with pytest.raises(SystemExit, match="resume requires --resume-ckpt"):
        main(["resume", *flags])


@pytest.mark.parametrize(
    "canonical_equals", [False, True], ids=["canonical-spaced", "canonical-equals"]
)
@pytest.mark.parametrize("legacy_equals", [False, True], ids=["legacy-spaced", "legacy-equals"])
@pytest.mark.parametrize("canonical", ["--architecture", "--model-config"])
@pytest.mark.parametrize(
    "legacy,value",
    [("--base", "transformer"), ("--base", "respsann"), ("--sine-freq-init", "0.73")],
)
def test_streaming_conflicting_policy_options_reject_before_setup(
    canonical_equals, legacy_equals, canonical, legacy, value, monkeypatch, tmp_path
):
    import psannlm._train.cli as streaming

    def forbidden(*args, **kwargs):
        pytest.fail("conflicting options reached tokenizer or model construction")

    monkeypatch.setattr(streaming, "_prepare_tokenizer", forbidden)
    monkeypatch.setattr(streaming, "build_lm_model", forbidden)
    flags = option(
        canonical, "residual" if canonical == "--architecture" else "absent.yaml", canonical_equals
    )
    flags += option(legacy, value, legacy_equals)
    flags += ["--data-manifest", str(tmp_path / "absent.txt")]
    with pytest.raises(ValueError, match="--base/--sine"):
        main(["train", *flags])
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "canonical,legacy",
    [
        ("--arch=residual", "--bas=transformer"),
        ("--model-c=absent.yaml", "--sine-train=false"),
        ("--model-c=absent.yaml", "--sine-freq-i=0.73"),
    ],
)
def test_argparse_unambiguous_abbreviations_do_not_bypass_conflicts(canonical, legacy):
    # The second pair uses an unambiguous field abbreviation (freq-init-std also
    # starts with freq-i, so argparse must reject that ambiguous spelling).
    if "freq-i=" in legacy:
        with pytest.raises(SystemExit):
            main(["train", canonical, legacy, "--data-manifest=absent.txt"])
    else:
        with pytest.raises(ValueError, match="--base/--sine"):
            main(["train", canonical, legacy, "--data-manifest=absent.txt"])


@pytest.mark.parametrize("equals", [False, True], ids=["spaced", "equals"])
@pytest.mark.parametrize("mode", ["architecture", "model-config", "matching", "conflicting"])
def test_streaming_model_options_build_or_reject_matching_and_conflicting_dimensions(
    equals, mode, tmp_path, monkeypatch
):
    import json
    import psannlm._train.cli as streaming

    text = "abc def ghij klmn opq rst uvw xyz " * 12
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(str(corpus), encoding="utf-8")
    tok = Tokenizer(
        TokenizerConfig(
            backend="tokenizers", vocab_size=128, min_frequency=1, hf_passthrough_ids=True
        )
    )
    tok.fit([text])
    tok.save(
        str(tmp_path / "tokenizer.json"),
        special_tokens_map_path=str(tmp_path / "special_tokens_map.json"),
    )
    config = replace(small_config(), vocab_size=tok.vocab_size)
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(to_mapping(config)), encoding="utf-8")
    values = {
        "--data-manifest": manifest,
        "--tokenizer-backend": "tokenizers",
        "--tokenizer-model-path": tmp_path / "tokenizer.json",
        "--tokenizer-special-map-path": tmp_path / "special_tokens_map.json",
        "--max-length": 7,
        "--batch-tokens": 14,
        "--max-steps": 3,
        "--warmup-steps": 0,
        "--lr": 0.003,
        "--amp": "fp32",
        "--ddp": "off",
        "--device": "cpu",
        "--num-workers": 0,
        "--checkpoint-dir": tmp_path / "out",
    }
    if mode == "architecture":
        values.update(
            {
                "--architecture": "residual",
                "--d-model": 24,
                "--n-layers": 2,
                "--n-heads": 3,
                "--d-mlp": 36,
                "--pos-enc": "alibi",
            }
        )
    else:
        values["--model-config"] = model_file
        if mode in {"matching", "conflicting"}:
            values["--d-model"] = 24 if mode == "matching" else 48
    flags = [token for key, value in values.items() for token in option(key, value, equals)]
    seen = []
    build = streaming.build_lm_model

    def capture(config):
        result = build(config)
        seen.append((result.model, result.model.lm_head.weight.detach().clone()))
        return result

    monkeypatch.setattr(streaming, "build_lm_model", capture)
    if mode == "conflicting":
        with pytest.raises(ValueError, match="d_model.*conflict|conflict.*d_model"):
            main(["train", *flags])
        assert not seen and not (tmp_path / "out/final.pt").exists()
    else:
        assert main(["train", *flags]) == 0
        assert len(seen) == 1
        model, before = seen[0]
        assert not torch.equal(before, model.lm_head.weight)
        saved = torch.load(tmp_path / "out/final.pt", weights_only=True)
        assert saved["config"] == to_mapping(config) and saved["state"]["step"] == 3
        torch.testing.assert_close(
            saved["model"]["lm_head.weight"], model.lm_head.weight, rtol=0, atol=0
        )


@pytest.mark.parametrize("name", ["custom", "Residual", "legacy:custom", "", " residual", None])
@pytest.mark.parametrize("explicit", [False, True])
def test_unrepresentable_builder_names_are_rejected_without_registry_mutation(
    name, explicit, monkeypatch
):
    monkeypatch.setattr(registry, "_BUILDERS", dict(registry._BUILDERS))
    original = dict(registry._BUILDERS)
    with pytest.raises((TypeError, ValueError), match="registry.name"):
        registry.register_lm_builder(name, original["residual"], replace=explicit)
    assert registry._BUILDERS == original
    assert registry.available_lm_architectures() == (
        "transformer",
        "residual",
        "wave",
        "geometric-sparse",
    )
    config = small_config()
    x = torch.arange(14).reshape(2, 7)
    torch.manual_seed(127)
    a = build_lm_model(config).model
    torch.manual_seed(127)
    b = original["residual"](registry.LMBuildRequest(config)).model
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["transformer", "residual", "wave", "geometric-sparse"])
def test_replacement_builder_executes_typed_mapping_training_and_two_generation_persistence(
    kind, monkeypatch, tmp_path
):
    monkeypatch.setattr(registry, "_BUILDERS", dict(registry._BUILDERS))
    original = registry._BUILDERS[kind]
    seen = []

    def replacement(request):
        result = original(request)
        with torch.no_grad():
            result.model.lm_head.weight.mul_(1.37)
        seen.append(request.config)
        return result

    config = small_config(kind)
    raw = to_mapping(config)
    unchanged = deepcopy(raw)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_lm_builder(kind, replacement)
    registry.replace_lm_builder(kind, replacement)
    x = torch.arange(14).reshape(2, 7)
    torch.manual_seed(191)
    expected = original(registry.LMBuildRequest(config)).model
    torch.manual_seed(191)
    lm = PSANNLM(config=raw, device="cpu")
    actual = lm._ensure_model(29)
    assert not torch.equal(actual(x), expected(x))
    with torch.no_grad():
        expected.lm_head.weight.mul_(1.37)
    torch.testing.assert_close(actual(x), expected(x), rtol=0, atol=0)
    before = actual.lm_head.weight.detach().clone()
    optimizer = torch.optim.AdamW(actual.parameters(), lr=0.003)
    torch.nn.functional.cross_entropy(actual(x).flatten(0, 1), (x + 1).flatten()).backward()
    assert torch.count_nonzero(actual.lm_head.weight.grad) > 20
    optimizer.step()
    assert not torch.equal(before, actual.lm_head.weight)
    expected_logits = actual.eval()(x)
    for generation in (1, 2):
        path = tmp_path / f"generation{generation}.pt"
        lm.save(path)
        saved = torch.load(path, weights_only=True)
        assert saved["config"] == unchanged
        lm = PSANNLM.load(path, map_location="cpu")
        torch.testing.assert_close(lm._model.eval()(x), expected_logits, rtol=0, atol=0)
        for key, value in actual.state_dict().items():
            torch.testing.assert_close(lm._model.state_dict()[key], value, rtol=0, atol=0)
    assert seen == [config, config, config]
    assert raw == unchanged and normalize_lm_config(raw) == config


@pytest.mark.parametrize("bad", ["result", "module", "capabilities", "kind", "positions"])
def test_replacement_builder_rejects_invalid_result_or_capability_mismatch(bad, monkeypatch):
    monkeypatch.setattr(registry, "_BUILDERS", dict(registry._BUILDERS))
    original = registry._BUILDERS["residual"]

    def broken(request):
        result = original(request)
        if bad == "result":
            return result.model
        if bad == "module":
            return registry.LMBuildResult(object(), result.capabilities)
        if bad == "capabilities":
            return registry.LMBuildResult(result.model, object())
        return registry.LMBuildResult(
            result.model,
            replace(
                result.capabilities,
                **(
                    {"kind": "transformer"}
                    if bad == "kind"
                    else {"positional_encodings": ("rope",)}
                ),
            ),
        )

    registry.replace_lm_builder("residual", broken)
    with pytest.raises((ValueError, TypeError), match="registry"):
        build_lm_model(small_config())


@pytest.mark.parametrize("builder", [None, 3, "builder"])
def test_replacement_requires_callable_without_mutation(builder):
    original = dict(registry._BUILDERS)
    with pytest.raises(TypeError, match="registry.builder"):
        registry.replace_lm_builder("residual", builder)
    assert registry._BUILDERS == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", ""),
        ("trainer_identifier", " bad"),
        ("export_identifier", None),
        ("kv_cache", 1),
        ("gradient_checkpointing", "true"),
        ("positional_encodings", ["alibi"]),
        ("positional_encodings", ()),
        ("positional_encodings", ("alibi", "alibi")),
        ("positional_encodings", ("unknown",)),
    ],
)
def test_replacement_capabilities_are_strict_before_dispatch_returns(field, value, monkeypatch):
    monkeypatch.setattr(registry, "_BUILDERS", dict(registry._BUILDERS))
    original = registry._BUILDERS["residual"]

    def broken(request):
        result = original(request)
        return registry.LMBuildResult(result.model, replace(result.capabilities, **{field: value}))

    registry.replace_lm_builder("residual", broken)
    with pytest.raises((ValueError, TypeError), match="registry.capabilities." + field):
        build_lm_model(small_config())


def test_compatibility_replacement_warns_once_and_does_not_bypass_capability_validation(
    monkeypatch,
):
    monkeypatch.setattr(registry, "_BUILDERS", dict(registry._BUILDERS))
    original = registry._BUILDERS["residual"]
    seen = []

    def replacement(request):
        seen.append(request.config)
        return original(request)

    with pytest.warns(DeprecationWarning) as caught:
        registry.register_lm_builder("residual", replacement, replace=True)
    assert len(caught) == 1 and caught[0].filename == __file__
    raw = to_mapping(small_config())
    raw["architecture"]["temporal"] = {"mode": "interleave"}
    with pytest.raises(ValueError, match="architecture.temporal"):
        build_lm_model(raw)
    assert seen == []
    model = build_lm_model(small_config()).model
    tokens = torch.arange(14).reshape(2, 7)
    model(tokens).square().sum().backward()
    assert torch.count_nonzero(model.lm_head.weight.grad) > 20
    assert seen == [small_config()]


@pytest.mark.parametrize("unknown", ["temperture", "top_k_tokens", "unused"])
@pytest.mark.parametrize("length", [0, 7])
def test_canonical_generate_unknown_options_reject_before_model_or_tokenizer_work(unknown, length):
    lm = PSANNLM(config=small_config(), device="cpu")
    with pytest.raises(TypeError, match=unknown):
        lm.generate("abc", max_new_tokens=length, **{unknown: 0.37})
    assert lm._model is None and lm._tokenizer is None


@pytest.mark.parametrize("legacy", [False, True])
def test_generation_known_options_execute_sampling_and_legacy_unknowns_only_warn(
    legacy, monkeypatch
):
    from psannlm.lm.infer import generate as generation

    tok = Tokenizer(TokenizerConfig(backend="simple"))
    tok.fit(["abc def ghij klmn opq rst uvw xyz"])
    config = replace(small_config(), vocab_size=tok.vocab_size)
    if legacy:
        with pytest.warns(DeprecationWarning):
            lm = psannLM(config=config, device="cpu")
    else:
        lm = PSANNLM(config=config, device="cpu")
    torch.manual_seed(797)
    model = lm._ensure_model(tok.vocab_size).eval()
    lm.attach_tokenizer(tok)
    seen = []
    sample = generation.sample_next_token

    def capture(logits, **kwargs):
        seen.append((logits.detach().clone(), kwargs))
        return sample(logits, **kwargs)

    monkeypatch.setattr(generation, "sample_next_token", capture)
    kwargs = dict(max_new_tokens=7, top_k=3, top_p=0.67, temperature=0.0, repetition_penalty=1.43)
    # Compare the full executed sampling inputs and decoded output with an
    # independently composed autoregressive loop, including repetition penalties.
    tokens = torch.tensor([tok.encode("abc", add_specials=True)])
    expected = []
    logits_expected = []
    with torch.no_grad():
        for _ in range(7):
            logits = model(tokens)[:, -1, :]
            if expected:
                logits.scatter_add_(
                    -1, torch.tensor([expected]), torch.full((1, len(expected)), -1.43)
                )
            logits_expected.append(logits.clone())
            token = sample(logits, temperature=0.0, top_k=3, top_p=0.67).item()
            expected.append(token)
            tokens = torch.cat([tokens, torch.tensor([[token]])], dim=1)
            if token == tok.eos_id:
                break
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = lm.generate(
            "abc", **kwargs, **({"temperture": 0.73, "unused": 11} if legacy else {})
        )
    assert len(caught) == int(legacy)
    if legacy:
        assert caught[0].category is DeprecationWarning and caught[0].filename == __file__
        assert "temperture" in str(caught[0].message) and "unused" in str(caught[0].message)
    assert actual == tok.decode(expected, skip_specials=True)
    assert len(seen) == len(expected) and len(expected) > 1
    for (actual_logits, controls), expected_logits in zip(seen, logits_expected):
        torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
        assert controls == dict(temperature=0.0, top_k=3, top_p=0.67)


def test_maintained_contributor_guide_executes_typed_replacement_and_gradient(monkeypatch):
    monkeypatch.setattr(registry, "_BUILDERS", dict(registry._BUILDERS))
    root = Path(__file__).resolve().parents[2]
    text = (root / "docs/how_to_add_model_benchmark_dataset.md").read_text(encoding="utf-8")
    assert "psannlm/lm/models/registry.py" not in text
    assert "replace_lm_builder" in text and "capabilit" in text.lower()
    (snippet,) = re.findall(r"```python\n(.*?)```", text, re.S)
    namespace = {}
    exec(compile(snippet, "contributor-guide", "exec"), namespace)
    config = small_config()
    model = build_lm_model(config).model
    tokens = torch.arange(14).reshape(2, 7)
    before = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.013)
    torch.nn.functional.cross_entropy(
        model(tokens).flatten(0, 1), (tokens + 1).flatten()
    ).backward()
    assert torch.count_nonzero(model.lm_head.weight.grad) > 20
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)
    assert model.lm_config == config
