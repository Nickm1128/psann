"""Exploration results must be reproducible, disjoint, bounded and interpretable."""

import json
from copy import deepcopy
from itertools import islice

import pytest
import torch
from psannlm import LMArchitectureConfig, LMTemporalConfig, LMTrainer, TrainConfig
from psannlm.architectures import build_lm_model
from psannlm.persistence import load_lm_checkpoint
from scripts._explore_lm.cli import main
from scripts._explore_lm.config import preset, trials
from scripts._explore_lm.data import Blocks, RepeatBlocks, load_cache, prepare
from scripts._explore_lm.diagnostics import ActivationProbe, causality_probe, probe
from scripts._explore_lm.runner import evaluate, model_for, plan, run, summarize
from torch.utils.data import DataLoader

from psann.activations import PhaseSineParam, SineParam


def small_config():
    cfg = preset("smoke")
    cfg["train"].update(max_steps=4, warmup_steps=0)
    cfg["exploration"].update(eval_interval=2, eval_tokens=64, max_seconds=60)
    return cfg


def test_hash_holdout_is_disjoint_deduplicated_and_repeatable(tmp_path):
    cfg = small_config()
    cfg["data"]["texts"] += cfg["data"]["texts"][:4]
    first, second = prepare(cfg), prepare(cfg)
    assert first["metadata"] == second["metadata"]
    assert first["metadata"]["duplicate_documents_removed"] == 4
    hashes = first["metadata"]["document_hashes"]
    assert not set(hashes["train"]) & set(hashes["validation"])
    assert first["train"].equal(second["train"])
    batch = Blocks(first["train"])[0]
    assert batch["input_ids"][1:].equal(batch["labels"][:-1])
    path = tmp_path / "cache.pt"
    torch.save(first, path)
    assert load_cache(path, cfg)["metadata"] == first["metadata"]
    changed = deepcopy(cfg)
    changed["data"]["split_seed"] += 1
    with pytest.raises(ValueError, match="conflicts"):
        load_cache(path, changed)
    first["train"][0, 0] = (first["train"][0, 0] + 1) % 10
    torch.save(first, path)
    with pytest.raises(ValueError, match="content hash"):
        load_cache(path, cfg)


def test_trial_expansion_preserves_source_and_validates_warmup_and_names():
    config = small_config()
    config["sweep"] = {"train.lr": [0.001, 0.003]}
    config["bench"]["seeds"] = [1, 2]
    before = deepcopy(config)
    rows = trials(config, 32)
    assert len(rows) == 12
    assert len({row["id"] for row in rows}) == 12
    assert config == before
    config["train"]["warmup_steps"] = 2
    with pytest.raises(ValueError, match="warmup"):
        trials(config, 32)
    config["train"]["warmup_steps"] = 0
    config["models"]["bad/label"] = config["models"]["transformer"]
    with pytest.raises(ValueError, match="labels"):
        trials(config, 32)


def test_repeated_training_order_is_independent_of_architecture_rng():
    cache = prepare(small_config())
    dataset = Blocks(cache["train"])
    first = list(islice(iter(RepeatBlocks(dataset, 7)), 10))
    torch.rand(300)
    second = list(islice(iter(RepeatBlocks(dataset, 7)), 10))
    assert all(a["input_ids"].equal(b["input_ids"]) for a, b in zip(first, second))


def test_probe_reports_psann_derivatives_without_changing_weights_or_rng():
    cfg = small_config()
    cache = prepare(cfg)
    trial = trials(cfg, 40)[1]
    model = model_for(trial, "cpu")
    batch = next(iter(DataLoader(Blocks(cache["train"]), batch_size=2)))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = probe(model, batch)
    assert rng.equal(torch.get_rng_state())
    assert model.training
    assert result["finite_logits"] and result["finite_gradients"]
    assert result["activations"]
    assert result["gradients"]["activation"]["with_gradient"] > 0
    for row in result["activations"].values():
        assert row["derivative"]["rms"] > 0
        assert 0 <= row["damped_fraction"] <= 1
    for name, value in model.state_dict().items():
        assert value.equal(before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluation_exact_token_cap_and_rng_preservation():
    cfg = small_config()
    cache = prepare(cfg)
    trial = trials(cfg, 40)[0]
    model = model_for(trial, "cpu")
    dataset = Blocks(cache["validation"])
    state = torch.get_rng_state().clone()
    result = evaluate(model, dataset, 32, 19)
    assert result["val_tokens"] == 19
    assert model.training and state.equal(torch.get_rng_state())
    rows = torch.stack([dataset[0]["input_ids"], dataset[1]["input_ids"]])
    labels = torch.stack([dataset[0]["labels"], dataset[1]["labels"]]).flatten()[:19]
    logits = model.eval()(rows).flatten(0, 1)[:19]
    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert result["val_loss"] == pytest.approx(float(expected.detach()), abs=1e-6)


def test_trainer_observer_is_passive_and_counts_accumulated_tokens(tmp_path):
    cfg = small_config()
    cache = prepare(cfg)
    config = trials(cfg, 40)[0]["model"]
    final, events = [], []
    for observed in (False, True):
        torch.manual_seed(28)
        model = build_lm_model(config).model
        trainer = LMTrainer(
            TrainConfig(
                steps_per_epoch=3,
                batch_tokens=32,
                grad_accum_steps=2,
                lr=0.003,
                warmup_steps=0,
                amp="fp32",
                ddp="off",
                dataloader_num_workers=0,
                checkpoint_dir=str(tmp_path / str(observed)),
                log_interval_steps=50,
            )
        )
        trainer.train(
            model,
            RepeatBlocks(Blocks(cache["train"]), 5),
            max_length=16,
            device="cpu",
            on_step=events.append if observed else None,
        )
        final.append({name: value.clone() for name, value in model.state_dict().items()})
    assert [row["step"] for row in events] == [1, 2, 3]
    assert [row["tokens"] for row in events] == [64, 64, 64]
    assert all(final[0][name].equal(final[1][name]) for name in final[0])


def test_bounded_trials_train_save_curves_and_reproduce(tmp_path):
    config = small_config()
    cache = prepare(config)
    results = run(config, cache, tmp_path / "first")
    assert len(results) == 3 and all(row["status"] == "ok" for row in results)
    again = run(config, cache, tmp_path / "second")
    for left, right in zip(results, again):
        assert left["val_loss"] == right["val_loss"]
        assert left["train_tokens"] == 4 * 64
        curve = json.loads((tmp_path / "first" / left["id"] / "curve.json").read_text())
        assert [row["step"] for row in curve] == [0, 1, 2, 3, 4]
        assert "val_loss" in curve[2] and "gradient_groups" in curve[2]
        restored = load_lm_checkpoint(tmp_path / "first" / left["id"] / "model.pt")
        assert restored.tokenizer is not None
    assert (tmp_path / "first/leaderboard.csv").is_file()
    assert summarize(tmp_path / "first")[0]["val_loss"] == min(row["val_loss"] for row in results)
    with pytest.raises(ValueError, match="absent or empty"):
        run(config, cache, tmp_path / "first")


def test_empty_budget_and_cli_prepare_plan(tmp_path):
    config = small_config()
    cache = prepare(config)
    assert run(config, cache, tmp_path / "limited", max_seconds=1e-12) == []
    assert summarize(tmp_path / "limited") == []
    config_path, cache_path = tmp_path / "smoke.yaml", tmp_path / "cache.pt"
    assert main(["init", "--out", str(config_path)]) == 0
    assert main(["prepare", "--config", str(config_path), "--out", str(cache_path)]) == 0
    assert (
        main(
            [
                "plan",
                "--config",
                str(config_path),
                "--cache",
                str(cache_path),
                "--out",
                str(tmp_path / "plan.json"),
            ]
        )
        == 0
    )
    planned = json.loads((tmp_path / "plan.json").read_text())
    assert len(planned["trials"]) == 3
    assert all(row["parameters"] > 0 for row in planned["trials"])
    assert (
        main(
            [
                "probe",
                "--config",
                str(config_path),
                "--cache",
                str(cache_path),
                "--out",
                str(tmp_path / "probe.json"),
                "--limit",
                "1",
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "probe.json").read_text())[0]["causality"]["passed"]


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_time_budget_must_be_finite_and_positive(tmp_path, budget):
    config = small_config()
    with pytest.raises(ValueError, match="finite and positive"):
        run(config, prepare(config), tmp_path / "invalid", max_seconds=budget)
    assert not (tmp_path / "invalid").exists()


def test_target_audit_uses_explicit_target_dimensions():
    config = small_config()
    config["exploration"]["target_dimensions"] = {
        "d_model": 48,
        "n_layers": 2,
        "n_heads": 3,
        "d_mlp": 96,
    }
    cache = prepare(config)
    proxy, target = plan(config, cache), plan(config, cache, target=True)
    assert target["scale"] == "target"
    assert target["trials"][0]["parameters"] > proxy["trials"][0]["parameters"]
    assert target["trials"][0]["model"]["d_model"] == 48


def test_cache_accepts_implicit_simple_backend(tmp_path):
    config = small_config()
    config["tokenizer"] = {}
    cache = prepare(config)
    path = tmp_path / "cache.pt"
    torch.save(cache, path)
    assert load_cache(path, config)["metadata"] == cache["metadata"]


@pytest.mark.parametrize(
    "mode, expected",
    [("disabled", True), ("interleave", False), ("replace", False), ("attention-only", True)],
)
def test_causality_check_detects_temporal_future_token_leakage(mode, expected):
    from psannlm.architectures import to_mapping

    config = small_config()
    config["models"] = {"candidate": trials(config, 40)[1]["model"]}
    config["models"]["candidate"]["architecture"] = to_mapping(
        LMArchitectureConfig.wave(temporal=LMTemporalConfig(mode=mode))
    )
    model = model_for(trials(config, 40)[0], "cpu")
    state = torch.get_rng_state().clone()
    result = causality_probe(model, torch.arange(16).reshape(1, -1))
    assert result["passed"] is expected
    assert model.training and state.equal(torch.get_rng_state())


def test_failed_causality_trial_is_not_trained_or_shortlisted(tmp_path):
    config = small_config()
    config["models"] = {"temporal": config["models"]["wave_default"]}
    config["models"]["temporal"]["architecture"]["temporal"]["mode"] = "interleave"
    result = run(config, prepare(config), tmp_path / "noncausal")
    assert result[0]["status"] == "failed" and "Causality check" in result[0]["error"]
    assert not (tmp_path / "noncausal" / result[0]["id"] / "checkpoints").exists()
    assert json.loads((tmp_path / "noncausal/shortlist.json").read_text()) == []


@pytest.mark.parametrize("activation", [SineParam, PhaseSineParam])
@pytest.mark.parametrize("decay_mode", ["abs", "relu", "none"])
def test_activation_derivative_diagnostic_matches_autograd(activation, decay_mode):
    kwargs = {"phase_init": 0.7} if activation is PhaseSineParam else {}
    module = activation(
        2,
        amplitude_init=1.2,
        frequency_init=2.5,
        decay_init=0.3,
        decay_mode=decay_mode,
        **kwargs,
    )
    model = torch.nn.Sequential(module)
    observer = ActivationProbe(model)
    inputs = torch.tensor([[-2.0, -0.5], [0.25, 1.0], [2.0, 3.0]], requires_grad=True)
    try:
        model(inputs).sum().backward()
        expected = float(inputs.grad.square().mean().sqrt())
        assert observer.activations["0"]["derivative"]["rms"] == pytest.approx(expected)
    finally:
        observer.close()


def test_report_aggregates_seeds_and_restricts_pareto_to_equal_budgets(tmp_path):
    rows = []
    for identifier, seed, loss, tokens in (
        ("a1", 1, 2.0, 100),
        ("a2", 2, 4.0, 100),
        ("b1", 1, 1.0, 200),
    ):
        trial = {
            "name": identifier[0],
            "model": {"candidate": identifier[0]},
            "train": {"max_steps": tokens},
        }
        directory = tmp_path / identifier
        directory.mkdir()
        (directory / "trial.json").write_text(json.dumps(trial))
        rows.append(
            {
                "id": identifier,
                "name": identifier[0],
                "seed": seed,
                "status": "ok",
                "val_loss": loss,
                "train_tokens": tokens,
                "val_tokens": 20,
                "training_tokens_per_s": 10.0,
            }
        )
    rows += [{"id": "failed", "status": "failed"}]
    (tmp_path / "results.json").write_text(json.dumps(rows))
    (tmp_path / "experiment.json").write_text(json.dumps({"planned_trials": 5}))
    ranked = summarize(tmp_path)
    assert {row["id"]: row["pareto_efficient"] for row in ranked} == {
        "b1": True,
        "a1": True,
        "a2": False,
    }
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert (summary["completed"], summary["failed_or_incomplete"], summary["not_started"]) == (
        3,
        1,
        1,
    )
    aggregates = {row["name"]: row for row in summary["variants"]}
    assert aggregates["a"]["val_loss_mean"] == 3.0
    assert aggregates["a"]["val_loss_std"] == pytest.approx(2**0.5)
    assert aggregates["b"]["val_loss_std"] is None
