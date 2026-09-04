"""Durable boundaries for the canonical Phase-5 episodic surface."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig, ConvolutionConfig
from psann.episodic import (
    EpisodeScheduleConfig,
    EpisodicTrainer,
    HISSOConfig,
    SupervisedWarmStartConfig,
    normalize_strategy,
)
from psann.episodic.rewards import FINANCE_PORTFOLIO_STRATEGY
from psann.episodic.runtime import transform_actions
from psann.preprocessing import LSMConfig, PreprocessorConfig, PreprocessorTrainingConfig


def _custom_reward(actions: torch.Tensor, context: torch.Tensor, **_kwargs: object) -> torch.Tensor:
    return -(actions - context.mean(dim=-1, keepdim=True)).square().mean(dim=(-1, -2))


@pytest.mark.parametrize(
    "strategy",
    [
        HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=4, batch_episodes=2, updates_per_epoch=1)
        ),
        {
            "kind": "hisso",
            "schedule": {"episode_length": 4, "batch_episodes": 2, "updates_per_epoch": 1},
        },
        "hisso",
    ],
)
def test_normalize_strategy_typed_mapping_and_preset_are_frozen(strategy):
    resolved = normalize_strategy(strategy)
    assert isinstance(resolved, HISSOConfig)
    assert isinstance(resolved.schedule, EpisodeScheduleConfig)
    with pytest.raises((AttributeError, TypeError)):
        resolved.schedule.episode_length = 9  # type: ignore[misc]


@pytest.mark.parametrize(
    "value, path",
    [
        ({"kind": "other"}, "strategy.kind"),
        ({"kind": "hisso", "unknown": 1}, "strategy.unknown"),
        (
            {"kind": "hisso", "schedule": {"episode_length": True}},
            "strategy.schedule.episode_length",
        ),
        (
            {"kind": "hisso", "schedule": {"episode_length": "4"}},
            "strategy.schedule.episode_length",
        ),
        ({"kind": "hisso", "transition_penalty": float("nan")}, "strategy.transition_penalty"),
    ],
)
def test_normalize_strategy_rejects_invalid_values_with_paths(value, path):
    with pytest.raises((TypeError, ValueError), match=path):
        normalize_strategy(value)


def test_width_one_softmax_preserves_samples_not_batch_axis():
    actions = transform_actions(np.array([2.0, -3.0, 7.0], dtype=np.float32), "softmax")
    np.testing.assert_allclose(actions, np.ones(3, dtype=np.float32))


def test_set_params_rebuilds_frozen_parents_transactionally():
    trainer = EpisodicTrainer(estimator=PSANNRegressor(), strategy="hisso")
    trainer.set_params(
        strategy__warm_start=SupervisedWarmStartConfig(epochs=1),
        strategy__warm_start__epochs=2,
        strategy__schedule__batch_episodes=3,
    )
    assert normalize_strategy(trainer.strategy).warm_start == SupervisedWarmStartConfig(epochs=2)
    assert normalize_strategy(trainer.strategy).schedule.batch_episodes == 3
    before = trainer.strategy
    with pytest.raises(ValueError, match="Unknown parameter"):
        trainer.set_params(strategy__schedule__missing=1)
    assert trainer.strategy == before


def test_canonical_trainer_schema_v3_custom_callable_two_generation_closure(tmp_path):
    X = np.arange(40, dtype=np.float32).reshape(20, 2) + 1
    strategy = HISSOConfig(
        schedule=EpisodeScheduleConfig(episode_length=4, batch_episodes=2, updates_per_epoch=1),
        reward=_custom_reward,
        primary_transform="tanh",
        warm_start=SupervisedWarmStartConfig(epochs=1),
    )
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=4, random_state=0), strategy=strategy
    ).fit(X, y=X.mean(axis=1))
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    trainer.save(first)
    payload = torch.load(first, weights_only=False)
    assert payload["schema_version"] == 3
    assert payload["fitted"]["episodic"]["config"]["reward"] == {"kind": "callable"}
    assert callable(payload["artifacts"]["episodic_reward"])
    loaded = EpisodicTrainer.load(first)
    np.testing.assert_allclose(loaded.predict(X[:3]), trainer.predict(X[:3]), rtol=1e-6)
    loaded.save(second)
    again = EpisodicTrainer.load(second)
    assert again.strategy == strategy
    np.testing.assert_allclose(again.predict(X[:3]), trainer.predict(X[:3]), rtol=1e-6)


@pytest.mark.parametrize("artifact", [None, 4])
def test_schema_v3_callable_descriptor_requires_callable_artifact(tmp_path, artifact):
    X = np.ones((8, 2), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=4, random_state=0),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2), reward=_custom_reward
        ),
    ).fit(X)
    path = tmp_path / "bad.pt"
    trainer.save(path)
    payload = torch.load(path, weights_only=False)
    if artifact is None:
        payload["artifacts"].pop("episodic_reward")
    else:
        payload["artifacts"]["episodic_reward"] = artifact
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Schema-v3 artifacts.episodic_reward"):
        PSANNRegressor.load(path)


def test_schema_v3_rejects_runtime_objects_in_portable_history_or_profile(tmp_path):
    X = np.ones((8, 2), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=4, random_state=0),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2)),
    ).fit(X)
    trainer.estimator._episodic_profile_ = {"optimizer": object()}
    with pytest.raises(TypeError, match="Schema-v3 fitted.episodic.profile.optimizer"):
        trainer.save(tmp_path / "not-portable.pt")


def test_schema_v3_requires_explicit_episodic_discriminator(tmp_path):
    X = np.ones((8, 2), dtype=np.float32)
    estimator = PSANNRegressor(epochs=1, batch_size=4, random_state=0).fit(X, X[:, 0])
    path = tmp_path / "missing-episodic.pt"
    estimator.save(path)
    payload = torch.load(path, weights_only=False)
    payload["fitted"].pop("episodic")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Schema-v3 fitted.episodic is missing"):
        PSANNRegressor.load(path)


@pytest.mark.parametrize("topology", ["dense", "conv2d"])
@pytest.mark.parametrize("trainable", [False, True])
def test_canonical_hisso_preprocessor_weight_policy_and_optimizer_membership(
    topology, trainable, monkeypatch
):
    """The canonical trainer retains Phase-4 group policy during real updates."""

    if topology == "dense":
        X = np.arange(48, dtype=np.float32).reshape(12, 4) / 10
        y = X.mean(axis=1)
        architecture = ArchitectureConfig.dense()
        component = LSMConfig.dense(output_dim=4, hidden_layers=1, hidden_units=4)
    else:
        X = np.arange(108, dtype=np.float32).reshape(12, 1, 3, 3) / 10
        y = X.mean(axis=(1, 2, 3))
        architecture = ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3))
        component = LSMConfig.convolutional(output_dim=2, hidden_units=3)
    estimator = PSANNRegressor(
        architecture=architecture,
        preprocessor=PreprocessorConfig(
            component, PreprocessorTrainingConfig(trainable=trainable, lr=0.02)
        ),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        lr=0.01,
        random_state=0,
        device="cpu",
    )
    trainer = EpisodicTrainer(
        estimator=estimator,
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=4, batch_episodes=2),
            warm_start=SupervisedWarmStartConfig(epochs=1, batch_size=4),
        ),
    )
    from psann.estimators import _fit_utils as fit_utils

    before: list[torch.Tensor] = []
    after: list[torch.Tensor] = []
    original_warm_start = fit_utils.run_hisso_supervised_warmstart

    def capture_warm_start(*args, **kwargs):
        model = args[0].model_
        before.extend(parameter.detach().clone() for parameter in model.preproc.parameters())
        result = original_warm_start(*args, **kwargs)
        after.extend(parameter.detach().clone() for parameter in model.preproc.parameters())
        return result

    monkeypatch.setattr(fit_utils, "run_hisso_supervised_warmstart", capture_warm_start)
    trainer.fit(X, y)
    groups = {
        group["psann_parameter_group"]: group
        for group in estimator._hisso_trainer_.optimizer.param_groups
    }
    preprocessor_ids = {id(parameter) for parameter in estimator.preprocessor_.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in estimator._hisso_trainer_.optimizer.param_groups
        for parameter in group["params"]
    }
    assert all(
        parameter.requires_grad is trainable for parameter in estimator.preprocessor_.parameters()
    )
    if trainable:
        assert groups["preprocessor"]["lr"] == pytest.approx(0.02)
        assert preprocessor_ids <= optimizer_ids
        assert any(not torch.equal(left, right) for left, right in zip(before, after))
    else:
        assert "preprocessor" not in groups
        assert preprocessor_ids.isdisjoint(optimizer_ids)
        assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_legacy_facades_warn_once_at_the_user_call_site():
    from psann.hisso import hisso_infer_series
    from psann.rewards import get_reward_strategy

    estimator = PSANNRegressor(epochs=1, batch_size=2).fit(
        np.ones((4, 2), dtype=np.float32), np.ones(4, dtype=np.float32)
    )
    with pytest.warns(DeprecationWarning) as caught:
        hisso_infer_series(estimator, np.ones((2, 2), dtype=np.float32))
    assert len(caught) == 1
    assert caught[0].filename.replace("\\", "/").endswith("tests/test_episodic_phase5.py")
    with pytest.warns(DeprecationWarning) as caught:
        get_reward_strategy("default")
    assert len(caught) == 1


def test_canonical_runtime_honors_schedule_clip_and_invalidation(monkeypatch):
    X = np.arange(24, dtype=np.float32).reshape(12, 2) + 1
    clip_values: list[float] = []
    from psann.episodic import runtime_loop

    original_clip = runtime_loop.clip_grad_norm_

    def capture_clip(parameters, value):
        clip_values.append(float(value))
        return original_clip(parameters, value)

    monkeypatch.setattr(runtime_loop, "clip_grad_norm_", capture_clip)
    estimator = PSANNRegressor(epochs=1, batch_size=2, random_state=11, device="cpu")
    trainer = EpisodicTrainer(
        estimator=estimator,
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(
                episode_length=3, batch_episodes=2, updates_per_epoch=2, random_state=99
            ),
            gradient_clip=0.25,
        ),
    ).fit(X)
    assert estimator._hisso_cfg_.random_state == 99
    assert trainer.profile_["updates_per_epoch"] == 2
    assert clip_values == [0.25, 0.25]
    trainer.set_params(strategy__schedule__random_state=7)
    for name in (
        "_episodic_strategy_",
        "_episodic_history_",
        "_episodic_profile_",
        "_hisso_trainer_",
        "_hisso_options_",
        "_hisso_cfg_",
        "_hisso_reward_fn_",
        "_hisso_context_extractor_",
    ):
        assert name not in estimator.__dict__


@pytest.mark.parametrize(
    "extractor, message",
    [
        (
            lambda data: np.ones((data.shape[0], data.shape[1], 3), dtype=np.float32),
            "must return a torch.Tensor",
        ),
        (
            lambda data: torch.ones((data.shape[0], data.shape[1], 3), device=data.device),
            "width mismatch",
        ),
    ],
)
def test_canonical_runtime_rejects_non_tensor_and_width_mismatched_context(extractor, message):
    X = np.ones((8, 2), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, device="cpu"),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2), context_extractor=extractor
        ),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        trainer.fit(X, np.ones((len(X), 2), dtype=np.float32))


def test_canonical_nonzero_penalty_requires_consuming_reward_before_runtime():
    X = np.ones((8, 2), dtype=np.float32)

    def reward(actions, context):
        return -(actions - context.mean(dim=-1, keepdim=True)).square().mean(dim=(-1, -2))

    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2), reward=reward, transition_penalty=0.5
        ),
    )
    with pytest.raises(ValueError, match="transition_penalty"):
        trainer.fit(X)
    assert "model_" not in trainer.estimator.__dict__


def test_schema_v3_registered_bundle_closes_two_generations(tmp_path):
    X = np.arange(20, dtype=np.float32).reshape(10, 2) + 1
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2), reward=FINANCE_PORTFOLIO_STRATEGY
        ),
    ).fit(X)
    first, second = tmp_path / "bundle-1.pt", tmp_path / "bundle-2.pt"
    trainer.save(first)
    payload = torch.load(first, weights_only=False)
    assert payload["fitted"]["episodic"]["config"]["reward"] == "finance"
    restored = EpisodicTrainer.load(first)
    restored.save(second)
    assert EpisodicTrainer.load(second).strategy.reward == "finance"
