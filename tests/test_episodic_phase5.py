"""Durable boundaries for the canonical Phase-5 episodic surface."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psann import PSANNRegressor
from psann.episodic import (
    EpisodeScheduleConfig,
    EpisodicTrainer,
    HISSOConfig,
    SupervisedWarmStartConfig,
    normalize_strategy,
)
from psann.episodic.runtime import transform_actions


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
