"""Durable boundaries for the canonical Phase-5 episodic surface."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psann import PSANNRegressor, StateConfig
from psann.architectures import (
    AttentionConfig,
    ArchitectureConfig,
    ConvolutionConfig,
    ContextConfig,
    GeometryConfig,
    ResidualConfig,
)
from psann.episodic import (
    EpisodeScheduleConfig,
    EpisodicTrainer,
    HISSOConfig,
    SupervisedWarmStartConfig,
    normalize_strategy,
)
from psann.episodic.rewards import FINANCE_PORTFOLIO_STRATEGY, RewardStrategyBundle
from psann.episodic.runtime import transform_actions
from psann.episodic.runtime_loop import HISSOTrainer
from psann.episodic.legacy_config import HISSOTrainerConfig
from psann.preprocessing import (
    LSMConfig,
    ModulePreprocessorConfig,
    PreprocessorConfig,
    PreprocessorTrainingConfig,
)


def _custom_reward(actions: torch.Tensor, context: torch.Tensor, **_kwargs: object) -> torch.Tensor:
    return -(actions - context.mean(dim=-1, keepdim=True)).square().mean(dim=(-1, -2))


def _custom_context(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.reshape(inputs.shape[0], inputs.shape[1], -1).mean(dim=-1, keepdim=True)


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


def test_typed_hisso_config_normalizes_nested_mappings_without_mutating_callers():
    schedule = {"episode_length": 3, "batch_episodes": 2, "updates_per_epoch": 1}
    warm_start = {"epochs": 1, "shuffle": False}
    strategy = HISSOConfig(schedule=schedule, warm_start=warm_start)
    assert strategy.schedule == EpisodeScheduleConfig(**schedule)
    assert strategy.warm_start == SupervisedWarmStartConfig(**warm_start)
    assert schedule == {"episode_length": 3, "batch_episodes": 2, "updates_per_epoch": 1}
    assert warm_start == {"epochs": 1, "shuffle": False}


@pytest.mark.parametrize(
    ("kwargs", "path"),
    [
        ({"schedule": []}, "strategy.schedule"),
        ({"warm_start": {"bad": 1}}, "strategy.warm_start.bad"),
    ],
)
def test_typed_hisso_config_rejects_invalid_nested_mappings_with_paths(kwargs, path):
    with pytest.raises((TypeError, ValueError), match=path):
        HISSOConfig(**kwargs)


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


def test_canonical_wrapper_clone_and_deep_parameters_are_strategy_safe():
    from sklearn.base import clone

    strategy = HISSOConfig(
        schedule=EpisodeScheduleConfig(episode_length=3, batch_episodes=2, updates_per_epoch=1),
        warm_start=SupervisedWarmStartConfig(epochs=1, shuffle=False),
    )
    trainer = EpisodicTrainer(estimator=PSANNRegressor(epochs=1, batch_size=2), strategy=strategy)
    cloned = clone(trainer)
    assert cloned is not trainer
    assert cloned.strategy == strategy
    assert cloned.get_params(deep=True)["strategy__schedule__episode_length"] == 3
    assert cloned.get_params(deep=True)["strategy__warm_start__shuffle"] is False


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


def test_schema_v3_custom_context_callable_closes_two_generations(tmp_path):
    X = np.arange(40, dtype=np.float32).reshape(20, 2) + 1
    strategy = HISSOConfig(
        schedule=EpisodeScheduleConfig(episode_length=4, batch_episodes=2, updates_per_epoch=1),
        reward=_custom_reward,
        context_extractor=_custom_context,
    )
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=4, random_state=0), strategy=strategy
    ).fit(X)
    first, second = tmp_path / "context-1.pt", tmp_path / "context-2.pt"
    trainer.save(first)
    payload = torch.load(first, weights_only=False)
    assert payload["fitted"]["episodic"]["config"]["context_extractor"] == {"kind": "callable"}
    assert callable(payload["artifacts"]["episodic_context"])
    restored = EpisodicTrainer.load(first)
    restored.save(second)
    again = EpisodicTrainer.load(second)
    assert again.strategy.context_extractor is _custom_context
    np.testing.assert_allclose(again.predict(X[:3]), trainer.predict(X[:3]), rtol=1e-6)


def test_schema_v3_rejects_unregistered_reward_bundle_before_writing(tmp_path):
    X = np.ones((8, 2), dtype=np.float32)
    bundle = RewardStrategyBundle(_custom_reward, description="not registered")
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2), reward=bundle),
    ).fit(X)
    with pytest.raises(TypeError, match="unregistered RewardStrategyBundle"):
        trainer.save(tmp_path / "unregistered.pt")


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


@pytest.mark.parametrize(
    ("field", "descriptor", "artifact_key"),
    [
        ("reward", "default", "episodic_reward"),
        ("context_extractor", None, "episodic_context"),
    ],
)
def test_schema_v3_rejects_unexpected_callable_artifacts(tmp_path, field, descriptor, artifact_key):
    X = np.ones((8, 2), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=4, random_state=0),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2)),
    ).fit(X)
    path = tmp_path / f"unexpected-{field}.pt"
    trainer.save(path)
    payload = torch.load(path, weights_only=False)
    payload["fitted"]["episodic"]["config"][field] = descriptor
    payload["artifacts"][artifact_key] = _custom_reward
    torch.save(payload, path)
    with pytest.raises(ValueError, match=f"Schema-v3 artifacts.{artifact_key} is unexpected"):
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


def test_canonical_invalidation_clears_replaced_and_refit_estimator_runtimes():
    X = np.ones((8, 2), dtype=np.float32)
    strategy = HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=1))
    first = PSANNRegressor(epochs=1, batch_size=2, random_state=0)
    wrapper = EpisodicTrainer(estimator=first, strategy=strategy).fit(X)
    old_runtime = first._hisso_trainer_
    wrapper.fit(X)
    assert first._hisso_trainer_ is not old_runtime

    replacement = PSANNRegressor(epochs=1, batch_size=2, random_state=1)
    EpisodicTrainer(estimator=replacement, strategy=strategy).fit(X)
    wrapper.set_params(estimator=replacement)
    for estimator in (first, replacement):
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


@pytest.mark.parametrize("state_reset", ["epoch", "none"], ids=["epoch", "none"])
def test_canonical_stateful_warm_start_default_shuffle_is_resolved_before_training(
    monkeypatch, state_reset
):
    from psann.estimators import _fit_utils as fit_utils

    seen: list[bool | None] = []
    original = fit_utils.run_hisso_supervised_warmstart

    def capture(*args, **kwargs):
        seen.append(kwargs["config"].shuffle)
        return original(*args, **kwargs)

    monkeypatch.setattr(fit_utils, "run_hisso_supervised_warmstart", capture)
    X = np.ones((8, 2), dtype=np.float32)
    EpisodicTrainer(
        estimator=PSANNRegressor(
            epochs=1, batch_size=2, stateful=True, state_reset=state_reset, random_state=0
        ),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2),
            warm_start=SupervisedWarmStartConfig(epochs=1),
        ),
    ).fit(X, X[:, 0])
    assert seen == [False]


def test_canonical_unsupported_pairs_reject_before_episodic_optimization(monkeypatch):
    from psann.episodic import runtime_loop

    def unexpected_train(*_args, **_kwargs):
        raise AssertionError("episodic optimization must not start")

    monkeypatch.setattr(runtime_loop.HISSOTrainer, "train", unexpected_train)
    token_lsm = PreprocessorConfig(LSMConfig.dense(output_dim=4))
    attention = ArchitectureConfig.dense(attention=AttentionConfig(num_heads=1))
    with pytest.raises(ValueError, match="tokens-to-tokens"):
        EpisodicTrainer(
            estimator=PSANNRegressor(
                architecture=attention, preprocessor=token_lsm, epochs=1, batch_size=2
            ),
            strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2)),
        ).fit(np.ones((8, 2, 3), dtype=np.float32))

    per_element = ArchitectureConfig.convolutional(
        convolution=ConvolutionConfig(channels=3, per_element=True), residual=ResidualConfig()
    )
    with pytest.raises(ValueError, match="per_element"):
        EpisodicTrainer(
            estimator=PSANNRegressor(architecture=per_element, epochs=1, batch_size=2),
            strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2)),
        ).fit(np.ones((8, 1, 3, 3), dtype=np.float32))


def test_canonical_gradient_clip_none_is_effective_and_never_invokes_clipping(monkeypatch):
    from psann.episodic import runtime_loop

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("gradient clipping must be disabled")

    monkeypatch.setattr(runtime_loop, "clip_grad_norm_", fail_if_called)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, random_state=0, device="cpu"),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=1), gradient_clip=None
        ),
    ).fit(np.ones((6, 2), dtype=np.float32))
    assert trainer.profile_["gradient_clip"] is None


def test_canonical_stateful_warm_start_rejects_explicit_shuffle():
    X = np.ones((6, 2), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(
            epochs=1, batch_size=2, stateful=True, state_reset="epoch", device="cpu"
        ),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2),
            warm_start=SupervisedWarmStartConfig(epochs=1, shuffle=True),
        ),
    )
    with pytest.raises(ValueError, match="warm_start.shuffle=True"):
        trainer.fit(X, X[:, 0])


def test_canonical_wrapper_rejects_empty_input_before_episode_optimization():
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2)),
    )
    with pytest.raises(ValueError, match="non-empty"):
        trainer.fit(np.empty((0, 2), dtype=np.float32))


def test_canonical_warm_start_rejects_target_length_before_episodic_updates():
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2),
            warm_start=SupervisedWarmStartConfig(epochs=1),
        ),
    )
    with pytest.raises(ValueError, match="length must match X"):
        trainer.fit(np.ones((6, 2), dtype=np.float32), np.ones(5, dtype=np.float32))


def test_canonical_transform_is_applied_once_in_training_prediction_and_evaluation():
    X = np.arange(16, dtype=np.float32).reshape(8, 2) + 1
    observed: list[torch.Tensor] = []

    def reward(actions, context, **_kwargs):
        observed.append(actions.detach().cpu())
        return -(actions - context).square().mean(dim=(-1, -2))

    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, random_state=0, device="cpu"),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=1),
            reward=reward,
            primary_transform="softmax",
        ),
    ).fit(X, np.ones((len(X), 2), dtype=np.float32))
    raw = trainer.estimator.predict(X[:3])
    np.testing.assert_allclose(trainer.predict(X[:3]), transform_actions(raw, "softmax"), rtol=1e-6)
    assert np.isfinite(trainer.evaluate(X[:3]))
    assert len(observed) >= 2
    for actions in observed:
        torch.testing.assert_close(actions.sum(dim=-1), torch.ones_like(actions[..., 0]))


def test_canonical_evaluation_reuses_scaled_rank3_context_runtime_path():
    X = np.arange(24, dtype=np.float32).reshape(12, 2) + 2
    observed: list[tuple[int, float]] = []

    def context(inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.ndim == 3
        observed.append((inputs.ndim, float(inputs.mean())))
        return inputs

    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, scaler="standard", random_state=0),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=3, batch_episodes=1),
            context_extractor=context,
        ),
    ).fit(X)
    assert np.isfinite(trainer.evaluate(X))
    assert len(observed) >= 2
    assert all(rank == 3 for rank, _ in observed)
    assert abs(observed[-1][1]) < 1e-5


@pytest.mark.parametrize(
    ("primary_width", "context_width"),
    [(2, 2), (2, 1), (1, 3)],
    ids=["exact", "singleton-broadcast", "reduced-for-scalar-action"],
)
def test_canonical_context_alignment_is_identical_in_training_and_evaluation(
    primary_width, context_width
):
    """The strict runtime owns exact, broadcast, and scalar-reduction alignment."""

    X = np.arange(24, dtype=np.float32).reshape(12, 2) + 1
    received: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def context(inputs: torch.Tensor) -> torch.Tensor:
        base = inputs[..., :1]
        return base.expand(*base.shape[:-1], context_width)

    def reward(actions: torch.Tensor, aligned: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        received.append((tuple(actions.shape), tuple(aligned.shape)))
        return -(actions - aligned).square().mean(dim=(-1, -2))

    y = np.ones((len(X), primary_width), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, random_state=0, device="cpu"),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=3, batch_episodes=1),
            context_extractor=context,
            reward=reward,
            primary_transform="tanh",
        ),
    ).fit(X, y)
    assert trainer.predict(X[:4]).shape == (4, primary_width)
    assert np.isfinite(trainer.evaluate(X[:4]))
    assert len(received) >= 2
    assert all(action_shape == context_shape for action_shape, context_shape in received)


def test_legacy_reward_helper_delegates_to_the_runtime(monkeypatch):
    """The retained facade has no independent NumPy reward/context body."""

    from psann.hisso import hisso_evaluate_reward

    calls: list[tuple[int, ...]] = []
    original = HISSOTrainer.evaluate_prepared

    def capture(self, prepared):
        calls.append(tuple(np.asarray(prepared).shape))
        return original(self, prepared)

    monkeypatch.setattr(HISSOTrainer, "evaluate_prepared", capture)
    estimator = PSANNRegressor(epochs=1, batch_size=2, random_state=0).fit(
        np.ones((8, 2), dtype=np.float32), hisso=True, hisso_window=2
    )
    with pytest.warns(DeprecationWarning, match="psann.hisso"):
        assert np.isfinite(hisso_evaluate_reward(estimator, np.ones((4, 2), dtype=np.float32)))
    assert calls == [(4, 2)]


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


def test_schema_v3_portfolio_alias_persists_as_finance_across_generations(tmp_path):
    X = np.arange(20, dtype=np.float32).reshape(10, 2) + 1
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2), reward="PoRtFoLiO"),
    ).fit(X)
    first, second = tmp_path / "portfolio-1.pt", tmp_path / "portfolio-2.pt"
    trainer.save(first)
    assert (
        torch.load(first, weights_only=False)["fitted"]["episodic"]["config"]["reward"] == "finance"
    )
    loaded = EpisodicTrainer.load(first)
    loaded.save(second)
    assert (
        torch.load(second, weights_only=False)["fitted"]["episodic"]["config"]["reward"]
        == "finance"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["fitted"]["episodic"]["effective"].update(
                {"unknown": torch.tensor(1)}
            ),
            "effective.unknown",
        ),
        (
            lambda payload: payload["fitted"]["episodic"]["profile"].update({"bad": object()}),
            "profile.bad",
        ),
    ],
)
def test_schema_v3_rejects_nonportable_or_unknown_episodic_metadata(tmp_path, mutate, message):
    X = np.ones((8, 2), dtype=np.float32)
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2)),
    ).fit(X)
    path = tmp_path / "bad-metadata.pt"
    trainer.save(path)
    payload = torch.load(path, weights_only=False)
    mutate(payload)
    torch.save(payload, path)
    with pytest.raises((TypeError, ValueError), match=message):
        PSANNRegressor.load(path)


@pytest.mark.parametrize("legacy_version", [1, 2])
def test_legacy_episodic_schema_migration_closes_two_v3_generations(tmp_path, legacy_version):
    """v1/v2 HISSO metadata is rehydrated before the first new-format save."""

    X = np.arange(20, dtype=np.float32).reshape(10, 2) + 1
    legacy = PSANNRegressor(epochs=1, batch_size=2, random_state=5).fit(
        X,
        hisso=True,
        hisso_window=2,
        hisso_batch_episodes=2,
        hisso_updates_per_epoch=1,
    )
    source, first, second = (
        tmp_path / f"legacy-v{legacy_version}.pt",
        tmp_path / f"migrated-v{legacy_version}-1.pt",
        tmp_path / f"migrated-v{legacy_version}-2.pt",
    )
    legacy.save(source)
    payload = torch.load(source, weights_only=False)
    payload["schema_version"] = legacy_version
    payload["fitted"].pop("episodic")
    torch.save(payload, source)
    migrated = PSANNRegressor.load(source)
    migrated.save(first)
    loaded = EpisodicTrainer.load(first)
    loaded.save(second)
    assert loaded.strategy.schedule.episode_length == 2
    assert torch.load(second, weights_only=False)["schema_version"] == 3
    np.testing.assert_allclose(loaded.predict(X[:3]), EpisodicTrainer.load(second).predict(X[:3]))


def test_legacy_episode_facade_delegates_training_and_evaluation_to_runtime():
    """The deprecated facade is a warning-only adapter, not another runtime."""

    from psann.episodes import EpisodeConfig, EpisodeTrainer

    with pytest.warns(DeprecationWarning, match="EpisodeConfig"):
        config = EpisodeConfig(episode_length=3, batch_episodes=2, random_state=4)
    model = torch.nn.Linear(2, 1)
    with pytest.warns(DeprecationWarning, match="EpisodeTrainer"):
        trainer = EpisodeTrainer(model, ep_cfg=config, device="cpu")
    assert isinstance(trainer._runtime, HISSOTrainer)
    trainer.train(np.ones((8, 2), dtype=np.float32), epochs=1, verbose=0)
    assert isinstance(trainer.evaluate(np.ones((8, 2), dtype=np.float32), n_batches=2), float)


def test_strict_runtime_propagates_state_lifecycle_failures():
    class BrokenState(torch.nn.Linear):
        def reset_state(self) -> None:
            raise RuntimeError("state reset failed")

    runtime = HISSOTrainer(
        BrokenState(2, 1),
        cfg=HISSOTrainerConfig(episode_length=2, episodes_per_batch=1),
        device=torch.device("cpu"),
        lr=0.01,
        reward_fn=None,
        context_extractor=None,
        input_noise_std=None,
        stateful=True,
        state_reset="epoch",
        strict=True,
    )
    with pytest.raises(RuntimeError, match="state reset failed"):
        runtime.train(
            np.ones((4, 2), dtype=np.float32), epochs=1, verbose=0, lr_max=None, lr_min=None
        )


@pytest.mark.parametrize(
    ("architecture", "X"),
    [
        (ArchitectureConfig.dense(), np.ones((8, 2), dtype=np.float32)),
        (
            ArchitectureConfig.dense(residual=ResidualConfig()),
            np.ones((8, 2), dtype=np.float32),
        ),
        (
            ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3)),
            np.ones((8, 1, 3, 3), dtype=np.float32),
        ),
        (ArchitectureConfig.for_wave(), np.ones((8, 2), dtype=np.float32)),
        (ArchitectureConfig.for_sequence(), np.ones((8, 2), dtype=np.float32)),
        (
            ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(1, 2))),
            np.ones((8, 2), dtype=np.float32),
        ),
    ],
)
def test_canonical_episodic_architecture_runtime_matrix(architecture, X):
    """Each retained topology reaches canonical fit, transform, prediction and reward."""

    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(
            architecture=architecture,
            hidden_layers=1,
            hidden_units=4,
            epochs=1,
            batch_size=2,
            random_state=3,
            device="cpu",
        ),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=2),
            primary_transform="tanh",
        ),
    ).fit(X)
    assert trainer.predict(X[:2]).shape[0] == 2
    assert np.isfinite(trainer.evaluate(X[:4]))
    assert trainer.history_[0]["episodes"] == 2


@pytest.mark.parametrize(
    ("name", "architecture", "preprocessor", "X"),
    [
        (
            "dense-attention-custom-tokens",
            ArchitectureConfig.dense(attention=AttentionConfig(num_heads=1)),
            lambda: PreprocessorConfig(
                ModulePreprocessorConfig(torch.nn.Linear(3, 4), "tokens", "tokens", 4)
            ),
            np.ones((8, 2, 3), dtype=np.float32),
        ),
        (
            "wave-attention-custom-tokens",
            ArchitectureConfig.for_wave(attention=AttentionConfig(num_heads=1)),
            lambda: PreprocessorConfig(
                ModulePreprocessorConfig(torch.nn.Linear(3, 4), "tokens", "tokens", 4)
            ),
            np.ones((8, 2, 3), dtype=np.float32),
        ),
        (
            "sequence-custom-tokens",
            ArchitectureConfig.for_sequence(),
            lambda: PreprocessorConfig(
                ModulePreprocessorConfig(torch.nn.Linear(3, 4), "tokens", "tokens", 4)
            ),
            np.ones((8, 2, 3), dtype=np.float32),
        ),
        (
            "conv1d-channels-first",
            ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3)),
            lambda: None,
            np.ones((8, 1, 4), dtype=np.float32),
        ),
        (
            "conv1d-channels-last",
            ArchitectureConfig.convolutional(
                convolution=ConvolutionConfig(channels=3, data_format="channels_last")
            ),
            lambda: None,
            np.ones((8, 4, 1), dtype=np.float32),
        ),
        (
            "conv2d-channels-last",
            ArchitectureConfig.convolutional(
                convolution=ConvolutionConfig(channels=3, data_format="channels_last")
            ),
            lambda: None,
            np.ones((8, 3, 3, 1), dtype=np.float32),
        ),
        (
            "conv3d-channels-first",
            ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3)),
            lambda: None,
            np.ones((8, 1, 2, 2, 2), dtype=np.float32),
        ),
        (
            "residual-convolution",
            ArchitectureConfig.convolutional(
                convolution=ConvolutionConfig(channels=3), residual=ResidualConfig()
            ),
            lambda: None,
            np.ones((8, 1, 3, 3), dtype=np.float32),
        ),
        (
            "wave-convolution-context-channels-last",
            ArchitectureConfig.for_wave(
                convolution=ConvolutionConfig(channels=3, data_format="channels_last"),
                context=ContextConfig(builder="cosine", builder_params={"include_sin": False}),
            ),
            lambda: None,
            np.ones((8, 3, 3, 2), dtype=np.float32),
        ),
        (
            "geometric-dense-lsm",
            ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(2, 2))),
            lambda: PreprocessorConfig(LSMConfig.dense(output_dim=4)),
            np.ones((8, 4), dtype=np.float32),
        ),
        (
            "geometric-custom-flat",
            ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(2, 2))),
            lambda: PreprocessorConfig(
                ModulePreprocessorConfig(torch.nn.Linear(4, 4), "flat", "flat", 4)
            ),
            np.ones((8, 4), dtype=np.float32),
        ),
    ],
    ids=[
        "dense-attention-custom-tokens",
        "wave-attention-custom-tokens",
        "sequence-custom-tokens",
        "conv1d-channels-first",
        "conv1d-channels-last",
        "conv2d-channels-last",
        "conv3d-channels-first",
        "residual-convolution",
        "wave-convolution-context-channels-last",
        "geometric-dense-lsm",
        "geometric-custom-flat",
    ],
)
def test_canonical_episodic_topology_and_preprocessor_matrix(name, architecture, preprocessor, X):
    """Every retained Phase-5 topology executes wrapper fit/predict/evaluate."""

    del name
    trainer = EpisodicTrainer(
        estimator=PSANNRegressor(
            architecture=architecture,
            preprocessor=preprocessor(),
            hidden_layers=1,
            hidden_units=4,
            epochs=1,
            batch_size=2,
            random_state=0,
            device="cpu",
        ),
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=1)),
    ).fit(X)
    assert trainer.predict(X[:2]).shape[0] == 2
    assert np.isfinite(trainer.evaluate(X[:4]))


@pytest.mark.parametrize(
    ("state_reset", "expected_resets"),
    [("batch", 4), ("epoch", 2), ("none", 0)],
)
def test_canonical_runtime_state_cadence_and_commit_counts(state_reset, expected_resets):
    class CountingState(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 1)
            self.resets = 0
            self.commits = 0

        def reset_state(self) -> None:
            self.resets += 1

        def commit_state_updates(self) -> None:
            self.commits += 1

    model = CountingState()
    runtime = HISSOTrainer(
        model,
        cfg=HISSOTrainerConfig(
            episode_length=2,
            episodes_per_batch=2,
            episode_batch_size=1,
            updates_per_epoch=2,
        ),
        device=torch.device("cpu"),
        lr=0.01,
        reward_fn=None,
        context_extractor=None,
        input_noise_std=None,
        stateful=True,
        state_reset=state_reset,
        strict=True,
    )
    runtime.train(np.ones((6, 2), dtype=np.float32), epochs=2, verbose=0, lr_max=None, lr_min=None)
    assert model.resets == expected_resets
    assert model.commits == 4
    assert runtime.profile["episodes_sampled"] == 4


def test_canonical_and_legacy_schedule_counts_are_explicit_and_deterministic():
    X = np.arange(32, dtype=np.float32).reshape(16, 2) + 1
    canonical = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, random_state=17, device="cpu"),
        strategy=HISSOConfig(
            schedule=EpisodeScheduleConfig(
                episode_length=3, batch_episodes=2, updates_per_epoch=2, random_state=17
            )
        ),
    ).fit(X)
    with pytest.warns(DeprecationWarning, match=r"fit\(\.\.\., hisso=True\)"):
        legacy = PSANNRegressor(epochs=1, batch_size=2, random_state=17, device="cpu").fit(
            X,
            hisso=True,
            hisso_window=3,
            hisso_batch_episodes=2,
            hisso_updates_per_epoch=2,
        )
    assert (
        canonical.profile_["episodes_sampled"]
        == legacy._hisso_trainer_.profile["episodes_sampled"]
        == 4
    )
    assert canonical.estimator._hisso_cfg_.random_state == legacy._hisso_cfg_.random_state == 17
    assert canonical.history_[0]["episodes"] == legacy._hisso_trainer_.history[0]["episodes"] == 4


def test_canonical_short_series_schedule_uses_local_rng_and_preserves_episode_count():
    X = np.arange(6, dtype=np.float32).reshape(3, 2) + 1
    strategy = HISSOConfig(
        schedule=EpisodeScheduleConfig(
            episode_length=8, batch_episodes=3, updates_per_epoch=2, random_state=29
        )
    )
    first = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, random_state=1), strategy=strategy
    ).fit(X)
    second = EpisodicTrainer(
        estimator=PSANNRegressor(epochs=1, batch_size=2, random_state=1), strategy=strategy
    ).fit(X)
    assert first.history_ == second.history_
    assert first.history_[0]["episodes"] == 1
    assert first.profile_["episodes_sampled"] == 1


def test_canonical_stateful_prediction_uses_clean_noncommitting_sequence_route(monkeypatch):
    X = np.arange(16, dtype=np.float32).reshape(8, 2) / 10
    estimator = PSANNRegressor(
        epochs=1,
        batch_size=2,
        random_state=0,
        stateful=True,
        state=StateConfig(rho=0.9),
        state_reset="batch",
    )
    trainer = EpisodicTrainer(
        estimator=estimator,
        strategy=HISSOConfig(schedule=EpisodeScheduleConfig(episode_length=2, batch_episodes=1)),
    ).fit(X)
    calls: list[dict[str, object]] = []
    original = estimator.predict_sequence

    def capture(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator, "predict_sequence", capture)
    assert trainer.predict(X[:3]).shape[0] == 3
    assert calls == [
        {"context": None, "reset_state": True, "return_sequence": True, "update_state": False}
    ]
