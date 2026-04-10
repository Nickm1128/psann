from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from psann import PSANNRegressor, ResPSANNRegressor
from psann.hisso import HISSOTrainer, HISSOTrainerConfig

pytestmark = pytest.mark.slow


def test_hisso_noise_injection_records_rewards() -> None:
    rng = np.random.default_rng(11)
    X = rng.standard_normal((48, 6)).astype(np.float32)
    y = rng.standard_normal((48, 3)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=12,
        epochs=2,
        batch_size=12,
        lr=7e-4,
        random_state=11,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=12,
        hisso_reward_fn=lambda alloc, ctx: -(alloc.pow(2).mean(dim=-1)),
        hisso_primary_transform="tanh",
        noisy=0.1,
        verbose=0,
    )

    trainer = getattr(est, "_hisso_trainer_", None)
    assert isinstance(trainer, HISSOTrainer)
    assert trainer.input_noise_std == pytest.approx(0.1)
    assert trainer.history
    rewards = [entry.get("reward") for entry in trainer.history]
    assert all(np.isfinite(float(val)) for val in rewards if val is not None)
    profile_duration = float(trainer.profile.get("total_time_s", 0.0))
    assert profile_duration >= 0.0
    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert options.input_noise_std == pytest.approx(0.1)


def test_hisso_warmstart_short_series_truncates_episode() -> None:
    rng = np.random.default_rng(12)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    y = rng.standard_normal((10, 1)).astype(np.float32)

    est = ResPSANNRegressor(
        hidden_layers=1,
        hidden_units=10,
        epochs=3,
        batch_size=5,
        lr=8e-4,
        random_state=12,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=64,
        hisso_reward_fn=lambda alloc, ctx: -(alloc.pow(2).mean(dim=-1)),
        hisso_primary_transform="identity",
        hisso_supervised={"y": y, "epochs": 2, "batch_size": 5},
        verbose=0,
    )

    trainer = getattr(est, "_hisso_trainer_", None)
    assert isinstance(trainer, HISSOTrainer)
    assert trainer.history
    assert all(entry.get("episodes") == 1 for entry in trainer.history)
    cfg = getattr(est, "_hisso_cfg_", None)
    assert cfg is not None
    assert cfg.episode_length == X.shape[0]
    assert trainer.cfg.episode_length == X.shape[0]
    last_reward = trainer.history[-1].get("reward")
    assert last_reward is not None
    assert np.isfinite(float(last_reward))
    assert est.history_ == trainer.history


def test_hisso_profile_reports_batched_device_transfer() -> None:
    rng = np.random.default_rng(13)
    X = rng.standard_normal((48, 5)).astype(np.float32)
    y = rng.standard_normal((48, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=2,
        batch_size=12,
        lr=1e-3,
        random_state=13,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=12,
        hisso_reward_fn=lambda alloc, ctx: -(alloc.pow(2).mean(dim=-1)),
        verbose=0,
    )

    trainer = getattr(est, "_hisso_trainer_", None)
    assert isinstance(trainer, HISSOTrainer)
    profile = trainer.profile
    assert profile.get("dataset_transfer_batches") == 1
    assert profile.get("dataset_bytes") == int(X.nbytes)
    assert profile.get("episodes_sampled", 0) >= 1
    assert profile.get("episode_view_is_shared") is True
    transfer_time = float(profile.get("dataset_transfer_time_s", 0.0))
    assert transfer_time >= 0.0
    numpy_time = float(profile.get("dataset_numpy_to_tensor_time_s", 0.0))
    assert numpy_time >= 0.0


def test_hisso_seed_controls_episode_sampling() -> None:
    rng = np.random.default_rng(14)
    X = rng.standard_normal((64, 4)).astype(np.float32)
    y = rng.standard_normal((64, 1)).astype(np.float32)

    def fit_once() -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
        est = PSANNRegressor(
            hidden_layers=1,
            hidden_units=10,
            epochs=3,
            batch_size=16,
            lr=8e-4,
            random_state=14,
        )
        est.fit(
            X,
            y,
            hisso=True,
            hisso_window=16,
            hisso_reward_fn=lambda alloc, ctx: -(alloc.pow(2).mean(dim=-1)),
            verbose=0,
        )
        trainer = getattr(est, "_hisso_trainer_", None)
        assert isinstance(trainer, HISSOTrainer)
        preds = est.predict(X)
        return list(trainer.history), preds, dict(trainer.profile)

    history_a, preds_a, profile_a = fit_once()
    history_b, preds_b, profile_b = fit_once()

    assert len(history_a) == len(history_b)
    for entry_a, entry_b in zip(history_a, history_b):
        assert entry_a.get("epoch") == entry_b.get("epoch")
        assert entry_a.get("episodes") == entry_b.get("episodes")
        reward_a = entry_a.get("reward")
        reward_b = entry_b.get("reward")
        if reward_a is None or reward_b is None:
            assert reward_a is reward_b
        else:
            assert reward_a == pytest.approx(reward_b, rel=1e-6, abs=1e-8)

    np.testing.assert_allclose(preds_a, preds_b, rtol=1e-6, atol=1e-6)
    assert profile_a.get("episodes_sampled") == profile_b.get("episodes_sampled")
    assert profile_a.get("dataset_transfer_batches") == profile_b.get("dataset_transfer_batches")


def test_hisso_vectorized_updates_track_episode_batches() -> None:
    rng = np.random.default_rng(16)
    X = rng.standard_normal((64, 3)).astype(np.float32)
    model = torch.nn.Linear(3, 2)
    cfg = HISSOTrainerConfig(
        episode_length=8,
        episodes_per_batch=12,
        episode_batch_size=4,
        updates_per_epoch=3,
        primary_dim=2,
        primary_transform="identity",
        random_state=16,
    )
    seen_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def reward_fn(actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        seen_shapes.append((tuple(actions.shape), tuple(context.shape)))
        return -(actions.pow(2).mean(dim=-1).mean(dim=-1))

    trainer = HISSOTrainer(
        model,
        cfg=cfg,
        device=torch.device("cpu"),
        lr=1e-3,
        reward_fn=reward_fn,
        context_extractor=None,
        input_noise_std=None,
    )
    trainer.train(X, epochs=2, verbose=0, lr_max=None, lr_min=None)

    assert trainer.history
    assert all(entry.get("episodes") == 12 for entry in trainer.history)
    assert len(seen_shapes) == 2 * 3
    for action_shape, context_shape in seen_shapes:
        assert action_shape == (4, 8, 2)
        assert context_shape == (4, 8, 2)

    profile = trainer.profile
    assert profile.get("episode_batch_size") == 4
    assert profile.get("updates_per_epoch") == 3
    assert profile.get("episodes_sampled") == 24
    assert profile.get("episode_view_is_shared") is False
    assert float(profile.get("episode_gather_time_s_total", 0.0)) >= 0.0
    assert float(profile.get("forward_time_s_total", 0.0)) >= 0.0
    assert float(profile.get("reward_time_s_total", 0.0)) >= 0.0
    assert float(profile.get("backward_time_s_total", 0.0)) >= 0.0
    assert float(profile.get("optimizer_time_s_total", 0.0)) >= 0.0


def test_hisso_default_schedule_matches_explicit_compat_schedule() -> None:
    rng = np.random.default_rng(17)
    X = rng.standard_normal((96, 4)).astype(np.float32)
    torch.manual_seed(17)
    seed_model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 2),
    )
    init_state = {k: v.detach().clone() for k, v in seed_model.state_dict().items()}

    def run_once(cfg: HISSOTrainerConfig) -> list[dict[str, Any]]:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2),
        )
        model.load_state_dict(init_state)
        trainer = HISSOTrainer(
            model,
            cfg=cfg,
            device=torch.device("cpu"),
            lr=1e-3,
            reward_fn=lambda actions, ctx: -(actions.pow(2).mean(dim=-1).mean(dim=-1)),
            context_extractor=None,
            input_noise_std=None,
        )
        trainer.train(X, epochs=3, verbose=0, lr_max=None, lr_min=None)
        return trainer.history

    base_cfg = HISSOTrainerConfig(
        episode_length=12,
        episodes_per_batch=8,
        primary_dim=2,
        primary_transform="identity",
        random_state=17,
    )
    compat_cfg = HISSOTrainerConfig(
        episode_length=12,
        episodes_per_batch=8,
        episode_batch_size=1,
        updates_per_epoch=8,
        primary_dim=2,
        primary_transform="identity",
        random_state=17,
    )

    history_base = run_once(base_cfg)
    history_compat = run_once(compat_cfg)
    assert len(history_base) == len(history_compat)
    for entry_base, entry_compat in zip(history_base, history_compat):
        assert entry_base.get("episodes") == entry_compat.get("episodes")
        reward_base = entry_base.get("reward")
        reward_compat = entry_compat.get("reward")
        assert reward_base is not None and reward_compat is not None
        assert float(reward_base) == pytest.approx(float(reward_compat), rel=1e-7, abs=1e-9)


def test_hisso_vectorized_conv_inputs_align_context_shape() -> None:
    rng = np.random.default_rng(18)
    X = rng.standard_normal((80, 2, 4, 4)).astype(np.float32)
    model = torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(2 * 4 * 4, 2),
    )
    cfg = HISSOTrainerConfig(
        episode_length=10,
        episodes_per_batch=8,
        episode_batch_size=4,
        updates_per_epoch=2,
        primary_dim=2,
        primary_transform="identity",
        random_state=18,
    )
    context_shapes: list[tuple[int, ...]] = []

    def reward_fn(actions: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        context_shapes.append(tuple(context.shape))
        return -(actions.pow(2).mean(dim=-1).mean(dim=-1))

    trainer = HISSOTrainer(
        model,
        cfg=cfg,
        device=torch.device("cpu"),
        lr=1e-3,
        reward_fn=reward_fn,
        context_extractor=None,
        input_noise_std=None,
    )
    trainer.train(X, epochs=1, verbose=0, lr_max=None, lr_min=None)

    assert context_shapes
    assert all(shape == (4, 10, 2) for shape in context_shapes)
