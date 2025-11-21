from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from psann import PSANNRegressor, ResConvPSANNRegressor, ResPSANNRegressor
from psann.hisso import (
    HISSOTrainer,
    HISSOTrainerConfig,
    coerce_warmstart_config,
    ensure_hisso_trainer_config,
    hisso_evaluate_reward,
    hisso_infer_series,
)

pytestmark = pytest.mark.slow


def test_coerce_warmstart_config_requires_targets() -> None:
    with pytest.raises(ValueError):
        coerce_warmstart_config({"epochs": 2}, y_default=None)

    cfg = coerce_warmstart_config({"y": np.zeros((3, 1))}, y_default=None)
    assert cfg is not None
    assert cfg.targets.shape == (3, 1)


def test_hisso_fit_sets_trainer_state() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 4)).astype(np.float32)
    y = rng.standard_normal((32, 1)).astype(np.float32)

    def reward_fn(primary: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return -(primary.pow(2).mean(dim=-1))

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        random_state=0,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=8,
        hisso_reward_fn=reward_fn,
        verbose=0,
    )

    trainer = getattr(est, "_hisso_trainer_", None)
    assert isinstance(trainer, HISSOTrainer)
    assert getattr(est, "_hisso_trained_", False) is True
    assert est.history_ == trainer.history
    assert trainer.history
    assert isinstance(trainer.history[-1].get("reward"), (float, type(None)))
    assert len(trainer.history) == est.epochs
    assert trainer.history[-1].get("episodes", 0) >= 1
    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert options.reward_fn is reward_fn
    assert options.primary_transform == "softmax"


def test_hisso_infer_series_matches_predict() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=6,
        epochs=1,
        batch_size=6,
        lr=1e-3,
        random_state=1,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=6,
        hisso_reward_fn=lambda alloc, ctx: -(alloc.pow(2).mean(dim=-1)),
    )

    cfg = HISSOTrainerConfig(episode_length=6, primary_dim=1)
    preds = hisso_infer_series(
        est,
        X,
        trainer_cfg=cfg,
    )

    np.testing.assert_allclose(preds, est.predict(X), rtol=1e-6, atol=1e-6)

    reward_val = hisso_evaluate_reward(
        est,
        X,
        trainer_cfg=cfg,
    )
    assert isinstance(reward_val, float)
    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    stored_reward_fn = options.reward_fn
    context_extractor = options.context_extractor
    assert options.primary_transform == "softmax"
    device = est._device()
    context_tensor = torch.from_numpy(X).to(device)
    if context_extractor is not None:
        ctx_val = context_extractor(context_tensor)
        if isinstance(ctx_val, tuple):
            ctx_val = ctx_val[0]
        context_tensor = ctx_val.detach()
    else:
        context_tensor = context_tensor.detach()
    primary_tensor = torch.from_numpy(preds).to(device)
    expected_reward = stored_reward_fn(primary_tensor, context_tensor)
    if isinstance(expected_reward, torch.Tensor):
        expected_value = float(expected_reward.mean().detach().cpu().item())
    else:
        expected_value = float(expected_reward)
    assert pytest.approx(expected_value, rel=1e-6) == reward_val


def test_hisso_primary_transform_tanh_propagates_to_evaluation() -> None:
    rng = np.random.default_rng(5)
    X = rng.standard_normal((16, 2)).astype(np.float32)
    y = rng.standard_normal((16, 1)).astype(np.float32)

    def reward_fn(primary: torch.Tensor, _context: torch.Tensor) -> torch.Tensor:
        return primary.mean(dim=-1)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=6,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        random_state=5,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=4,
        hisso_reward_fn=reward_fn,
        hisso_primary_transform="tanh",
        verbose=0,
    )

    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert options.primary_transform == "tanh"

    preds = est.predict(X)
    transformed = np.tanh(preds)
    inferred = hisso_infer_series(est, X)
    np.testing.assert_allclose(inferred, transformed, rtol=1e-6, atol=1e-6)

    device = est._device()
    manual_reward = reward_fn(
        torch.from_numpy(transformed).to(device),
        torch.from_numpy(X).to(device),
    )
    expected_value = float(manual_reward.mean().detach().cpu().item())

    reward_without_cfg = hisso_evaluate_reward(est, X)
    reward_with_cfg = hisso_evaluate_reward(est, X, trainer_cfg=est._hisso_cfg_)
    assert pytest.approx(expected_value, rel=1e-6) == reward_without_cfg
    assert pytest.approx(expected_value, rel=1e-6) == reward_with_cfg


def test_ensure_hisso_trainer_config_from_mapping() -> None:
    cfg = ensure_hisso_trainer_config({"episode_length": 10, "primary_dim": 2})
    assert isinstance(cfg, HISSOTrainerConfig)
    assert cfg.episode_length == 10
    assert cfg.primary_dim == 2


def test_hisso_fit_without_reward_uses_default() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = rng.standard_normal((20, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1, hidden_units=4, epochs=1, batch_size=5, lr=5e-4, random_state=2
    )
    est.fit(X, y, hisso=True, hisso_window=5)

    trainer = getattr(est, "_hisso_trainer_", None)
    assert isinstance(trainer, HISSOTrainer)
    assert trainer.history
    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert callable(options.reward_fn)


def test_hisso_primary_transform_validation() -> None:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((8, 2)).astype(np.float32)
    y = rng.standard_normal((8, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1, hidden_units=4, epochs=1, batch_size=4, lr=1e-3, random_state=7
    )
    with pytest.raises(ValueError):
        est.fit(
            X,
            y,
            hisso=True,
            hisso_primary_transform="bogus",
        )


def test_respsann_hisso_uses_shared_options() -> None:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((24, 5)).astype(np.float32)
    y = rng.standard_normal((24, 1)).astype(np.float32)

    est = ResPSANNRegressor(
        hidden_layers=1,
        hidden_units=10,
        epochs=1,
        batch_size=6,
        lr=1e-3,
        random_state=8,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=6,
        hisso_primary_transform="tanh",
    )

    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert options.primary_transform == "tanh"

    preds = est.predict(X)
    transformed = np.tanh(preds)
    inferred = hisso_infer_series(est, X)
    np.testing.assert_allclose(inferred, transformed, rtol=1e-6, atol=1e-6)


def test_resconv_hisso_preserve_shape_routes_options() -> None:
    rng = np.random.default_rng(9)
    X = rng.standard_normal((16, 2, 4, 4)).astype(np.float32)
    y = rng.standard_normal((16, 1)).astype(np.float32)

    est = ResConvPSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        conv_channels=4,
        epochs=1,
        batch_size=4,
        lr=5e-4,
        random_state=9,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=4,
        hisso_primary_transform="identity",
    )

    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert options.primary_transform == "identity"
    preds = est.predict(X)
    inferred = hisso_infer_series(est, X)
    np.testing.assert_allclose(inferred, preds, rtol=1e-6, atol=1e-6)


def test_resconv_hisso_rejects_per_element_mode() -> None:
    rng = np.random.default_rng(10)
    X = rng.standard_normal((12, 1, 4, 4)).astype(np.float32)
    y = rng.standard_normal((12, 1, 4, 4)).astype(np.float32)

    est = ResConvPSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        conv_channels=4,
        per_element=True,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        random_state=10,
    )

    with pytest.raises(ValueError, match="per_element=False"):
        est.fit(
            X,
            y,
            hisso=True,
            hisso_window=4,
        )


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
        history = list(trainer.history)
        profile = dict(trainer.profile)
        return history, preds, profile

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
