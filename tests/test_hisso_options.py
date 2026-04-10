from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import torch

from psann import PSANNRegressor
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

    est.fit(X, y, hisso=True, hisso_window=8, hisso_reward_fn=reward_fn, verbose=0)

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


def test_hisso_fit_schedule_knobs_route_to_trainer_config() -> None:
    rng = np.random.default_rng(21)
    X = rng.standard_normal((48, 4)).astype(np.float32)
    y = rng.standard_normal((48, 2)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=10,
        epochs=2,
        batch_size=12,
        lr=1e-3,
        random_state=21,
    )

    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=8,
        hisso_batch_episodes=4,
        hisso_updates_per_epoch=3,
        hisso_reward_fn=lambda alloc, ctx: -(alloc.pow(2).mean(dim=-1).mean(dim=-1)),
        verbose=0,
    )

    options = getattr(est, "_hisso_options_", None)
    cfg = getattr(est, "_hisso_cfg_", None)
    trainer = getattr(est, "_hisso_trainer_", None)

    assert options is not None
    assert options.batch_episodes == 4
    assert options.updates_per_epoch == 3

    assert cfg is not None
    assert cfg.episode_batch_size == 4
    assert cfg.updates_per_epoch == 3
    assert cfg.episodes_per_batch == 12

    assert isinstance(trainer, HISSOTrainer)
    assert trainer.history
    assert all(entry.get("episodes") == 12 for entry in trainer.history)
    assert trainer.profile.get("episode_batch_size") == 4
    assert trainer.profile.get("updates_per_epoch") == 3


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
    preds = hisso_infer_series(est, X, trainer_cfg=cfg)

    np.testing.assert_allclose(preds, est.predict(X), rtol=1e-6, atol=1e-6)

    reward_val = hisso_evaluate_reward(est, X, trainer_cfg=cfg)
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
    cfg = ensure_hisso_trainer_config(
        {
            "episode_length": 10,
            "primary_dim": 2,
            "episode_batch_size": 4,
            "updates_per_epoch": 3,
        }
    )
    assert isinstance(cfg, HISSOTrainerConfig)
    assert cfg.episode_length == 10
    assert cfg.primary_dim == 2
    assert cfg.episode_batch_size == 4
    assert cfg.updates_per_epoch == 3


def test_hisso_fit_without_reward_uses_default() -> None:
    rng = np.random.default_rng(2)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = rng.standard_normal((20, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=5,
        lr=5e-4,
        random_state=2,
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
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        random_state=7,
    )
    with pytest.raises(ValueError):
        est.fit(X, y, hisso=True, hisso_primary_transform="bogus")


def test_hisso_transition_penalty_forwarded_to_reward_fn() -> None:
    rng = np.random.default_rng(14)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 1)).astype(np.float32)
    seen_penalties: list[float] = []

    def reward_fn(
        primary: torch.Tensor,
        _context: torch.Tensor,
        *,
        transition_penalty: float = 0.0,
    ) -> torch.Tensor:
        seen_penalties.append(float(transition_penalty))
        return -(primary.pow(2).mean(dim=-1)) - float(transition_penalty)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=6,
        epochs=1,
        batch_size=6,
        lr=1e-3,
        random_state=14,
    )
    penalty = 0.125
    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=6,
        hisso_transition_penalty=penalty,
        hisso_reward_fn=reward_fn,
        verbose=0,
    )

    assert seen_penalties
    for value in seen_penalties:
        assert value == pytest.approx(penalty)

    seen_penalties.clear()
    reward_val = hisso_evaluate_reward(est, X)
    assert isinstance(reward_val, float)
    assert seen_penalties
    assert seen_penalties[-1] == pytest.approx(penalty)


def test_hisso_trans_cost_alias_forwarded_to_reward_fn() -> None:
    rng = np.random.default_rng(15)
    X = rng.standard_normal((24, 3)).astype(np.float32)
    y = rng.standard_normal((24, 1)).astype(np.float32)
    seen_costs: list[float] = []

    def reward_fn(
        primary: torch.Tensor,
        _context: torch.Tensor,
        *,
        trans_cost: float = 0.0,
    ) -> torch.Tensor:
        seen_costs.append(float(trans_cost))
        return -(primary.pow(2).mean(dim=-1)) - float(trans_cost)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=6,
        epochs=1,
        batch_size=6,
        lr=1e-3,
        random_state=15,
    )
    penalty = 0.2
    est.fit(
        X,
        y,
        hisso=True,
        hisso_window=6,
        hisso_trans_cost=penalty,
        hisso_reward_fn=reward_fn,
        verbose=0,
    )

    assert seen_costs
    for value in seen_costs:
        assert value == pytest.approx(penalty)

    seen_costs.clear()
    reward_val = hisso_evaluate_reward(est, X)
    assert isinstance(reward_val, float)
    assert seen_costs
    assert seen_costs[-1] == pytest.approx(penalty)


def test_hisso_context_extractor_numpy_fallback_warns_once_with_user_stacklevel() -> None:
    rng = np.random.default_rng(19)
    X = rng.standard_normal((64, 3)).astype(np.float32)
    y = rng.standard_normal((64, 1)).astype(np.float32)
    calls = {"tensor": 0, "numpy": 0}

    def numpy_only_extractor(inputs: Any) -> np.ndarray:
        if isinstance(inputs, torch.Tensor):
            calls["tensor"] += 1
            raise TypeError("tensor inputs are not supported")
        calls["numpy"] += 1
        arr = np.asarray(inputs, dtype=np.float32)
        return arr.mean(axis=-1, keepdims=True)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=6,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        random_state=19,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(
            X,
            y,
            hisso=True,
            hisso_window=8,
            hisso_batch_episodes=4,
            hisso_updates_per_epoch=3,
            hisso_reward_fn=lambda alloc, ctx: -(alloc.pow(2).mean(dim=-1).mean(dim=-1)),
            hisso_context_extractor=numpy_only_extractor,
            verbose=0,
        )

    fallback_warnings = [
        warning
        for warning in caught
        if "context_extractor fell back to NumPy input" in str(warning.message)
    ]
    assert len(fallback_warnings) == 1
    fallback = fallback_warnings[0]
    assert issubclass(fallback.category, RuntimeWarning)
    message = str(fallback.message)
    assert "especially on CUDA" in message
    assert "accept torch.Tensor" in message
    assert "same device/dtype" in message
    assert fallback.filename.replace("\\", "/").endswith("tests/test_hisso_options.py")
    assert calls["tensor"] >= 1
    assert calls["numpy"] >= 1
