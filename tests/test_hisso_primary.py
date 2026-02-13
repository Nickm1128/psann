from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from psann import PSANNRegressor, ResConvPSANNRegressor, ResPSANNRegressor
from psann.hisso import (
    HISSOTrainer,
    HISSOTrainerConfig,
    coerce_warmstart_config,
    ensure_hisso_trainer_config,
    hisso_evaluate_reward,
    hisso_infer_series,
)
from psann.training import TrainingLoopConfig, run_training_loop

pytestmark = pytest.mark.slow


class _CountingStateful(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.reset_calls = 0
        self.commit_calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def reset_state(self) -> None:
        self.reset_calls += 1

    def commit_state_updates(self) -> None:
        self.commit_calls += 1


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
    assert fallback.filename.replace("\\", "/").endswith("tests/test_hisso_primary.py")
    assert calls["tensor"] >= 1
    assert calls["numpy"] >= 1


@pytest.mark.parametrize(
    ("state_reset", "expected_resets_per_epoch"),
    [
        ("batch", 4),
        ("epoch", 1),
        ("none", 0),
    ],
)
def test_hisso_trainer_stateful_hooks_respect_state_reset(
    state_reset: str,
    expected_resets_per_epoch: int,
) -> None:
    model = _CountingStateful()
    cfg = HISSOTrainerConfig(
        episode_length=5,
        episodes_per_batch=4,
        primary_dim=1,
        primary_transform="identity",
    )
    trainer = HISSOTrainer(
        model,
        cfg=cfg,
        device=torch.device("cpu"),
        lr=1e-2,
        reward_fn=None,
        context_extractor=None,
        input_noise_std=None,
        stateful=True,
        state_reset=state_reset,
    )
    X = np.linspace(-1.0, 1.0, 40, dtype=np.float32).reshape(-1, 1)
    epochs = 3
    trainer.train(X, epochs=epochs, verbose=0, lr_max=None, lr_min=None)

    expected_resets = epochs * expected_resets_per_epoch
    expected_commits = epochs * int(cfg.episodes_per_batch)
    assert model.reset_calls == expected_resets
    assert model.commit_calls == expected_commits


@pytest.mark.parametrize("state_reset", ["batch", "epoch", "none"])
def test_hisso_stateful_hooks_match_supervised_loop_pattern(state_reset: str) -> None:
    epochs = 2
    steps_per_epoch = 4

    supervised_model = _CountingStateful()
    inputs = torch.randn(steps_per_epoch * 5, 1)
    targets = 0.5 * inputs
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=5, shuffle=False)
    loop_cfg = TrainingLoopConfig(
        epochs=epochs,
        patience=1,
        early_stopping=False,
        stateful=True,
        state_reset=state_reset,
        verbose=0,
        lr_max=None,
        lr_min=None,
    )
    run_training_loop(
        supervised_model,
        optimizer=torch.optim.SGD(supervised_model.parameters(), lr=0.05),
        loss_fn=torch.nn.MSELoss(),
        train_loader=loader,
        device=torch.device("cpu"),
        cfg=loop_cfg,
    )

    hisso_model = _CountingStateful()
    hisso_cfg = HISSOTrainerConfig(
        episode_length=5,
        episodes_per_batch=steps_per_epoch,
        primary_dim=1,
        primary_transform="identity",
    )
    trainer = HISSOTrainer(
        hisso_model,
        cfg=hisso_cfg,
        device=torch.device("cpu"),
        lr=0.05,
        reward_fn=None,
        context_extractor=None,
        input_noise_std=None,
        stateful=True,
        state_reset=state_reset,
    )
    X_hisso = np.random.default_rng(123).standard_normal((64, 1)).astype(np.float32)
    trainer.train(X_hisso, epochs=epochs, verbose=0, lr_max=None, lr_min=None)

    assert hisso_model.commit_calls == supervised_model.commit_calls
    assert hisso_model.reset_calls == supervised_model.reset_calls


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
