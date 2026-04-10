from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from psann.hisso import HISSOTrainer, HISSOTrainerConfig
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


@pytest.mark.parametrize(
    ("state_reset", "expected_resets_per_epoch"),
    [("batch", 4), ("epoch", 1), ("none", 0)],
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
