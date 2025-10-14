import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from psann.training import TrainingLoopConfig, run_training_loop


def _clone_state_dict(module: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_run_training_loop_returns_best_state_for_early_stopping():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    initial_state = _clone_state_dict(model)

    inputs = torch.zeros(8, 3)
    targets = torch.zeros(8, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)

    cfg = TrainingLoopConfig(
        epochs=3,
        patience=2,
        early_stopping=True,
        stateful=False,
        state_reset="batch",
        verbose=0,
        lr_max=None,
        lr_min=None,
    )

    train_loss, best_state = run_training_loop(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        train_loader=loader,
        device=torch.device("cpu"),
        cfg=cfg,
        val_inputs=inputs.clone(),
        val_targets=targets.clone(),
    )

    assert isinstance(train_loss, float)
    assert best_state is not None
    for key, value in best_state.items():
        assert isinstance(value, torch.Tensor)
        assert not value.requires_grad
        assert value.device.type == "cpu"
        assert value.shape == initial_state[key].shape


class _DummyStateful(torch.nn.Module):
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


def test_run_training_loop_invokes_state_hooks_per_batch():
    torch.manual_seed(42)
    model = _DummyStateful()

    inputs = torch.randn(6, 1)
    targets = 0.5 * inputs
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=3, shuffle=False)

    cfg = TrainingLoopConfig(
        epochs=2,
        patience=1,
        early_stopping=False,
        stateful=True,
        state_reset="batch",
        verbose=0,
        lr_max=None,
        lr_min=None,
    )

    run_training_loop(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        loss_fn=torch.nn.MSELoss(),
        train_loader=loader,
        device=torch.device("cpu"),
        cfg=cfg,
    )

    batches_per_epoch = math.ceil(len(inputs) / loader.batch_size)
    expected_calls = cfg.epochs * batches_per_epoch

    assert model.reset_calls == expected_calls
    assert model.commit_calls == expected_calls
