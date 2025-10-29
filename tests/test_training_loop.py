import math

import pytest
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


class _TrackingLinear(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(1, 1)
        self.commit_calls = 0
        self.load_calls = 0
        self.loaded_state_dict: dict | None = None

    def commit_state_updates(self) -> None:
        self.commit_calls += 1
        with torch.no_grad():
            self.weight.add_(0.05)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        self.load_calls += 1
        self.loaded_state_dict = {k: v.clone() for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict)


class _ScheduledLoss:
    def __init__(self, values) -> None:
        self.values = list(values)
        self.calls = 0

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        idx = min(self.calls, len(self.values) - 1)
        value = self.values[idx]
        self.calls += 1
        return pred.sum() * 0.0 + torch.tensor(value, dtype=pred.dtype, device=pred.device)


def test_training_loop_early_stopping_restores_best_state_and_calls_hooks():
    torch.manual_seed(1)
    model = _TrackingLinear()

    inputs = torch.zeros(1, 1)
    targets = torch.zeros(1, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=1, shuffle=False)

    cfg = TrainingLoopConfig(
        epochs=5,
        patience=1,
        early_stopping=True,
        stateful=False,
        state_reset="batch",
        verbose=0,
        lr_max=0.1,
        lr_min=0.05,
    )

    scheduled_loss = _ScheduledLoss([0.5, 0.4, 0.6, 0.7, 0.8])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    epoch_records = []
    lr_records = []
    improved_flags = []
    patience_records = []

    def epoch_callback(epoch, train_loss, val_loss, improved, patience_left):
        epoch_records.append((epoch, train_loss, val_loss))
        improved_flags.append(improved)
        patience_records.append(patience_left)
        lr_records.append(optimizer.param_groups[0]["lr"])

    grad_calls = 0

    def gradient_hook(module):
        nonlocal grad_calls
        grad_calls += 1

    train_loss, best_state = run_training_loop(
        model,
        optimizer=optimizer,
        loss_fn=scheduled_loss,
        train_loader=loader,
        device=torch.device("cpu"),
        cfg=cfg,
        gradient_hook=gradient_hook,
        epoch_callback=epoch_callback,
    )

    # Early stopping should trigger after the third epoch (index 2).
    assert len(epoch_records) == 3
    assert model.load_calls == 1
    assert grad_calls == 3  # one batch per epoch
    assert all(record[2] is None for record in epoch_records)  # no validation path used

    # Learning rate schedule should interpolate linearly between lr_max and lr_min.
    expected_lrs = [0.1, 0.0875, 0.075]
    assert lr_records == pytest.approx(expected_lrs, rel=1e-6, abs=1e-6)

    # Best state should reflect the epoch with lowest loss (second epoch).
    assert best_state is not None
    assert torch.allclose(model.weight, best_state["weight"])
    assert torch.allclose(model.bias, best_state["bias"])

    # Commit hook bumps weights each epoch; reload should bring them back to the saved best.
    assert model.commit_calls == 3
    assert scheduled_loss.calls == 3
    assert train_loss == pytest.approx(0.6, rel=1e-6)

    # Epoch callback flags should show improvements for first two epochs, then patience exhausted.
    assert improved_flags == [True, True, False]
    assert patience_records == [1, 1, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available")
def test_training_loop_early_stopping_runs_on_cuda():
    torch.manual_seed(7)
    device = torch.device("cuda")
    model = torch.nn.Linear(2, 1).to(device)

    inputs = torch.randn(12, 2)
    targets = torch.zeros(12, 1)
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)

    cfg = TrainingLoopConfig(
        epochs=4,
        patience=1,
        early_stopping=True,
        stateful=False,
        state_reset="batch",
        verbose=0,
        lr_max=None,
        lr_min=None,
    )

    scheduled_loss = _ScheduledLoss([0.5, 0.4, 0.6, 0.7])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_loss, best_state = run_training_loop(
        model,
        optimizer=optimizer,
        loss_fn=scheduled_loss,
        train_loader=loader,
        device=device,
        cfg=cfg,
    )

    # Early stopping should cut training after the third epoch.
    assert scheduled_loss.calls == 3
    assert train_loss == pytest.approx(0.6, rel=1e-6)

    assert best_state is not None
    for key, value in best_state.items():
        assert isinstance(value, torch.Tensor)
        assert not value.requires_grad
        assert value.device.type == "cpu"

    # Model parameters should match the stored best state once moved back to CPU.
    for name, param in model.state_dict().items():
        assert torch.allclose(param.detach().cpu(), best_state[name])
