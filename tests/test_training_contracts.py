from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from psann import PSANNRegressor
from psann.training import TrainingLoopConfig, run_training_loop


def _data(samples: int = 16, features: int = 2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    X = rng.normal(size=(samples, features)).astype(np.float32)
    y = (0.5 * X.sum(axis=1, keepdims=True) - 0.1).astype(np.float32)
    return X, y


@pytest.mark.parametrize(
    "loss",
    ["mse", "l2", "l1", "mae", "smooth_l1", "huber_smooth", "huber"],
)
@pytest.mark.parametrize("reduction", ["mean", "sum"])
def test_documented_builtin_losses_and_scalar_reductions_train(loss, reduction):
    X, y = _data()
    estimator = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=8,
        random_state=4,
        loss=loss,
        loss_reduction=reduction,
    )

    estimator.fit(X, y)

    assert math.isfinite(estimator.history_[0]["train_loss"])


def test_loss_reduction_none_fails_before_model_construction():
    X, y = _data()
    estimator = PSANNRegressor(epochs=1, loss_reduction="none")

    with pytest.raises(ValueError, match="optimizer-driven training"):
        estimator.fit(X, y)

    assert not hasattr(estimator, "model_")


@pytest.mark.parametrize(
    ("estimator_kwargs", "fit_kwargs", "message"),
    [
        ({"optimizer": "adma"}, {}, "Unknown optimizer"),
        ({"loss": "mystery"}, {}, "Unknown loss"),
        (
            {"loss": "mse", "loss_params": {"beta": 1.0}},
            {},
            "Unsupported loss_params",
        ),
        ({"activation_type": "swish"}, {}, "Unknown activation_type"),
        ({"batch_size": 0}, {}, "batch_size must be"),
        ({"lr": 0.0}, {}, "lr must be"),
        ({"early_stopping": True, "patience": 0}, {}, "patience must be"),
        ({}, {"scheduler": "plateau"}, "scheduler must be"),
        ({}, {"scheduler_params": {"gamma": 0.5}}, "scheduler_params requires"),
        ({}, {"lr_max": 0.1}, "provided together"),
        (
            {},
            {"scheduler": "step", "lr_max": 0.1, "lr_min": 0.01},
            "cannot be combined",
        ),
        ({"amp": True, "amp_dtype": "fp32"}, {}, "amp_dtype must be"),
        ({"compile_backend": ""}, {}, "compile_backend must be"),
    ],
)
def test_invalid_training_configuration_fails_before_model_construction(
    estimator_kwargs,
    fit_kwargs,
    message,
):
    X, y = _data()
    estimator = PSANNRegressor(epochs=1, **estimator_kwargs)

    with pytest.raises(ValueError, match=message):
        estimator.fit(X, y, **fit_kwargs)

    assert not hasattr(estimator, "model_")


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("X", np.nan, "missing values"),
        ("X", np.inf, "infinite values"),
        ("y", np.nan, "missing values"),
        ("y", -np.inf, "infinite values"),
    ],
)
def test_nonfinite_training_data_is_rejected_at_the_boundary(target, value, message):
    X, y = _data()
    if target == "X":
        X[0, 0] = value
    else:
        y[0, 0] = value
    estimator = PSANNRegressor(epochs=1)

    with pytest.raises(ValueError, match=message):
        estimator.fit(X, y)

    assert not hasattr(estimator, "model_")


class _WrongOutputRegressor(PSANNRegressor):
    def _build_dense_core(
        self,
        input_dim: int,
        output_dim: int,
        **_: Any,
    ) -> torch.nn.Module:
        return torch.nn.Linear(input_dim, output_dim + 1)


def test_prediction_target_shape_is_checked_before_optimizer_creation():
    X, y = _data()
    estimator = _WrongOutputRegressor(epochs=1)

    with pytest.raises(ValueError, match="before the first optimizer step"):
        estimator.fit(X, y)

    assert estimator._optimizer_ is None


def test_warm_start_accepts_compatible_fit_and_rejects_shape_changes():
    X, y = _data()
    estimator = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=8,
        random_state=9,
        warm_start=True,
    )
    estimator.fit(X, y)
    model_identity = id(estimator.model_)
    before = {
        name: parameter.detach().clone()
        for name, parameter in estimator.model_.state_dict().items()
    }

    estimator.fit(X, y)

    assert id(estimator.model_) == model_identity
    assert estimator._model_rebuilt_ is False
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in estimator.model_.state_dict().items()
    )

    X_incompatible = np.pad(X, ((0, 0), (0, 1)))
    with pytest.raises(ValueError, match="warm_start=True is incompatible"):
        estimator.fit(X_incompatible, y)


class _AlternatingLoss:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.calls == 1:
            return prediction.sum() * torch.tensor(float("nan"))
        return torch.nn.functional.mse_loss(prediction, target)


class _FiniteLossWithNonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction):
        ctx.prediction_shape = prediction.shape
        return prediction.sum() * 0.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(
            ctx.prediction_shape,
            float("nan"),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )


def _loop_config(**overrides: Any) -> TrainingLoopConfig:
    values = {
        "epochs": 1,
        "patience": 1,
        "early_stopping": False,
        "stateful": False,
        "state_reset": "batch",
        "verbose": 0,
        "lr_max": None,
        "lr_min": None,
    }
    values.update(overrides)
    return TrainingLoopConfig(**values)


def _two_batch_loop(policy: str):
    torch.manual_seed(3)
    model = torch.nn.Linear(1, 1)
    inputs = torch.arange(4, dtype=torch.float32).reshape(-1, 1)
    targets = 0.25 * inputs
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    events = []
    result = run_training_loop(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=_AlternatingLoss(),
        train_loader=loader,
        device=torch.device("cpu"),
        cfg=_loop_config(nonfinite_policy=policy),
        callbacks=[events.append],
    )
    return model, result, events


def test_nonfinite_error_policy_fails_immediately_and_emits_event():
    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        _two_batch_loop("error")


def test_nonfinite_skip_step_policy_skips_only_the_bad_batch():
    _, (history, _), events = _two_batch_loop("skip_step")

    assert history[0]["attempted_steps"] == 2
    assert history[0]["steps"] == 1
    assert history[0]["skipped_nonfinite_steps"] == 1
    assert any(event.name == "nonfinite_step" for event in events)


def test_nonfinite_continue_policy_is_explicit_and_observable():
    model, (history, _), events = _two_batch_loop("continue")

    assert history[0]["loss_nonfinite_steps"] >= 1
    assert any(event.name == "nonfinite_step" for event in events)
    assert any(not torch.isfinite(parameter).all() for parameter in model.parameters())


def test_nonfinite_gradient_is_detected_before_optimizer_step():
    model = torch.nn.Linear(1, 1)
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.ones(2, 1)),
        batch_size=2,
    )

    def nonfinite_gradient_loss(
        prediction: torch.Tensor,
        _: torch.Tensor,
    ) -> torch.Tensor:
        return _FiniteLossWithNonfiniteGradient.apply(prediction)

    with pytest.raises(FloatingPointError, match="Non-finite gradient"):
        run_training_loop(
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=nonfinite_gradient_loss,
            train_loader=loader,
            device=torch.device("cpu"),
            cfg=_loop_config(nonfinite_policy="error"),
        )


def test_callback_and_gradient_hook_failures_raise_by_default():
    model = torch.nn.Linear(1, 1)
    loader = DataLoader(
        TensorDataset(torch.ones(2, 1), torch.ones(2, 1)),
        batch_size=2,
    )

    def fail_hook(_: torch.nn.Module) -> None:
        raise RuntimeError("hook failed")

    with pytest.raises(RuntimeError, match="hook failed"):
        run_training_loop(
            model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
            loss_fn=torch.nn.MSELoss(),
            train_loader=loader,
            device=torch.device("cpu"),
            cfg=_loop_config(),
            gradient_hook=fail_hook,
        )


def test_callback_warning_policy_keeps_training_observable():
    X, y = _data()
    events = []

    def fail_epoch_end(event) -> None:
        events.append(event.name)
        if event.name == "epoch_end":
            raise RuntimeError("observer failed")

    estimator = PSANNRegressor(epochs=1, hidden_layers=1, hidden_units=8)
    with pytest.warns(RuntimeWarning, match="observer failed"):
        estimator.fit(
            X,
            y,
            callbacks=[fail_epoch_end],
            callback_error_policy="warn",
        )

    assert "train_end" in events
    assert any(event["name"] == "epoch_end" for event in estimator.training_events_)


def test_callback_failure_emits_failure_event_before_reraising():
    X, y = _data()

    def fail_epoch_end(event) -> None:
        if event.name == "epoch_end":
            raise RuntimeError("observer failed")

    estimator = PSANNRegressor(epochs=1, hidden_layers=1, hidden_units=8)
    with pytest.raises(RuntimeError, match="observer failed"):
        estimator.fit(X, y, callbacks=[fail_epoch_end])

    assert estimator.training_events_[-1]["name"] == "failure"
    assert estimator.training_events_[-1]["data"]["error_type"] == "RuntimeError"


def test_early_stop_event_is_stable():
    model = torch.nn.Linear(1, 1)
    loader = DataLoader(
        TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1)),
        batch_size=1,
    )
    events = []

    class ConstantLoss:
        def __call__(self, prediction, target):
            del target
            return prediction.sum() * 0.0 + 1.0

    run_training_loop(
        model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=ConstantLoss(),
        train_loader=loader,
        device=torch.device("cpu"),
        cfg=_loop_config(epochs=3, early_stopping=True, patience=1),
        callbacks=[events.append],
    )

    assert any(event.name == "early_stop" for event in events)


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_events_metrics_metadata_and_logger_adapter():
    X, y = _data()
    logger = logging.Logger("psann-test")
    handler = _RecordHandler()
    logger.addHandler(handler)
    observed = []
    estimator = PSANNRegressor(
        epochs=2,
        batch_size=8,
        hidden_layers=1,
        hidden_units=8,
        random_state=5,
    )

    estimator.fit(
        X,
        y,
        validation_data=(X, y),
        metrics={"mae": lambda prediction, target: (prediction - target).abs().mean()},
        callbacks=[observed.append],
        logger=logger,
        scheduler="step",
        scheduler_params={"step_size": 1, "gamma": 0.5},
    )

    event_names = [event.name for event in observed]
    assert event_names[0] == "train_start"
    assert "validation_end" in event_names
    assert "epoch_end" in event_names
    assert event_names[-1] == "train_end"
    assert "train_mae" in estimator.history_[0]
    assert "val_mae" in estimator.history_[0]
    assert estimator.training_metadata_["optimizer"] == "Adam"
    assert estimator.training_metadata_["parameter_count"] > 0
    assert estimator.training_metadata_["train_input_shape"] == (16, 2)
    assert handler.records
    assert all(hasattr(record, "psann_event") for record in handler.records)


def test_compile_fallback_warns_and_is_emitted_on_cpu():
    X, y = _data()
    estimator = PSANNRegressor(
        epochs=1,
        hidden_layers=1,
        hidden_units=8,
        compile=True,
        device="cpu",
    )

    with pytest.warns(RuntimeWarning, match="compile fallback"):
        estimator.fit(X, y, fallback_policy="warn")

    fallbacks = [event for event in estimator.training_events_ if event["name"] == "fallback"]
    assert any(event["data"]["component"] == "compile" for event in fallbacks)


def test_amp_fallback_can_be_made_strict_on_cpu():
    X, y = _data()
    estimator = PSANNRegressor(
        epochs=1,
        hidden_layers=1,
        hidden_units=8,
        amp=True,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="amp fallback"):
        estimator.fit(X, y, fallback_policy="error")
    assert estimator.training_events_[-1]["name"] == "failure"


def test_unavailable_accelerator_policy_can_error_or_warn(monkeypatch):
    X, y = _data()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    strict = PSANNRegressor(epochs=1, device="cuda")
    with pytest.raises(RuntimeError, match="unavailable"):
        strict.fit(X, y, fallback_policy="error")
    assert not hasattr(strict, "model_")

    fallback = PSANNRegressor(
        epochs=1,
        hidden_layers=1,
        hidden_units=8,
        device="cuda",
    )
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        fallback.fit(X, y, fallback_policy="warn")
    assert fallback.training_metadata_["device"] == "cpu"
    assert any(
        event["name"] == "fallback" and event["data"]["component"] == "device"
        for event in fallback.training_events_
    )
