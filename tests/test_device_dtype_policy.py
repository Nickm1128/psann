import numpy as np
import torch
import torch.nn as nn
from psann import PSANNRegressor
from psann.estimators._fit_utils import (
    _prepare_noise_tensor,
    _prepare_validation_tensors,
    normalise_fit_args,
    prepare_inputs_and_scaler,
)


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_calls = 0

    def to(self, *args, **kwargs):
        self.to_calls += 1
        return super().to(*args, **kwargs)


def test_ensure_model_device_caches_move():
    est = PSANNRegressor(device="cpu")
    dummy = DummyModule()
    est.model_ = dummy
    est._model_device_ = None

    device = torch.device("cpu")
    est._ensure_model_device(device)
    assert est._model_device_ == device
    assert dummy.to_calls == 1

    est._ensure_model_device(device)
    assert dummy.to_calls == 1


def test_prepare_validation_tensors_cpu_float32():
    rs = np.random.RandomState(0)
    X = rs.randn(8, 3).astype(np.float64)
    y = rs.randn(8, 1).astype(np.float64)
    X_val = rs.randn(2, 3).astype(np.float64)
    y_val = rs.randn(2, 1).astype(np.float64)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=4,
        preserve_shape=False,
        device="cpu",
    )

    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=(X_val, y_val),
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)

    val_inputs_t, val_targets_t, ctx_t = _prepare_validation_tensors(
        est,
        prepared,
        fit_args.validation,
        device=torch.device("cpu"),
    )

    assert val_inputs_t is not None and val_targets_t is not None
    assert val_inputs_t.dtype == torch.float32
    assert val_targets_t.dtype == torch.float32
    assert val_inputs_t.device.type == "cpu"
    assert val_targets_t.device.type == "cpu"
    assert ctx_t is None

    expected_inputs = est._apply_fitted_scaler(est._flatten(X_val)).astype(np.float32, copy=False)
    expected_targets = y_val.reshape(y_val.shape[0], -1).astype(np.float32, copy=False)
    np.testing.assert_allclose(val_inputs_t.numpy(), expected_inputs, rtol=0, atol=1e-6)
    np.testing.assert_allclose(val_targets_t.numpy(), expected_targets, rtol=0, atol=1e-6)


def test_prepare_noise_tensor_cpu_dtype():
    rs = np.random.RandomState(1)
    X = rs.randn(6, 4).astype(np.float32)
    y = rs.randn(6, 1).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=2,
        preserve_shape=False,
        device="cpu",
    )

    fit_args = normalise_fit_args(
        est,
        X,
        y,
        context=None,
        validation_data=None,
        noisy=None,
        verbose=0,
        lr_max=None,
        lr_min=None,
        hisso=False,
        hisso_kwargs={},
    )
    prepared, _ = prepare_inputs_and_scaler(est, fit_args)

    n_features = int(np.prod(est.input_shape_))
    noise_vector = np.linspace(0.1, 0.2, n_features, dtype=np.float32)

    noise_t = _prepare_noise_tensor(est, prepared, noise_vector, device=torch.device("cpu"))

    assert noise_t is not None
    assert noise_t.device.type == "cpu"
    assert noise_t.dtype == torch.float32
    np.testing.assert_allclose(noise_t.numpy(), noise_vector.reshape(1, -1), rtol=0, atol=1e-6)
