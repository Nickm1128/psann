import numpy as np
import pytest

pytest.importorskip("torch")
import torch

from psann import PSANNRegressor
from psann.estimators import _fit_utils as fit_utils
from psann.hisso import hisso_evaluate_reward
from psann.lsm import LSMExpander

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings("ignore:LSMExpander:UserWarning"),
    pytest.mark.filterwarnings("ignore:.*hidden_width.*:DeprecationWarning"),
]


@pytest.mark.parametrize("cuda_available", [False, True])
def test_hisso_smoke_device_warmstart_reward(monkeypatch, cuda_available):
    rng = np.random.default_rng(21)
    X = rng.standard_normal((32, 4)).astype(np.float32)
    y = rng.standard_normal((32, 1)).astype(np.float32)

    warm_calls: list[bool] = []
    original_warmstart = fit_utils.run_hisso_supervised_warmstart

    def tracking_warmstart(*args, **kwargs):
        warm_calls.append(True)
        return original_warmstart(*args, **kwargs)

    monkeypatch.setattr(
        fit_utils,
        "run_hisso_supervised_warmstart",
        tracking_warmstart,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    module_device_calls: list[str] = []
    tensor_device_calls: list[str] = []

    if cuda_available:
        original_module_to = torch.nn.Module.to
        original_tensor_to = torch.Tensor.to

        def fake_module_to(self, *args, **kwargs):
            target = args[0] if args else kwargs.get("device", None)
            target_str = str(target) if target is not None else ""
            if isinstance(target, torch.device):
                target_str = (
                    target.type if target.index is None else f"{target.type}:{target.index}"
                )
            if target_str.startswith("cuda"):
                module_device_calls.append(target_str)
                return self
            return original_module_to(self, *args, **kwargs)

        def fake_tensor_to(self, *args, **kwargs):
            target = args[0] if args else kwargs.get("device", None)
            target_str = str(target) if target is not None else ""
            if isinstance(target, torch.device):
                target_str = (
                    target.type if target.index is None else f"{target.type}:{target.index}"
                )
            if target_str.startswith("cuda"):
                tensor_device_calls.append(target_str)
                return self
            return original_tensor_to(self, *args, **kwargs)

        monkeypatch.setattr(torch.nn.Module, "to", fake_module_to, raising=False)
        monkeypatch.setattr(torch.Tensor, "to", fake_tensor_to, raising=False)

    reg = PSANNRegressor(
        hidden_layers=1,
        hidden_units=10,
        epochs=1,
        batch_size=8,
        lr=5e-3,
        random_state=7,
    )

    reg.fit(
        X,
        y,
        hisso=True,
        hisso_window=12,
        hisso_supervised={"y": y, "epochs": 1, "batch_size": 8},
    )

    trainer = getattr(reg, "_hisso_trainer_", None)
    assert trainer is not None
    assert trainer.history
    assert warm_calls

    expected_device = "cuda" if cuda_available else "cpu"
    assert trainer.profile.get("device") == expected_device

    if cuda_available:
        assert module_device_calls
        assert tensor_device_calls

    reward_val = hisso_evaluate_reward(reg, X[:16])
    assert isinstance(reward_val, float)
    assert np.isfinite(reward_val)
    options = getattr(reg, "_hisso_options_", None)
    assert options is not None
    assert options.reward_fn is not None
    assert options.context_extractor is None
    assert isinstance(trainer.history[-1].get("reward"), (float, type(None)))


@pytest.mark.parametrize("lsm_train", [False, True])
def test_hisso_with_lsm(lsm_train):
    rng = np.random.default_rng(8)
    X = rng.standard_normal((40, 3)).astype(np.float32)
    weights = rng.standard_normal((3, 1)).astype(np.float32)
    y = (X @ weights + 0.02 * rng.standard_normal((40, 1))).astype(np.float32)

    expander = LSMExpander(
        output_dim=6,
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        lr=5e-3,
        batch_size=16,
        random_state=11,
    )
    if not lsm_train:
        expander.fit(X, epochs=1)

    reg = PSANNRegressor(
        hidden_layers=1,
        hidden_units=12,
        epochs=1,
        batch_size=8,
        lr=3e-3,
        lsm=expander,
        lsm_train=lsm_train,
        lsm_pretrain_epochs=1,
        lsm_lr=3e-3,
        random_state=19,
    )

    reg.fit(
        X,
        y,
        hisso=True,
        hisso_window=16,
        hisso_supervised={"y": y, "epochs": 1, "batch_size": 8},
    )

    trainer = getattr(reg, "_hisso_trainer_", None)
    assert trainer is not None
    assert trainer.history

    preds = reg.predict(X[:5])
    assert preds.shape == (5, 1)
    assert np.isfinite(preds).all()

    reward_val = hisso_evaluate_reward(reg, X[:20])
    assert np.isfinite(reward_val)

    preproc = getattr(reg.model_, "preproc", None)
    assert preproc is not None
    params_requires_grad = [p.requires_grad for p in preproc.parameters()]
    if lsm_train:
        assert any(params_requires_grad)
    else:
        assert not any(params_requires_grad)
