"""Runtime behavior for canonical preprocessing composition."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("torch")

from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig, ConvolutionConfig
from psann.preprocessing import (
    LSMConfig,
    LSMPretrainingConfig,
    PreprocessorConfig,
    PreprocessorTrainingConfig,
)


@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_dense_lsm_pretraining_and_training_policy_reach_fit(
    pretrained: bool, trainable: bool
) -> None:
    X = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    y = X.sum(axis=1)
    config = PreprocessorConfig(
        component=LSMConfig.dense(
            output_dim=4,
            hidden_layers=1,
            hidden_units=5,
            random_state=0,
            pretraining=LSMPretrainingConfig(epochs=1 if pretrained else 0, batch_size=4),
        ),
        training=PreprocessorTrainingConfig(trainable=trainable, lr=0.002),
    )
    estimator = PSANNRegressor(
        preprocessor=config,
        hidden_layers=1,
        hidden_units=6,
        epochs=1,
        batch_size=4,
        random_state=0,
    ).fit(X, y)
    assert estimator.model_.preproc is estimator.preprocessor_
    assert estimator.preprocessor_capabilities_.output_dim == 4
    assert estimator.predict(X[:2]).shape == (2,)
    assert all(
        parameter.requires_grad is trainable for parameter in estimator.preprocessor_.parameters()
    )
    if trainable:
        assert len(estimator._optimizer_.param_groups) == 2
        assert estimator._optimizer_.param_groups[1]["lr"] == pytest.approx(0.002)
    else:
        assert len(estimator._optimizer_.param_groups) == 1


def test_canonical_preprocessor_shallow_params_exclude_legacy_lsm_names() -> None:
    config = PreprocessorConfig(component=LSMConfig.dense(output_dim=4))
    estimator = PSANNRegressor(preprocessor=config)
    params = estimator.get_params(deep=False)
    assert params["preprocessor"] is config
    assert not any(name.startswith("lsm") for name in params)
    estimator.set_params(
        preprocessor__component__output_dim=6,
        preprocessor__component__pretraining__epochs=1,
        preprocessor__training__trainable=True,
    )
    assert estimator.preprocessor.component.output_dim == 6
    assert estimator.preprocessor.component.pretraining.epochs == 1
    assert estimator.preprocessor.training.trainable is True


def test_v2_dense_lsm_checkpoint_survives_two_generations(tmp_path) -> None:
    X = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    y = X.sum(axis=1)
    estimator = PSANNRegressor(
        preprocessor=PreprocessorConfig(
            component=LSMConfig.dense(
                output_dim=4,
                hidden_layers=1,
                hidden_units=5,
                pretraining=LSMPretrainingConfig(epochs=0, batch_size=4),
            )
        ),
        hidden_layers=1,
        hidden_units=6,
        epochs=1,
        batch_size=4,
        random_state=0,
    ).fit(X, y)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    estimator.save(str(first))
    loaded = PSANNRegressor.load(str(first))
    loaded.save(str(second))
    reloaded = PSANNRegressor.load(str(second))
    assert reloaded.preprocessor_capabilities_.output_dim == 4
    np.testing.assert_allclose(reloaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-5)


@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("trainable", [False, True])
def test_conv_lsm_pretraining_and_joint_policy_keep_weight_and_optimizer_invariants(
    pretrained: bool, trainable: bool
) -> None:
    """Conv2d LSM pretraining/freeze policy is observable beyond output shape."""

    X = np.arange(72, dtype=np.float32).reshape(8, 1, 3, 3) / 10
    y = X.mean(axis=(1, 2, 3))
    config = PreprocessorConfig(
        LSMConfig.convolutional(
            output_dim=2,
            hidden_units=3,
            random_state=0,
            pretraining=LSMPretrainingConfig(epochs=1 if pretrained else 0),
        ),
        PreprocessorTrainingConfig(trainable=trainable, lr=0.003),
    )
    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3)),
        preprocessor=config,
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=0,
        device="cpu",
        warm_start=True,
    ).fit(X, y)
    before_second_fit = [
        parameter.detach().clone() for parameter in estimator.preprocessor_.parameters()
    ]
    estimator.fit(X, y)
    after_second_fit = list(estimator.preprocessor_.parameters())
    optimizer_parameters = {
        id(parameter)
        for group in estimator._optimizer_.param_groups
        for parameter in group["params"]
    }
    preprocessor_parameters = list(estimator.preprocessor_.parameters())
    assert estimator.predict(X[:2]).shape == (2,)
    assert all(parameter.requires_grad is trainable for parameter in preprocessor_parameters)
    if trainable:
        groups = {
            group["psann_parameter_group"]: group for group in estimator._optimizer_.param_groups
        }
        assert groups["preprocessor"]["lr"] == pytest.approx(0.003)
        assert {id(parameter) for parameter in preprocessor_parameters} <= optimizer_parameters
        assert any(
            not torch.equal(before, after.detach())
            for before, after in zip(before_second_fit, after_second_fit)
        )
    else:
        assert {id(parameter) for parameter in preprocessor_parameters}.isdisjoint(
            optimizer_parameters
        )
        assert all(
            torch.equal(before, after.detach())
            for before, after in zip(before_second_fit, after_second_fit)
        )
    assert "ols_readout" in estimator.preprocessor_diagnostics_
