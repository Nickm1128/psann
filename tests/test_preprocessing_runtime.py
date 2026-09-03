"""Runtime behavior for canonical preprocessing composition."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from psann import PSANNRegressor
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
