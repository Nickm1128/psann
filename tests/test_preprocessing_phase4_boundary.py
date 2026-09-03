"""Durable Phase-4 behavior tests for the canonical preprocessing boundary."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psann.architectures import ArchitectureConfig, AttentionConfig, ConvolutionConfig
from psann.estimators import PSANNRegressor
from psann.preprocessing import (
    LSMConfig,
    LSMPretrainingConfig,
    ModulePreprocessorConfig,
    PreprocessorConfig,
    PreprocessorTrainingConfig,
)


def _dense_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.arange(24, dtype=np.float32).reshape(8, 3) / 10
    return X, X.sum(axis=1)


def _small_estimator(preprocessor: PreprocessorConfig, **kwargs: object) -> PSANNRegressor:
    return PSANNRegressor(
        preprocessor=preprocessor,
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=0,
        device="cpu",
        **kwargs,
    )


@pytest.mark.parametrize("trainable", [False, True])
def test_custom_flat_module_freeze_train_and_v2_round_trip(tmp_path, trainable: bool) -> None:
    X, y = _dense_data()
    module = torch.nn.Linear(3, 4)
    original = module.weight.detach().clone()
    config = PreprocessorConfig(
        ModulePreprocessorConfig(module, "flat", "flat", 4),
        PreprocessorTrainingConfig(trainable=trainable, lr=5e-3),
    )
    estimator = _small_estimator(config).fit(X, y)
    fitted = estimator.preprocessor_
    assert fitted is estimator.model_.preproc
    assert all(parameter.requires_grad is trainable for parameter in fitted.parameters())
    changed = not torch.equal(fitted.weight.detach(), original)
    assert changed is trainable
    first = tmp_path / "custom-first.pt"
    second = tmp_path / "custom-second.pt"
    estimator.save(str(first))
    restored = PSANNRegressor.load(str(first), map_location="cpu")
    restored.save(str(second))
    reloaded = PSANNRegressor.load(str(second), map_location="cpu")
    assert reloaded.preprocessor_capabilities_.serializable_kind == "module"
    np.testing.assert_allclose(reloaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-6)


def test_conv_lsm_two_v2_generations_keep_channels_and_predictions(tmp_path) -> None:
    X = np.arange(72, dtype=np.float32).reshape(8, 1, 3, 3) / 10
    y = X.mean(axis=(1, 2, 3))
    config = PreprocessorConfig(
        LSMConfig.convolutional(
            output_dim=2,
            hidden_units=3,
            pretraining=LSMPretrainingConfig(epochs=0),
            random_state=0,
        )
    )
    estimator = _small_estimator(
        config,
        architecture=ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3)),
    ).fit(X, y)
    first = tmp_path / "conv-first.pt"
    second = tmp_path / "conv-second.pt"
    estimator.save(str(first))
    restored = PSANNRegressor.load(str(first), map_location="cpu")
    restored.save(str(second))
    reloaded = PSANNRegressor.load(str(second), map_location="cpu")
    assert reloaded.preprocessor_capabilities_.output_dim == 2
    np.testing.assert_allclose(reloaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-6)


def test_attention_rejects_dense_lsm_before_pretraining(monkeypatch) -> None:
    from psann.estimators import regressor as regressor_module

    X = np.ones((8, 3), dtype=np.float32)
    config = PreprocessorConfig(
        LSMConfig.dense(
            output_dim=4,
            pretraining=LSMPretrainingConfig(epochs=1),
        )
    )
    called = False

    def unexpected_prepare(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("pretraining must not start for an invalid composition")

    monkeypatch.setattr(regressor_module, "prepare_preprocessor", unexpected_prepare)
    estimator = _small_estimator(
        config,
        architecture=ArchitectureConfig.dense(attention=AttentionConfig(num_heads=1)),
    )
    with pytest.raises(ValueError, match="tokens-to-tokens"):
        estimator.fit(X, X.sum(axis=1))
    assert not called
