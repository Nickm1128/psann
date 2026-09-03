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


@pytest.mark.parametrize(
    "architecture",
    [
        ArchitectureConfig.dense(attention=AttentionConfig(num_heads=1)),
        ArchitectureConfig.for_wave(attention=AttentionConfig(num_heads=1)),
        ArchitectureConfig.for_sequence(),
    ],
    ids=["dense-attention", "wave-attention", "sequence"],
)
def test_token_preprocessors_reach_supported_architecture_fit_predict(
    architecture: ArchitectureConfig,
) -> None:
    X = np.arange(48, dtype=np.float32).reshape(8, 2, 3) / 10
    y = X.sum(axis=(1, 2))
    config = PreprocessorConfig(
        ModulePreprocessorConfig(torch.nn.Linear(3, 4), "tokens", "tokens", 4)
    )
    estimator = _small_estimator(config, architecture=architecture).fit(X, y)
    assert estimator.preprocessor_ is estimator.model_.preproc
    assert estimator.predict(X[:2]).shape == (2,)


def test_conv1d_spatial_preprocessor_is_not_misclassified_as_tokens() -> None:
    X = np.arange(32, dtype=np.float32).reshape(8, 1, 4) / 10
    y = X.mean(axis=(1, 2))
    config = PreprocessorConfig(
        ModulePreprocessorConfig(torch.nn.Conv1d(1, 2, 1), "spatial-1d", "spatial-1d", 2)
    )
    estimator = _small_estimator(
        config,
        architecture=ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3)),
    ).fit(X, y)
    assert estimator.preprocessor_capabilities_.input_topology == "spatial-1d"
    assert estimator.predict(X[:2]).shape == (2,)


def test_custom_input_topology_and_declared_width_are_validated_before_fit() -> None:
    X, y = _dense_data()
    wrong_topology = PreprocessorConfig(
        ModulePreprocessorConfig(torch.nn.Linear(3, 4), "tokens", "flat", 4)
    )
    with pytest.raises(ValueError, match="input_topology"):
        _small_estimator(wrong_topology).fit(X, y)
    wrong_width = PreprocessorConfig(
        ModulePreprocessorConfig(torch.nn.Linear(3, 4), "flat", "flat", 5)
    )
    with pytest.raises(ValueError, match="output_dim"):
        _small_estimator(wrong_width).fit(X, y)


def test_nested_training_update_and_warm_start_keep_the_attached_preprocessor() -> None:
    X, y = _dense_data()
    config = PreprocessorConfig(
        LSMConfig.dense(output_dim=4), PreprocessorTrainingConfig(trainable=False)
    )
    estimator = _small_estimator(config, warm_start=True).fit(X, y)
    first = estimator.preprocessor_
    estimator.fit(X, y)
    assert estimator.preprocessor_ is first is estimator.model_.preproc
    estimator.set_params(
        preprocessor__training=PreprocessorTrainingConfig(trainable=True, lr=0.007)
    ).fit(X, y)
    assert all(parameter.requires_grad for parameter in estimator.preprocessor_.parameters())
    assert estimator.preprocessor_ is estimator.model_.preproc
    assert estimator._optimizer_.param_groups[1]["lr"] == pytest.approx(0.007)


def test_v2_metadata_is_strict_and_lsm_controller_survives_two_generations(tmp_path) -> None:
    X, y = _dense_data()
    estimator = _small_estimator(PreprocessorConfig(LSMConfig.dense(output_dim=4))).fit(X, y)
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    estimator.save(str(first))
    payload = torch.load(first, weights_only=False)
    payload["fitted"]["preprocessing"].pop("input_topology")
    broken = tmp_path / "broken.pt"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="fitted.preprocessing.input_topology"):
        PSANNRegressor.load(str(broken))
    loaded = PSANNRegressor.load(str(first))
    assert isinstance(loaded.score_reconstruction(X), float)
    loaded.save(str(second))
    assert isinstance(PSANNRegressor.load(str(second)).score_reconstruction(X), float)
