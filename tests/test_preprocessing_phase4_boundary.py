"""Durable Phase-4 behavior tests for the canonical preprocessing boundary."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from psann.architectures import (
    ArchitectureConfig,
    AttentionConfig,
    ContextConfig,
    ConvolutionConfig,
)
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
    payload = torch.load(first, weights_only=False)
    payload["fitted"]["preprocessing"]["output_topology"] = "tokens"
    torch.save(payload, broken)
    with pytest.raises(ValueError, match="fitted.preprocessing.output_topology"):
        PSANNRegressor.load(str(broken))
    payload = torch.load(first, weights_only=False)
    payload["fitted"]["preprocessing"]["output_dim"] = 4.0
    torch.save(payload, broken)
    with pytest.raises(TypeError, match="fitted.preprocessing.output_dim"):
        PSANNRegressor.load(str(broken))
    # map_location must choose CPU before construction, even if the serialized
    # constructor record came from a CUDA training host.
    payload = torch.load(first, weights_only=False)
    payload["estimator_params"]["device"] = "cuda"
    torch.save(payload, broken)
    assert PSANNRegressor.load(str(broken), map_location="cpu").device.type == "cpu"
    loaded = PSANNRegressor.load(str(first))
    assert isinstance(loaded.score_reconstruction(X), float)
    loaded.save(str(second))
    assert isinstance(PSANNRegressor.load(str(second)).score_reconstruction(X), float)


def test_custom_preprocessor_clone_and_parent_component_update_are_runtime_safe() -> None:
    from sklearn.base import clone

    X, y = _dense_data()
    config = PreprocessorConfig(ModulePreprocessorConfig(torch.nn.Linear(3, 4), "flat", "flat", 4))
    estimator = _small_estimator(config)
    cloned = clone(estimator)
    cloned.set_params(
        preprocessor__component=ModulePreprocessorConfig(torch.nn.Linear(3, 5), "flat", "flat", 5)
    ).fit(X, y)
    assert cloned.preprocessor_capabilities_.output_dim == 5
    assert cloned.predict(X[:2]).shape == (2,)


@pytest.mark.parametrize("attention", [False, True], ids=["flat", "token-attention"])
def test_wave_context_survives_canonical_preprocessor_wrapper(attention: bool) -> None:
    context = np.ones((8, 1), dtype=np.float32)
    if attention:
        X = np.arange(48, dtype=np.float32).reshape(8, 2, 3) / 10
        config = PreprocessorConfig(
            ModulePreprocessorConfig(torch.nn.Linear(3, 4), "tokens", "tokens", 4)
        )
        architecture = ArchitectureConfig.for_wave(
            attention=AttentionConfig(num_heads=1), context=ContextConfig(dim=1)
        )
        y = X.sum(axis=(1, 2))
    else:
        X, y = _dense_data()
        config = PreprocessorConfig(LSMConfig.dense(output_dim=4))
        architecture = ArchitectureConfig.for_wave(context=ContextConfig(dim=1))
    estimator = _small_estimator(config, architecture=architecture).fit(X, y, context=context)
    assert estimator.predict(X[:2], context=context[:2]).shape == (2,)


def test_cross_kind_component_replacements_fit_in_both_directions() -> None:
    X, y = _dense_data()
    estimator = _small_estimator(PreprocessorConfig(LSMConfig.dense(output_dim=4)))
    module = ModulePreprocessorConfig(torch.nn.Linear(3, 5), "flat", "flat", 5)
    estimator.set_params(preprocessor__component=module).fit(X, y)
    assert estimator.preprocessor_capabilities_.serializable_kind == "module"
    estimator.set_params(preprocessor__component=LSMConfig.dense(output_dim=4)).fit(X, y)
    assert estimator.preprocessor_capabilities_.serializable_kind == "lsm"


def test_custom_false_topology_and_non_tensor_output_reject_before_training() -> None:
    X = np.arange(48, dtype=np.float32).reshape(8, 2, 3) / 10
    y = X.sum(axis=(1, 2))
    false_flat = PreprocessorConfig(
        ModulePreprocessorConfig(torch.nn.Identity(), "tokens", "flat", 3)
    )
    with pytest.raises(ValueError, match="output_topology"):
        _small_estimator(false_flat).fit(X, y)

    class NotATensor(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> object:
            return value.tolist()

    not_tensor = PreprocessorConfig(ModulePreprocessorConfig(NotATensor(), "flat", "flat", 3))
    with pytest.raises(ValueError, match="torch.Tensor"):
        _small_estimator(not_tensor).fit(*_dense_data())


def test_reconstruction_score_uses_fitted_scaling_and_conv_layout() -> None:
    X, y = _dense_data()
    estimator = _small_estimator(
        PreprocessorConfig(LSMConfig.dense(output_dim=4)), scaler="standard"
    ).fit(X, y)
    scaled, _, _ = estimator._prepare_inference_inputs(X)
    assert estimator.score_reconstruction(X) == pytest.approx(
        estimator.preprocessor_controller_.score_reconstruction(scaled)
    )
    X_nhwc = np.arange(72, dtype=np.float32).reshape(8, 3, 3, 1) / 10
    y_nhwc = X_nhwc.mean(axis=(1, 2, 3))
    conv = _small_estimator(
        PreprocessorConfig(LSMConfig.convolutional(output_dim=2)),
        architecture=ArchitectureConfig.convolutional(
            convolution=ConvolutionConfig(data_format="channels_last")
        ),
    ).fit(X_nhwc, y_nhwc)
    assert isinstance(conv.score_reconstruction(X_nhwc), float)


def test_preprocessor_nested_gridsearch_and_joblib_round_trip(tmp_path) -> None:
    import joblib
    from sklearn.model_selection import GridSearchCV

    X, y = _dense_data()
    estimator = _small_estimator(PreprocessorConfig(LSMConfig.dense(output_dim=3)))
    search = GridSearchCV(
        estimator,
        {"preprocessor__component__output_dim": [3, 4]},
        cv=2,
        error_score="raise",
    ).fit(X, y)
    restored_path = tmp_path / "canonical.joblib"
    joblib.dump(search.best_estimator_, restored_path)
    restored = joblib.load(restored_path)
    assert restored.predict(X[:2]).shape == (2,)


def test_shape_changing_preprocessor_validation_enters_wrapper_once() -> None:
    X, y = _dense_data()
    estimator = _small_estimator(
        PreprocessorConfig(LSMConfig.dense(output_dim=4)), early_stopping=True
    ).fit(X, y, validation_data=(X[:2], y[:2]))
    assert estimator.predict(X[:2]).shape == (2,)


def test_frozen_preprocessor_is_excluded_from_supervised_optimizer() -> None:
    X, y = _dense_data()
    estimator = _small_estimator(
        PreprocessorConfig(LSMConfig.dense(output_dim=4), PreprocessorTrainingConfig(False))
    ).fit(X, y)
    optimizer_parameters = {
        id(parameter)
        for group in estimator._optimizer_.param_groups
        for parameter in group["params"]
    }
    assert not any(
        id(parameter) in optimizer_parameters for parameter in estimator.preprocessor_.parameters()
    )


def test_custom_checkpoint_rejects_unknown_preprocessor_metadata(tmp_path) -> None:
    X, y = _dense_data()
    estimator = _small_estimator(
        PreprocessorConfig(ModulePreprocessorConfig(torch.nn.Linear(3, 4), "flat", "flat", 4))
    ).fit(X, y)
    path = tmp_path / "custom.pt"
    estimator.save(str(path))
    payload = torch.load(path, weights_only=False)
    payload["estimator_params"]["preprocessor"]["surprise"] = 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="preprocessor.surprise"):
        PSANNRegressor.load(str(path))


@pytest.mark.parametrize("topology", ["dense", "conv2d"])
@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("separate_rate", [False, True])
def test_canonical_preprocessor_hisso_uses_separate_core_and_preprocessor_groups(
    topology: str, trainable: bool, separate_rate: bool, monkeypatch
) -> None:
    """Pure HISSO and its supervised warm start share canonical group policy."""

    if topology == "dense":
        X, y = _dense_data()
        component = LSMConfig.dense(output_dim=4)
        architecture = ArchitectureConfig.dense()
    else:
        X = np.arange(72, dtype=np.float32).reshape(8, 1, 3, 3) / 10
        y = X.mean(axis=(1, 2, 3))
        component = LSMConfig.convolutional(output_dim=2, hidden_units=3)
        architecture = ArchitectureConfig.convolutional(convolution=ConvolutionConfig(channels=3))
    preprocessor_lr = 0.123 if separate_rate else None
    estimator = PSANNRegressor(
        architecture=architecture,
        preprocessor=PreprocessorConfig(
            component, PreprocessorTrainingConfig(trainable=trainable, lr=preprocessor_lr)
        ),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        lr=0.001,
        device="cpu",
        random_state=0,
    )
    from psann.estimators import _fit_utils as fit_utils

    snapshots: list[list[torch.Tensor]] = []
    warm_snapshots: list[list[torch.Tensor]] = []
    original_hisso = fit_utils.run_hisso_training
    original_warmstart = fit_utils.run_hisso_supervised_warmstart

    def capture_hisso(*args, **kwargs):
        model = args[0].model_
        snapshots.append([parameter.detach().clone() for parameter in model.preproc.parameters()])
        return original_hisso(*args, **kwargs)

    def capture_warmstart(*args, **kwargs):
        model = args[0].model_
        warm_snapshots.append(
            [parameter.detach().clone() for parameter in model.preproc.parameters()]
        )
        return original_warmstart(*args, **kwargs)

    monkeypatch.setattr(fit_utils, "run_hisso_training", capture_hisso)
    monkeypatch.setattr(fit_utils, "run_hisso_supervised_warmstart", capture_warmstart)
    reward = lambda actions, _context: actions.sum(dim=-1)
    estimator.fit(X, y, hisso=True, hisso_window=4, hisso_reward_fn=reward)
    trainer = estimator._hisso_trainer_
    assert trainer is not None
    groups = {group["psann_parameter_group"]: group for group in trainer.optimizer.param_groups}
    assert groups["core"]["lr"] == pytest.approx(0.001)
    preprocessor_ids = {id(parameter) for parameter in estimator.preprocessor_.parameters()}
    optimizer_ids = {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    }
    if trainable:
        assert groups["preprocessor"]["lr"] == pytest.approx(0.123 if separate_rate else 0.001)
        assert preprocessor_ids <= optimizer_ids
    else:
        assert "preprocessor" not in groups
        assert preprocessor_ids.isdisjoint(optimizer_ids)

    # The supervised warm-start may override its core rate, but never collapses
    # canonical preprocessing into that core group.
    warm = PSANNRegressor(
        architecture=architecture,
        preprocessor=PreprocessorConfig(
            component, PreprocessorTrainingConfig(trainable=trainable, lr=preprocessor_lr)
        ),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        lr=0.001,
        device="cpu",
        random_state=0,
    )
    warm.fit(
        X,
        y,
        hisso=True,
        hisso_window=4,
        hisso_reward_fn=reward,
        hisso_supervised={"y": y, "epochs": 1, "batch_size": 4, "lr": 0.02},
    )
    warm_groups = {
        group["psann_parameter_group"]: group
        for group in warm._hisso_warmstart_optimizer_.param_groups
    }
    assert warm_groups["core"]["lr"] == pytest.approx(0.02)
    if trainable:
        assert warm_groups["preprocessor"]["lr"] == pytest.approx(0.123 if separate_rate else 0.001)
        assert any(
            not torch.equal(before, after)
            for before, after in zip(warm_snapshots[0], warm.preprocessor_.parameters())
        )
    else:
        assert "preprocessor" not in warm_groups


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda preprocessor: preprocessor["training"].update({"surprise": 1}),
            "training.surprise",
        ),
        (
            lambda preprocessor: preprocessor.__setitem__("training", []),
            "training must be a mapping",
        ),
        (
            lambda preprocessor: preprocessor["training"].__setitem__("trainable", "yes"),
            "training.trainable",
        ),
        (lambda preprocessor: preprocessor["training"].pop("lr"), "training.lr is missing"),
    ],
)
def test_custom_checkpoint_nested_metadata_errors_are_path_specific(
    tmp_path, mutation, error: str
) -> None:
    X, y = _dense_data()
    estimator = _small_estimator(
        PreprocessorConfig(ModulePreprocessorConfig(torch.nn.Linear(3, 4), "flat", "flat", 4))
    ).fit(X, y)
    path = tmp_path / "custom-nested.pt"
    estimator.save(str(path))
    payload = torch.load(path, weights_only=False)
    mutation(payload["estimator_params"]["preprocessor"])
    torch.save(payload, path)
    with pytest.raises((TypeError, ValueError), match=error):
        PSANNRegressor.load(str(path))
