from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import psann
from psann.platform import (
    ACTIVATIONS,
    BACKBONES,
    LOSSES,
    METRICS,
    NORMALIZATIONS,
    OPTIMIZERS,
    SCHEDULERS,
    DataSchema,
    InferenceConfig,
    ModelSpec,
    SupervisedData,
    TaskSpec,
    TorchModuleAdapter,
    TrainingConfig,
    adapt_module,
    create_model,
    register_backbone,
    register_schema_transform,
    train,
)
from psann.platform.tasks import create_task_adapter


def _config() -> TrainingConfig:
    return TrainingConfig(
        epochs=1,
        batch_size=4,
        learning_rate=5e-3,
        deterministic=True,
    )


def _parameters(*, conv: bool = False) -> dict[str, object]:
    values: dict[str, object] = {
        "hidden_layers": 1,
        "hidden_units": 4,
        "random_state": 7,
    }
    if conv:
        values["conv_channels"] = 4
    return values


def test_serializable_specs_round_trip(tmp_path: Path):
    model = ModelSpec(
        task=TaskSpec(
            kind="multilabel",
            class_names=("fraud", "review"),
            threshold=(0.7, 0.4),
        ),
        backbone="respsann_mlp",
        input_schema=DataSchema(
            feature_names=("amount", "velocity"),
            output_names=("fraud", "review"),
            input_shape=(2,),
            feature_policy="reorder",
            preprocessing={"scaler": "standard"},
        ),
        activation="gelu",
        normalization="layer",
        parameters={"hidden_layers": 2, "hidden_units": 16},
    )
    training = TrainingConfig(
        epochs=3,
        scheduler="step",
        scheduler_params={"step_size": 1, "gamma": 0.5},
        metrics=("subset_accuracy",),
    )
    inference = InferenceConfig(batch_size=32, feature_policy="reorder")

    assert ModelSpec.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    assert TrainingConfig.from_dict(json.loads(json.dumps(training.to_dict()))) == training
    assert InferenceConfig.from_dict(json.loads(json.dumps(inference.to_dict()))) == inference

    path = tmp_path / "model.json"
    psann.platform.save_spec(model, path)
    assert psann.platform.load_model_spec(path) == model


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_specs_reject_nonfinite_json_values(invalid: float):
    with pytest.raises(TypeError, match="finite JSON-serializable"):
        ModelSpec(parameters={"invalid": invalid})
    with pytest.raises(TypeError, match="finite JSON-serializable"):
        DataSchema(preprocessing={"invalid": invalid})
    with pytest.raises(TypeError, match="finite JSON-serializable"):
        TrainingConfig(scheduler_params={"invalid": invalid})
    with pytest.raises(TypeError, match="positive integer"):
        InferenceConfig(batch_size=invalid).to_dict()  # type: ignore[arg-type]


def test_binary_training_metric_uses_configured_threshold():
    adapter = create_task_adapter(TaskSpec(kind="binary", class_names=("no", "yes"), threshold=0.9))
    targets = np.asarray(["no", "yes"])
    encoded = torch.from_numpy(adapter.fit_targets(targets))
    logits = torch.tensor([[np.log(4.0)], [np.log(19.0)]], dtype=torch.float32)

    observed = adapter.training_metrics()["accuracy"](logits, encoded)

    assert observed.item() == pytest.approx(1.0)
    assert adapter.evaluate(targets, logits.numpy())["accuracy"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("threshold", "probabilities", "targets"),
    [
        (0.9, (0.8, 0.8), (0, 0)),
        ((0.9, 0.1), (0.8, 0.2), (0, 1)),
    ],
)
def test_multilabel_training_metric_uses_scalar_or_per_label_thresholds(
    threshold: float | tuple[float, ...],
    probabilities: tuple[float, ...],
    targets: tuple[int, ...],
):
    adapter = create_task_adapter(
        TaskSpec(
            kind="multilabel",
            class_names=("left", "right"),
            threshold=threshold,
        )
    )
    truth = np.asarray([targets, targets], dtype=np.float32)
    encoded = torch.from_numpy(adapter.fit_targets(truth))
    probability_array = np.asarray([probabilities, probabilities], dtype=np.float32)
    logits = torch.from_numpy(np.log(probability_array / (1.0 - probability_array)))

    observed = adapter.training_metrics()["subset_accuracy"](logits, encoded)

    assert observed.item() == pytest.approx(1.0)
    assert adapter.evaluate(truth, logits.numpy())["subset_accuracy"] == pytest.approx(1.0)


def test_multilabel_validation_history_uses_heterogeneous_thresholds():
    inputs = np.random.default_rng(31).normal(size=(12, 3)).astype(np.float32)
    targets = np.asarray([[0, 1], [1, 0], [1, 1]] * 4, dtype=np.float32)
    thresholds = np.asarray([0.9, 0.1], dtype=np.float32)
    model = create_model(
        ModelSpec(
            task=TaskSpec(
                kind="multilabel",
                class_names=("left", "right"),
                threshold=tuple(float(item) for item in thresholds),
            ),
            input_schema=DataSchema(input_shape=(3,)),
            parameters=_parameters(),
        )
    )
    run = train(
        model,
        (inputs, targets),
        validation_data=(inputs, targets),
        config=_config(),
    )
    expected = model.predict_proba(inputs) >= thresholds.reshape(1, -1)
    expected_subset_accuracy = float((expected == targets.astype(bool)).all(axis=1).mean())

    assert run.history[-1]["val_subset_accuracy"] == pytest.approx(expected_subset_accuracy)
    assert model.score(inputs, targets) == pytest.approx(expected_subset_accuracy)


@pytest.mark.parametrize(
    ("backbone", "shape"),
    [
        ("psann_mlp", (3,)),
        ("respsann_mlp", (3,)),
        ("psann_conv1d", (1, 4)),
        ("psann_conv2d", (1, 3, 3)),
        ("psann_conv3d", (1, 2, 2, 2)),
        ("respsann_conv2d", (1, 3, 3)),
        ("wave_resnet", (3,)),
        ("sgr_psann", (3,)),
    ],
)
def test_registered_backbone_regression_matrix(backbone: str, shape: tuple[int, ...]):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(8, *shape)).astype(np.float32)
    y = rng.normal(size=8).astype(np.float32)
    model = create_model(
        ModelSpec(
            backbone=backbone,
            input_schema=DataSchema(input_shape=shape),
            parameters=_parameters(conv="conv" in backbone),
        )
    )

    run = train(model, (X, y), config=_config())

    assert np.asarray(model.predict(X[:2])).shape == (2,)
    assert len(run.history) == 1
    assert {"mae", "mse", "r2"} <= set(run.metrics)


@pytest.mark.parametrize(
    ("backbone", "shape"),
    [
        ("psann_mlp", (3,)),
        ("respsann_mlp", (3,)),
        ("psann_conv1d", (1, 4)),
        ("psann_conv2d", (1, 3, 3)),
        ("psann_conv3d", (1, 2, 2, 2)),
        ("respsann_conv2d", (1, 3, 3)),
        ("wave_resnet", (3,)),
        ("sgr_psann", (3,)),
    ],
)
def test_registered_backbone_binary_matrix(backbone: str, shape: tuple[int, ...]):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(8, *shape)).astype(np.float32)
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    model = create_model(
        ModelSpec(
            task=TaskSpec(kind="binary"),
            backbone=backbone,
            input_schema=DataSchema(input_shape=shape),
            parameters=_parameters(conv="conv" in backbone),
        )
    )

    train(model, SupervisedData(X, y), config=_config())

    assert isinstance(model, psann.PSANNClassifier)
    assert model.predict_proba(X[:3]).shape == (3, 2)
    assert set(model.predict(X[:3])).issubset({0, 1})


@pytest.mark.parametrize(
    ("backbone", "shape"),
    [
        ("psann_mlp", (3,)),
        ("respsann_mlp", (3,)),
        ("psann_conv1d", (1, 4)),
        ("psann_conv2d", (1, 3, 3)),
        ("psann_conv3d", (1, 2, 2, 2)),
        ("respsann_conv2d", (1, 3, 3)),
        ("wave_resnet", (3,)),
        ("sgr_psann", (3,)),
    ],
)
@pytest.mark.parametrize("task", ["multiclass", "multilabel"])
def test_registered_backbone_multiclass_and_multilabel_matrix(
    backbone: str,
    shape: tuple[int, ...],
    task: str,
):
    rng = np.random.default_rng(13)
    X = rng.normal(size=(8, *shape)).astype(np.float32)
    targets = (
        np.asarray([0, 1, 2, 0, 1, 2, 0, 1])
        if task == "multiclass"
        else np.asarray([[0, 1], [1, 0]] * 4)
    )
    model = create_model(
        ModelSpec(
            task=TaskSpec(kind=task),  # type: ignore[arg-type]
            backbone=backbone,
            input_schema=DataSchema(input_shape=shape),
            parameters=_parameters(conv="conv" in backbone),
        )
    )

    train(model, (X, targets), config=_config())

    width = 3 if task == "multiclass" else 2
    assert model.predict_proba(X[:2]).shape == (2, width)


@pytest.mark.parametrize(
    ("task", "targets", "probability_shape", "prediction_shape"),
    [
        (
            TaskSpec(kind="binary", positive_label="yes"),
            np.asarray(["no", "yes"] * 6),
            (3, 2),
            (3,),
        ),
        (
            TaskSpec(kind="multiclass"),
            np.asarray([0, 1, 2] * 4),
            (3, 3),
            (3,),
        ),
        (
            TaskSpec(
                kind="multilabel",
                class_names=("left", "right"),
                threshold=(0.4, 0.6),
            ),
            np.asarray([[0, 1], [1, 0], [1, 1]] * 4),
            (3, 2),
            (3, 2),
        ),
    ],
)
def test_task_owned_classifier_contracts(
    task: TaskSpec,
    targets: np.ndarray,
    probability_shape: tuple[int, ...],
    prediction_shape: tuple[int, ...],
):
    X = np.random.default_rng(3).normal(size=(12, 4)).astype(np.float32)
    model = create_model(
        ModelSpec(
            task=task,
            input_schema=DataSchema(input_shape=(4,)),
            activation="silu",
            parameters=_parameters(),
        )
    )
    run = train(model, (X, targets), config=_config())

    assert model.predict_proba(X[:3]).shape == probability_shape
    assert model.predict(X[:3]).shape == prediction_shape
    assert run.metrics


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid", "gelu", "silu", "psann"])
def test_standard_dense_activations_are_factory_supported(activation: str):
    X = np.random.default_rng(4).normal(size=(8, 3)).astype(np.float32)
    y = X[:, 0]
    model = create_model(
        ModelSpec(
            activation=activation,
            input_schema=DataSchema(input_shape=(3,)),
            parameters=_parameters(),
        )
    )
    train(model, (X, y), config=_config())
    assert model.predict(X[:2]).shape == (2,)


def test_normalization_dropout_and_shape_capabilities_fail_before_fit():
    with pytest.raises(ValueError, match="does not support standard dropout"):
        create_model(ModelSpec(backbone="psann_mlp", dropout=0.2))
    with pytest.raises(ValueError, match="does not support normalization"):
        create_model(ModelSpec(backbone="psann_mlp", normalization="layer"))
    with pytest.raises(ValueError, match="input_shape has rank"):
        create_model(
            ModelSpec(
                backbone="psann_conv2d",
                input_schema=DataSchema(input_shape=(4,)),
            )
        )
    wave = create_model(
        ModelSpec(
            backbone="wave_resnet",
            dropout=0.2,
            normalization="rms",
            input_schema=DataSchema(input_shape=(3,)),
            parameters=_parameters(),
        )
    )
    assert wave.dropout == pytest.approx(0.2)
    assert wave.norm == "rms"


def test_registries_publish_stable_configuration_identifiers():
    assert {"psann_mlp", "respsann_mlp", "psann_conv3d"} <= set(BACKBONES.names())
    assert {"relu", "tanh", "sigmoid", "gelu", "silu", "psann"} <= set(ACTIVATIONS.names())
    assert {"none", "layer", "rms", "weight"} <= set(NORMALIZATIONS.names())
    assert {"adam", "adamw", "sgd"} <= set(OPTIMIZERS.names())
    assert {"none", "step", "cosine"} <= set(SCHEDULERS.names())
    assert {"mse", "binary_cross_entropy_with_logits", "cross_entropy"} <= set(LOSSES.names())
    assert {"mae", "accuracy", "subset_accuracy"} <= set(METRICS.names())


def test_model_spec_rejects_aliases_and_unserializable_values():
    with pytest.raises(ValueError, match="canonical fields"):
        create_model(ModelSpec(parameters={"hidden_width": 4}))
    with pytest.raises(TypeError, match="JSON-serializable"):
        ModelSpec(parameters={"factory": lambda: None})


def test_high_level_training_validates_registered_configuration():
    X = np.ones((4, 2), dtype=np.float32)
    y = np.ones(4, dtype=np.float32)
    model = create_model(
        ModelSpec(
            input_schema=DataSchema(input_shape=(2,)),
            parameters=_parameters(),
        )
    )
    with pytest.raises(ValueError, match="Unknown optimizer"):
        train(model, (X, y), config=TrainingConfig(epochs=1, optimizer="mystery"))


def test_classifier_validation_data_rejects_unknown_classes():
    X = np.random.default_rng(5).normal(size=(8, 2)).astype(np.float32)
    y = np.asarray([0, 1] * 4)
    model = psann.PSANNClassifier(
        task="binary",
        estimator_params={**_parameters(), "epochs": 1, "batch_size": 4},
    )
    with pytest.raises(ValueError, match="unknown classes"):
        model.fit(X, y, validation_data=(X[:2], np.asarray([0, 2])))


def test_classifier_fitted_state_clone_pipeline_and_grid_search():
    sklearn_base = pytest.importorskip("sklearn.base")
    pipeline_module = pytest.importorskip("sklearn.pipeline")
    preprocessing = pytest.importorskip("sklearn.preprocessing")
    selection = pytest.importorskip("sklearn.model_selection")
    X = np.random.default_rng(6).normal(size=(20, 3)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    classifier = psann.PSANNClassifier(
        estimator_params={
            **_parameters(),
            "epochs": 1,
            "batch_size": 10,
        }
    )
    with pytest.raises(Exception, match="not fitted"):
        classifier.predict(X)
    cloned = sklearn_base.clone(classifier)
    assert cloned.get_params()["backbone"] == "psann_mlp"
    pipeline = pipeline_module.Pipeline(
        [
            ("scale", preprocessing.StandardScaler()),
            ("model", classifier),
        ]
    ).fit(X, y)
    assert pipeline.predict(X[:2]).shape == (2,)
    search = selection.GridSearchCV(
        classifier,
        {"threshold": [0.4, 0.5]},
        cv=2,
    ).fit(X, y)
    assert search.best_params_["threshold"] in {0.4, 0.5}


def test_pandas_feature_schema_strict_reorder_and_failures():
    pandas = pytest.importorskip("pandas")
    X = pandas.DataFrame(
        np.random.default_rng(7).normal(size=(8, 3)),
        columns=["amount", "rate", "age"],
    )
    y = pandas.Series(np.arange(8, dtype=np.float32), name="target")
    model = psann.PSANNRegressor(
        epochs=1,
        batch_size=4,
        hidden_layers=1,
        hidden_units=4,
        random_state=0,
    ).fit(X, y)

    assert model.n_features_in_ == 3
    assert model.feature_names_in_.tolist() == ["amount", "rate", "age"]
    with pytest.raises(ValueError, match="feature order"):
        model.predict(X[["rate", "amount", "age"]])
    with pytest.raises(ValueError, match="missing="):
        model.predict(X[["amount", "rate"]])
    unexpected = X.assign(country=1.0)
    with pytest.raises(ValueError, match="unexpected="):
        model.predict(unexpected)
    duplicate = X.copy()
    duplicate.columns = ["amount", "amount", "age"]
    with pytest.raises(ValueError, match="duplicate"):
        psann.PSANNRegressor(epochs=1).fit(duplicate, y)
    model.set_feature_schema_policy("reorder")
    reordered = model.predict(X[["rate", "amount", "age"]])
    assert reordered.shape == (8,)


def test_configured_output_and_preprocessing_contract_survive_regressor_load(
    tmp_path: Path,
):
    pandas = pytest.importorskip("pandas")
    X = pandas.DataFrame(
        np.random.default_rng(8).normal(size=(8, 2)),
        columns=["x0", "x1"],
    )
    y = pandas.DataFrame(
        np.random.default_rng(9).normal(size=(8, 2)),
        columns=["low", "high"],
    )
    spec = ModelSpec(
        input_schema=DataSchema(
            feature_names=("x0", "x1"),
            output_names=("low", "high"),
            input_shape=(2,),
            feature_policy="reorder",
            preprocessing={"owner": "platform"},
            target_scaling={"kind": "standard"},
        ),
        parameters={**_parameters(), "scaler": "standard", "target_scaler": "standard"},
    )
    model = create_model(spec)
    train(model, (X, y), config=_config())
    path = tmp_path / "trusted.pt"
    model.save(str(path))
    loaded = psann.PSANNRegressor.load(str(path))

    assert loaded.feature_names_in_.tolist() == ["x0", "x1"]
    assert loaded.output_names_.tolist() == ["low", "high"]
    assert loaded.preprocessing_contract_["declared"]["owner"] == "platform"
    assert loaded._platform_model_spec_dict_["backbone"] == "psann_mlp"
    assert loaded.predict(X[["x1", "x0"]]).shape == (8, 2)


def test_classifier_schema_survives_trusted_snapshot(tmp_path: Path):
    pandas = pytest.importorskip("pandas")
    X = pandas.DataFrame(
        np.random.default_rng(10).normal(size=(8, 2)),
        columns=["first", "second"],
    )
    y = np.asarray([0, 1] * 4)
    classifier = psann.PSANNClassifier(
        task="binary",
        estimator_params={**_parameters(), "epochs": 1, "batch_size": 4},
    ).fit(X, y)
    path = tmp_path / "classifier.pt"
    classifier.save(path)
    loaded = psann.PSANNClassifier.load(path)

    assert loaded.feature_names_in_.tolist() == ["first", "second"]
    with pytest.raises(ValueError, match="feature order"):
        loaded.predict(X[["second", "first"]])


def test_scoped_plugin_registration_and_arbitrary_module_adapter(tmp_path: Path):
    def factory(parameters: dict[str, object] | object) -> torch.nn.Module:
        values = dict(parameters)  # type: ignore[arg-type]
        return torch.nn.Linear(int(values["input_dim"]), int(values["output_dim"]))

    registration = register_backbone(
        "test.linear_plugin",
        factory,
        supported_tasks=("regression",),
        input_ranks=(1,),
        activations=("relu",),
        factory_kind="torch_module",
        experimental=True,
        plugin="tests",
        replace=True,
    )
    model = create_model(
        ModelSpec(
            backbone=registration.identifier,
            activation="relu",
            input_schema=DataSchema(input_shape=(2,)),
            parameters={"input_dim": 2, "output_dim": 1},
        )
    )
    X = np.random.default_rng(11).normal(size=(8, 2)).astype(np.float32)
    y = X[:, :1]
    train(model, (X, y), config=_config())

    assert isinstance(model, TorchModuleAdapter)
    assert model.predict(X[:2]).shape == (2, 1)
    assert model.experimental_ is True
    artifact = model.export(tmp_path / "custom.psann")
    restored = psann.load_model(artifact)
    np.testing.assert_allclose(restored.predict(X[:2]), model.predict(X[:2]), atol=1e-7)
    assert psann.inspect_artifact(artifact).experimental is True


def test_registered_torch_module_rejects_classification_capability_claims():
    def factory(parameters: dict[str, object] | object) -> torch.nn.Module:
        values = dict(parameters)  # type: ignore[arg-type]
        return torch.nn.Linear(int(values["input_dim"]), 1)

    with pytest.raises(ValueError, match="regression only"):
        register_backbone(
            "test.invalid_module_classifier",
            factory,
            supported_tasks=("regression", "binary"),
            input_ranks=(1,),
            activations=("relu",),
            factory_kind="torch_module",
            experimental=True,
            plugin="tests",
            replace=True,
        )


def test_direct_arbitrary_module_adapter_uses_shared_training_loop():
    X = np.random.default_rng(12).normal(size=(8, 2)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    model = adapt_module(
        torch.nn.Linear(2, 1),
        task=TaskSpec(kind="binary"),
        epochs=1,
        batch_size=4,
        random_state=0,
    ).fit(X, y)
    assert model.predict_proba(X[:2]).shape == (2, 2)
    assert len(model.history_) == 1
    with pytest.raises(NotImplementedError, match="in-process"):
        model.export("arbitrary.psann")


def test_schema_transform_extension_points_are_lazy_and_validated():
    constructed: list[str] = []

    class Transform:
        def __init__(self) -> None:
            constructed.append("constructed")

    register_schema_transform(
        "missing_value_imputer",
        "test.imputer",
        Transform,
        replace=True,
    )
    spec = ModelSpec(
        input_schema=DataSchema(
            input_shape=(2,),
            missing_value_imputer="test.imputer",
        ),
        parameters=_parameters(),
    )
    create_model(spec)
    assert constructed == []
    with pytest.raises(ValueError, match="Unknown missing-value imputer"):
        create_model(
            ModelSpec(
                input_schema=DataSchema(
                    input_shape=(2,),
                    missing_value_imputer="missing.plugin",
                )
            )
        )


def test_training_run_exports_native_artifact(tmp_path: Path):
    X = np.ones((4, 2), dtype=np.float32)
    y = np.ones(4, dtype=np.float32)
    run = train(
        create_model(
            ModelSpec(
                input_schema=DataSchema(input_shape=(2,)),
                parameters=_parameters(),
            )
        ),
        (X, y),
        config=_config(),
    )
    artifact = run.export(tmp_path / "model.psann")
    loaded = psann.load_model(artifact)
    np.testing.assert_allclose(loaded.predict(X), run.model.predict(X))
