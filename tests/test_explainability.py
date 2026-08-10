from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import psann

shap = pytest.importorskip("shap")


def _targets(task: str, inputs: np.ndarray) -> np.ndarray:
    flattened = inputs.reshape(inputs.shape[0], -1)
    if task == "regression":
        return (flattened[:, 0] - 0.5 * flattened[:, 1]).astype(np.float32)
    if task == "multioutput":
        return np.stack((flattened[:, 0], flattened[:, 1]), axis=1).astype(np.float32)
    if task == "binary":
        return np.asarray(["yes" if index % 2 else "no" for index in range(len(inputs))])
    if task == "multiclass":
        return np.asarray(["red", "green", "blue"] * ((len(inputs) + 2) // 3))[: len(inputs)]
    if task == "multilabel":
        return np.stack(
            ((flattened[:, 0] > 0).astype(np.float32), (flattened[:, 1] > 0).astype(np.float32)),
            axis=1,
        )
    raise AssertionError(task)


def _fit(
    task: str = "regression",
    *,
    backbone: str = "psann_mlp",
    shape: tuple[int, ...] = (3,),
    activation: str = "psann",
    data_format: str = "channels_first",
    parameters: dict[str, object] | None = None,
):
    inputs = np.random.default_rng(601).normal(size=(12, *shape)).astype(np.float32)
    targets = _targets(task, inputs)
    if task == "regression":
        task_spec = psann.TaskSpec(kind="regression")
        output_names = ("forecast",)
    elif task == "multioutput":
        task_spec = psann.TaskSpec(kind="regression")
        output_names = ("north", "south")
    elif task == "binary":
        task_spec = psann.TaskSpec(kind="binary", positive_label="yes")
        output_names = ()
    elif task == "multiclass":
        task_spec = psann.TaskSpec(kind="multiclass")
        output_names = ()
    else:
        task_spec = psann.TaskSpec(kind="multilabel", class_names=("fraud", "review"))
        output_names = ("fraud", "review")
    names = tuple(f"feature_{index}" for index in range(shape[0])) if len(shape) == 1 else ()
    model = psann.create_model(
        psann.ModelSpec(
            task=task_spec,
            backbone=backbone,
            input_schema=psann.DataSchema(
                input_shape=shape,
                feature_names=names,
                output_names=output_names,
                data_format=data_format,
            ),
            activation=activation,
            parameters={
                "hidden_layers": 1,
                "hidden_units": 4,
                "random_state": 601,
                **dict(parameters or {}),
            },
        )
    )
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            deterministic=True,
        ),
    )
    return model, run, inputs


@pytest.fixture(scope="module")
def regression_model():
    return _fit()


@pytest.fixture(scope="module")
def relu_model():
    return _fit(
        activation="relu",
        parameters={"scaler": "standard", "target_scaler": "standard"},
    )


def test_base_import_does_not_import_shap():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, psann; assert 'shap' not in sys.modules; "
            "assert psann.ExplainerConfig().algorithm == 'auto'; "
            "assert 'shap' not in sys.modules",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


def test_explain_extra_tracks_python_compatibility_policy():
    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "shap>=0.50,<0.52; python_version < '3.12'" in pyproject
    assert "shap>=0.50,<0.53; python_version >= '3.12'" in pyproject


def test_explainer_config_roundtrip_and_validation():
    config = psann.ExplainerConfig(
        algorithm="gradient",
        output="forecast",
        background_size=4,
        max_background_samples=8,
        seed=71,
    )
    assert psann.ExplainerConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="background_size"):
        psann.ExplainerConfig(background_size=9, max_background_samples=8)
    with pytest.raises(ValueError, match="max_evaluations"):
        psann.ExplainerConfig(max_evaluations=0)


def test_background_is_explicit_sampled_and_persisted_only_with_approval(
    regression_model,
    tmp_path: Path,
):
    model, _, inputs = regression_model
    with pytest.raises(psann.BackgroundPolicyError, match="exactly one"):
        psann.make_explainer(model)
    with pytest.raises(psann.BackgroundPolicyError, match="exactly one"):
        psann.make_explainer(model, background=inputs[:2], reference_data=inputs)

    first = psann.summarize_background(inputs, max_samples=4, seed=17)
    second = psann.summarize_background(inputs, max_samples=4, seed=17)
    np.testing.assert_array_equal(first.values, second.values)
    with pytest.raises(psann.BackgroundPolicyError, match="approved"):
        psann.save_explainer_config(
            psann.ExplainerConfig(),
            tmp_path / "unsafe.json",
            background_summary=first,
            include_background=True,
        )

    approved = psann.summarize_background(
        inputs,
        max_samples=4,
        seed=17,
        approved_for_persistence=True,
        metadata={"review": "approved"},
    )
    path = psann.save_explainer_config(
        psann.ExplainerConfig(seed=17),
        tmp_path / "explainer.json",
        background_summary=approved,
        include_background=True,
    )
    config, loaded = psann.load_explainer_config(path)
    assert config.seed == 17
    assert loaded is not None
    np.testing.assert_array_equal(loaded.values, approved.values)

    config_only = psann.save_explainer_config(
        psann.ExplainerConfig(),
        tmp_path / "config-only.json",
    )
    payload = json.loads(config_only.read_text(encoding="utf-8"))
    assert "background_summary" not in payload


def test_named_regression_explanation_is_additive_and_deterministic(regression_model):
    model, _, inputs = regression_model
    config = psann.ExplainerConfig(
        max_evaluations=20,
        max_explanation_samples=2,
        seed=41,
    )
    explainer = psann.make_explainer(model, background=inputs[:4], config=config)
    first = explainer.explain(inputs[4:6])
    second = explainer.explain(inputs[4:6])

    assert isinstance(first.explanation, shap.Explanation)
    assert first.values.shape == (2, 3, 1)
    assert first.base_values.shape == (2, 1)
    assert first.output_names == ("forecast",)
    assert first.explanation.feature_names == ["feature_0", "feature_1", "feature_2"]
    assert first.metadata["additivity_error"] < 1e-5
    np.testing.assert_allclose(first.values, second.values, rtol=0.0, atol=0.0)


def test_runtime_methods_and_artifact_explanations_match(regression_model, tmp_path: Path):
    model, run, inputs = regression_model
    artifact = run.export(tmp_path / "forecast.psann")
    runtime = psann.load_runtime(artifact)
    config = psann.ExplainerConfig(max_evaluations=20, seed=19)

    fitted = psann.explain(model, inputs[5:7], background=inputs[:4], config=config)
    deployed = runtime.explain(inputs[5:7], background=inputs[:4], config=config)

    np.testing.assert_allclose(deployed.values, fitted.values, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        deployed.base_values,
        fitted.base_values,
        rtol=1e-6,
        atol=1e-7,
    )
    assert deployed.model_id is not None
    assert deployed.artifact_version is not None
    assert runtime.make_explainer(background=inputs[:4], config=config).runtime is runtime
    info = psann.inspect_artifact(artifact)
    assert "model_agnostic_explanations" in info.capabilities
    assert "gradient_explanations" in info.capabilities


@pytest.mark.parametrize(
    ("task", "expected_outputs"),
    [
        ("multioutput", ("north", "south")),
        ("binary", ("no", "yes")),
        ("multiclass", ("blue", "green", "red")),
        ("multilabel", ("fraud", "review")),
    ],
)
def test_task_output_shape_and_names(task: str, expected_outputs: tuple[str, ...]):
    model, _, inputs = _fit(task)
    result = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:3],
        config=psann.ExplainerConfig(max_evaluations=20),
    )
    assert result.values.shape == (1, 3, len(expected_outputs))
    assert result.base_values.shape == (1, len(expected_outputs))
    assert result.output_names == expected_outputs
    assert result.metadata["additivity_error"] < 1e-5


def test_binary_probability_logit_and_named_output_selection():
    model, _, inputs = _fit("binary", activation="relu")
    probability = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:3],
        config=psann.ExplainerConfig(
            output_kind="probability",
            output="yes",
            max_evaluations=20,
        ),
    )
    logit = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:3],
        config=psann.ExplainerConfig(output_kind="logit", max_evaluations=20),
    )
    assert probability.values.shape == (1, 3, 1)
    assert probability.output_names == ("yes",)
    assert probability.metadata["output_kind"] == "probability"
    assert logit.output_names == ("yes",)
    assert logit.metadata["output_kind"] == "logit"


@pytest.mark.parametrize(
    ("algorithm", "masker"),
    [
        ("permutation", "independent"),
        ("permutation", "partition"),
        ("partition", "domain"),
    ],
)
def test_independent_partition_and_domain_maskers(regression_model, algorithm, masker):
    model, _, inputs = regression_model
    result = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:3],
        config=psann.ExplainerConfig(
            algorithm=algorithm,
            masker=masker,
            max_evaluations=20,
        ),
    )
    assert result.metadata["masker"] == masker
    assert result.metadata["additivity_error"] < 1e-5


@pytest.mark.parametrize(
    ("backbone", "shape", "group_strategy", "group_count"),
    [
        ("psann_conv1d", (2, 3), "spatial_region", 3),
        ("sgr_psann", (2, 2), "time_step", 2),
    ],
)
def test_spatial_and_sequence_explanations_preserve_shape_and_groups(
    backbone,
    shape,
    group_strategy,
    group_count,
):
    model, _, inputs = _fit(backbone=backbone, shape=shape)
    result = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:3],
        config=psann.ExplainerConfig(max_evaluations=100),
    )
    assert result.values.shape == (1, *shape, 1)
    assert result.metadata["group_strategy"] == group_strategy
    assert len(result.feature_groups) == group_count
    assert sorted(index for group in result.feature_groups for index in group.indices) == list(
        range(int(np.prod(shape)))
    )
    assert result.metadata["additivity_error"] < 1e-5


def test_channel_grouping_preserves_data_format():
    model, _, inputs = _fit(backbone="psann_conv1d", shape=(3, 2))
    explainer = psann.make_explainer(
        model,
        background=inputs[:3],
        config=psann.ExplainerConfig(
            algorithm="partition",
            masker="domain",
            group_strategy="channel",
            max_evaluations=100,
        ),
    )
    assert [group.indices for group in explainer.feature_groups] == [
        (0, 1),
        (2, 3),
        (4, 5),
    ]


@pytest.mark.parametrize("scaler", [None, "standard", "minmax"])
def test_differentiable_adapter_matches_raw_runtime_for_builtin_scalers(scaler):
    model, _, inputs = _fit(
        activation="relu",
        parameters={"scaler": scaler, "target_scaler": scaler},
    )
    runtime = psann.create_inference_runtime(model)
    adapter = psann.DifferentiableInferenceAdapter(
        runtime,
        output_kind="prediction",
        output_indices=(0,),
    )
    with torch.no_grad():
        observed = adapter(torch.as_tensor(inputs[:3].reshape(3, -1))).cpu().numpy()
    expected = np.asarray(runtime.predict(inputs[:3]).values).reshape(3, 1)
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_differentiable_adapter_matches_channels_last_convolution():
    model, _, inputs = _fit(
        backbone="psann_conv1d",
        shape=(3, 2),
        data_format="channels_last",
        parameters={"scaler": "standard", "target_scaler": "standard"},
    )
    runtime = psann.create_inference_runtime(model)
    adapter = psann.DifferentiableInferenceAdapter(
        runtime,
        output_kind="prediction",
        output_indices=(0,),
    )
    with torch.no_grad():
        observed = adapter(torch.as_tensor(inputs[:2].reshape(2, -1))).cpu().numpy()
    expected = np.asarray(runtime.predict(inputs[:2]).values).reshape(2, 1)
    np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_gradient_and_deep_explainers_include_preprocessing(relu_model):
    model, _, inputs = relu_model
    gradient_config = psann.ExplainerConfig(
        algorithm="gradient",
        gradient_samples=100,
        seed=37,
    )
    first = psann.explain(
        model,
        inputs[5:7],
        background=inputs[:5],
        config=gradient_config,
    )
    second = psann.explain(
        model,
        inputs[5:7],
        background=inputs[:5],
        config=gradient_config,
    )
    deep = psann.explain(
        model,
        inputs[5:7],
        background=inputs[:5],
        config=psann.ExplainerConfig(algorithm="deep"),
    )
    assert first.metadata["algorithm"] == "gradient"
    assert first.metadata["state_policy"] == "frozen_clone"
    assert first.metadata["additivity_error"] < 0.05
    np.testing.assert_allclose(first.values, second.values, rtol=0.0, atol=0.0)
    assert deep.metadata["algorithm"] == "deep"
    assert deep.metadata["additivity_error"] < 1e-4


def test_uncertified_deep_explainer_falls_back_or_fails(regression_model):
    model, _, inputs = regression_model
    fallback = psann.make_explainer(
        model,
        background=inputs[:3],
        config=psann.ExplainerConfig(algorithm="deep", max_evaluations=20),
    )
    assert fallback.algorithm == "permutation"
    assert "ReLU" in str(fallback.fallback_reason)
    with pytest.raises(psann.ExplanationCapabilityError, match="ReLU"):
        psann.make_explainer(
            model,
            background=inputs[:3],
            config=psann.ExplainerConfig(algorithm="deep", fallback="error"),
        )


def test_custom_scaler_and_context_builder_fall_back_with_reason():
    class ShiftScaler:
        def fit(self, values):
            self.shift = np.asarray(values).mean(axis=0)
            return self

        def transform(self, values):
            return np.asarray(values) - self.shift

    inputs = np.random.default_rng(611).normal(size=(10, 2)).astype(np.float32)
    targets = inputs[:, 0]
    scaler_model = psann.PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=2,
        random_state=611,
        scaler=ShiftScaler(),
    ).fit(inputs, targets)
    scaler_explainer = psann.make_explainer(
        scaler_model,
        background=inputs[:3],
        config=psann.ExplainerConfig(algorithm="gradient", max_evaluations=10),
    )
    assert scaler_explainer.algorithm == "permutation"
    assert "custom input scaler" in str(scaler_explainer.fallback_reason)

    context_model = psann.WaveResNetRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=2,
        random_state=612,
        context_builder=lambda values: np.asarray(values, dtype=np.float32),
    ).fit(inputs, targets)
    context_explainer = psann.make_explainer(
        context_model,
        background=inputs[:3],
        config=psann.ExplainerConfig(algorithm="gradient", max_evaluations=10),
    )
    assert context_explainer.algorithm == "permutation"
    assert "custom context builder" in str(context_explainer.fallback_reason)


def test_explicit_context_requirement_fails_clearly():
    inputs = np.random.default_rng(613).normal(size=(10, 2)).astype(np.float32)
    context = np.random.default_rng(614).normal(size=(10, 1)).astype(np.float32)
    model = psann.WaveResNetRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=2,
        random_state=613,
        context_dim=1,
    ).fit(inputs, inputs[:, 0], context=context)
    with pytest.raises(psann.ExplanationCapabilityError, match="explicit context"):
        psann.make_explainer(model, background=inputs[:3])


def test_stateful_explanations_do_not_mutate_shared_state():
    model, _, inputs = _fit(parameters={"stateful": True})
    before = {name: tensor.detach().clone() for name, tensor in model.model_.state_dict().items()}
    model_agnostic = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:3],
        config=psann.ExplainerConfig(max_evaluations=20),
    )
    gradient = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:3],
        config=psann.ExplainerConfig(algorithm="gradient", gradient_samples=50),
    )
    for name, tensor in model.model_.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
    assert model_agnostic.metadata["state_policy"] == "stateless_runtime"
    assert gradient.metadata["state_policy"] == "frozen_clone"


def test_registered_intermediate_layer_explanations(relu_model):
    model, _, inputs = relu_model
    assert psann.list_explainable_layers(model) == ("hidden_0", "output")
    result = psann.explain(
        model,
        inputs[4:5],
        background=inputs[:4],
        config=psann.ExplainerConfig(
            algorithm="deep",
            layer="output",
        ),
    )
    assert result.values.shape == (1, 4, 1)
    assert result.explanation.data is None
    assert result.metadata["layer"] == "output"
    assert result.metadata["feature_names"] == [
        "output[0]",
        "output[1]",
        "output[2]",
        "output[3]",
    ]
    with pytest.raises(psann.ExplanationCapabilityError, match="not registered"):
        psann.make_explainer(
            model,
            background=inputs[:4],
            config=psann.ExplainerConfig(algorithm="gradient", layer="private.path"),
        )


def test_request_and_evaluation_limits_are_enforced(regression_model):
    model, _, inputs = regression_model
    with pytest.raises(psann.ExplanationCapabilityError, match="max_evaluations"):
        psann.make_explainer(
            model,
            background=inputs[:3],
            config=psann.ExplainerConfig(
                algorithm="permutation",
                max_evaluations=6,
            ),
        )
    explainer = psann.make_explainer(
        model,
        background=inputs[:3],
        config=psann.ExplainerConfig(
            max_evaluations=20,
            max_explanation_samples=1,
        ),
    )
    with pytest.raises(psann.ExplanationCapabilityError, match="configured maximum"):
        explainer.explain(inputs[4:6])


def test_drift_and_offline_report_exclude_raw_rows(regression_model, tmp_path: Path):
    model, _, inputs = regression_model
    explainer = psann.make_explainer(
        model,
        background=inputs[:4],
        config=psann.ExplainerConfig(max_evaluations=20),
    )
    reference = explainer.explain(inputs[4:6])
    current = explainer.explain(inputs[6:8])
    drift = psann.summarize_explanation_drift(reference, current)
    assert drift.feature_names == ("feature_0", "feature_1", "feature_2")
    assert drift.mean_absolute_shift >= 0.0
    report = psann.write_explanation_report(current, tmp_path / "summary.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["contains_raw_inputs"] is False
    assert payload["contains_row_level_attributions"] is False
    assert "values" not in payload
    assert set(payload["mean_absolute_attribution"]) == set(drift.feature_names)
