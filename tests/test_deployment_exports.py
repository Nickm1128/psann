from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import psann
from psann.platform import DataSchema, ModelSpec, TaskSpec, TrainingConfig

BACKBONES = [
    ("psann_mlp", (3,)),
    ("respsann_mlp", (3,)),
    ("psann_conv1d", (1, 4)),
    ("psann_conv2d", (1, 3, 3)),
    ("psann_conv3d", (1, 2, 2, 2)),
    ("respsann_conv2d", (1, 3, 3)),
    ("wave_resnet", (3,)),
    ("sgr_psann", (3,)),
]
TASKS = ("regression", "binary", "multiclass", "multilabel")
HAS_ONNX_STACK = all(
    importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime", "onnxscript")
)


def _targets(task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray([0.2, -0.1, 0.4, 0.8, -0.5, 0.3], dtype=np.float32)
    if task == "binary":
        return np.asarray([0, 1, 0, 1, 0, 1])
    if task == "multiclass":
        return np.asarray([0, 1, 2, 0, 1, 2])
    return np.asarray([[0, 1], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1]])


def _task_spec(task: str) -> TaskSpec:
    if task == "multilabel":
        return TaskSpec(kind="multilabel", class_names=("left", "right"))
    return TaskSpec(kind=task)  # type: ignore[arg-type]


def _model(backbone: str, shape: tuple[int, ...], task: str = "regression"):
    inputs = np.random.default_rng(301).normal(size=(6, *shape)).astype(np.float32)
    parameters: dict[str, object] = {
        "hidden_layers": 1,
        "hidden_units": 4,
        "random_state": 19,
    }
    if "conv" in backbone:
        parameters["conv_channels"] = 4
    model = psann.create_model(
        ModelSpec(
            task=_task_spec(task),
            backbone=backbone,
            input_schema=DataSchema(input_shape=shape),
            parameters=parameters,
        )
    )
    psann.train(
        model,
        (inputs, _targets(task)),
        config=TrainingConfig(epochs=1, batch_size=3, deterministic=True),
    )
    return model, inputs


@pytest.mark.parametrize(("backbone", "shape"), BACKBONES)
@pytest.mark.parametrize("task", TASKS)
def test_torch_export_is_parity_certified_for_declared_task_backbone_matrix(
    backbone,
    shape,
    task,
):
    model, inputs = _model(backbone, shape, task)

    report = psann.evaluate_export_capabilities(
        model,
        inputs,
        formats=("torch_export",),
    )

    assert report.backbone == backbone
    assert report.task == task
    assert report.supported_formats == ("torch_export",)
    assert report.advertised_formats == ("native", "torch_export")
    capability = report.capabilities[0]
    assert capability.supported
    assert capability.parity
    assert capability.dynamic_batch
    assert capability.max_abs_error is not None
    assert capability.max_abs_error <= 1e-5


@pytest.mark.skipif(not HAS_ONNX_STACK, reason="install psann[export] for ONNX certification")
@pytest.mark.parametrize(("backbone", "shape"), BACKBONES)
@pytest.mark.parametrize("task", TASKS)
def test_onnx_is_parity_certified_for_declared_task_backbone_matrix(
    backbone,
    shape,
    task,
):
    model, inputs = _model(backbone, shape, task)

    report = psann.evaluate_export_capabilities(model, inputs, formats=("onnx",))

    assert report.supported_formats == ("onnx",)
    assert report.advertised_formats == ("native", "onnx")
    capability = report.capabilities[0]
    assert capability.supported
    assert capability.parity
    assert capability.dynamic_batch
    assert capability.max_abs_error is not None
    assert capability.max_abs_error <= 1e-5


def test_derived_export_writes_program_and_preprocessing_contract(tmp_path: Path):
    model, inputs = _model("psann_mlp", (3,))
    model.preprocessing_contract_["declared"] = {"owner": "deployment-test"}

    exported = psann.export_derived(
        model,
        tmp_path / "forecast.pt2",
        format="torch_export",
        sample_inputs=inputs,
    )

    assert exported.path.is_file()
    assert exported.contract_path.is_file()
    contract = json.loads(exported.contract_path.read_text(encoding="utf-8"))
    assert contract["contract_format"] == "psann.preprocessing"
    assert contract["source_of_truth"] == "native_psann_artifact"
    assert contract["tensor_input"]["shape"] == ["batch", 3]
    assert contract["tensor_input"]["preprocessing"]["declared"]["owner"] == "deployment-test"
    assert contract["derived_export"]["certification"]["supported"] is True

    prepared, _, _ = model._prepare_inference_inputs(inputs[:3])
    program = torch.export.load(exported.path)
    with torch.inference_mode():
        expected = model.model_(torch.from_numpy(prepared))
        observed = program.module()(torch.from_numpy(prepared))
    torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not HAS_ONNX_STACK, reason="install psann[export] for ONNX writing")
def test_derived_onnx_export_writes_model_and_contract(tmp_path: Path):
    model, inputs = _model("respsann_mlp", (3,), "binary")

    exported = psann.export_derived(
        model,
        tmp_path / "classifier.onnx",
        format="onnx",
        sample_inputs=inputs,
    )

    assert exported.path.stat().st_size > 0
    assert exported.contract_path.is_file()
    contract = json.loads(exported.contract_path.read_text(encoding="utf-8"))
    assert contract["tensor_output"]["kind"] == "raw_logits"
    assert contract["derived_export"]["format"] == "onnx"


def test_derived_export_requires_certified_suffix(tmp_path: Path):
    model, inputs = _model("psann_mlp", (3,))
    with pytest.raises(ValueError, match=r"\.pt2"):
        psann.export_derived(
            model,
            tmp_path / "model.bin",
            format="torch_export",
            sample_inputs=inputs,
        )


def test_export_evaluation_rejects_unknown_format():
    model, inputs = _model("psann_mlp", (3,))
    with pytest.raises(ValueError, match="Unknown"):
        psann.evaluate_export_capabilities(
            model,
            inputs,
            formats=("tensorrt",),  # type: ignore[arg-type]
        )


def test_onnx_is_advertised_only_when_optional_stack_passes():
    model, inputs = _model("psann_mlp", (3,))

    report = psann.evaluate_export_capabilities(model, inputs, formats=("onnx",))
    capability = report.capabilities[0]
    if HAS_ONNX_STACK:
        assert capability.supported, capability.reason
        assert report.advertised_formats == ("native", "onnx")
    else:
        assert not capability.supported
        assert report.advertised_formats == ("native",)
        assert "optional" in str(capability.reason)


def test_export_requires_two_samples_for_dynamic_batch():
    model, inputs = _model("psann_mlp", (3,))
    with pytest.raises(psann.DerivedExportError, match="at least two"):
        psann.evaluate_export_capabilities(
            model,
            inputs[:1],
            formats=("torch_export",),
        )
