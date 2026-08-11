from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import psann

pytestmark = pytest.mark.gpu


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for workplace accelerator certification.")


def _fixture(seed: int = 1001):
    inputs = np.random.default_rng(seed).normal(size=(16, 4)).astype(np.float32)
    targets = (inputs[:, 0] - 0.25 * inputs[:, 1]).astype(np.float32)
    spec = psann.ModelSpec(
        input_schema=psann.DataSchema(input_shape=(4,)),
        activation="relu",
        parameters={"hidden_layers": 1, "hidden_units": 8, "random_state": seed},
    )
    return spec, inputs, targets


def _write_evidence(output_dir: Path, name: str, value: dict) -> None:
    (output_dir / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_cuda_fp32_forward_backward_artifact_and_inference(output_dir: Path):
    _require_cuda()
    spec, inputs, targets = _fixture()
    model = psann.create_model(spec)
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            device="cuda",
            deterministic=True,
            fallback_policy="error",
        ),
    )
    core = run.model
    assert next(core.model_.parameters()).device.type == "cuda"
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in core.model_.parameters()
    )
    artifact = run.export(output_dir / "cuda-fp32.psann")
    loaded = psann.load_model(artifact, device="cuda")
    runtime = psann.load_runtime(
        artifact,
        config=psann.InferenceConfig(
            batch_size=5,
            device="cuda",
            fallback_policy="error",
        ),
    )
    expected = np.asarray(core.predict(inputs))
    observed = np.asarray(runtime.predict(inputs).values)
    restored = np.asarray(loaded.predict(inputs))
    np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(restored, expected, rtol=1e-4, atol=1e-5)
    _write_evidence(
        output_dir,
        "cuda-fp32.json",
        {
            "device": str(runtime.device),
            "forward": "passed",
            "backward": "passed",
            "artifact": "passed",
            "inference": "passed",
        },
    )


@pytest.mark.parametrize("amp_dtype", ["float16", "bfloat16"])
def test_cuda_amp_training_matrix(amp_dtype: str, output_dir: Path):
    _require_cuda()
    if amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        pytest.skip("The active CUDA device does not support bfloat16.")
    spec, inputs, targets = _fixture(1002)
    model = psann.create_model(spec)
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            device="cuda",
            amp=True,
            amp_dtype=amp_dtype,
            fallback_policy="error",
        ),
    )
    starts = [event for event in run.model.training_events_ if event["name"] == "train_start"]
    assert starts[-1]["data"]["amp_effective"] is True
    assert amp_dtype in starts[-1]["data"]["amp_dtype"]
    assert np.isfinite(run.metrics["mse"])
    _write_evidence(
        output_dir,
        f"cuda-amp-{amp_dtype}.json",
        {
            "amp_dtype": amp_dtype,
            "amp_effective": True,
            "mse": run.metrics["mse"],
        },
    )


def test_cuda_compile_training(output_dir: Path):
    _require_cuda()
    spec, inputs, targets = _fixture(1005)
    model = psann.create_model(spec)
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            device="cuda",
            compile=True,
            fallback_policy="error",
        ),
    )
    starts = [event for event in run.model.training_events_ if event["name"] == "train_start"]
    assert starts[-1]["data"]["compile_effective"] is True
    assert np.isfinite(run.metrics["mse"])
    _write_evidence(
        output_dir,
        "cuda-compile.json",
        {
            "compile_requested": True,
            "compile_effective": True,
            "mse": run.metrics["mse"],
        },
    )


def test_cuda_resume_and_supported_exports(output_dir: Path):
    _require_cuda()
    spec, inputs, targets = _fixture(1003)
    checkpoint_dir = output_dir / "resume"
    first = psann.create_model(spec)
    psann.train(
        first,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            device="cuda",
            deterministic=True,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=1,
            fallback_policy="error",
        ),
    )
    checkpoint = checkpoint_dir / "latest.psann-train"
    assert checkpoint.is_file()
    resumed = psann.create_model(spec)
    run = psann.train(
        resumed,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=2,
            batch_size=4,
            device="cuda",
            deterministic=True,
            resume_from=str(checkpoint),
            fallback_policy="error",
        ),
    )
    assert len(run.history) == 2
    artifact = run.export(output_dir / "cuda-resumed.psann")

    export_model = psann.load_model(artifact, device="cpu")
    torch_export = psann.export_derived(
        export_model,
        output_dir / "cuda-scheduled.pt2",
        format="torch_export",
        sample_inputs=inputs,
    )
    assert torch_export.path.is_file()
    report = psann.evaluate_export_capabilities(
        export_model,
        inputs,
        formats=("torch_export", "onnx"),
    )
    assert "torch_export" in report.supported_formats
    assert "onnx" in report.supported_formats
    _write_evidence(
        output_dir,
        "cuda-resume-export.json",
        {
            "resume_epochs": len(run.history),
            "native_artifact": artifact.name,
            "supported_exports": list(report.supported_formats),
        },
    )


def test_cuda_explanation_and_memory_evidence(output_dir: Path):
    _require_cuda()
    spec, inputs, targets = _fixture(1004)
    model = psann.create_model(spec)
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            device="cuda",
            fallback_policy="error",
        ),
    )
    torch.cuda.reset_peak_memory_stats()
    runtime = psann.create_inference_runtime(
        run.model,
        config=psann.InferenceConfig(device="cuda", fallback_policy="error"),
    )
    result = runtime.explain(
        inputs[:2],
        background=inputs[2:8],
        config=psann.ExplainerConfig(
            algorithm="gradient",
            gradient_samples=32,
            fallback="error",
        ),
    )
    assert result.metadata["algorithm"] == "gradient"
    assert result.metadata["state_policy"] == "frozen_clone"
    _write_evidence(
        output_dir,
        "cuda-explanation-memory.json",
        {
            "algorithm": result.metadata["algorithm"],
            "additivity_error": result.metadata["additivity_error"],
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
    )
