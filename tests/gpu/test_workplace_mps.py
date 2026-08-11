from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import psann

pytestmark = pytest.mark.gpu


def test_mps_float32_training_and_inference_observation(output_dir: Path):
    backend = getattr(torch.backends, "mps", None)
    if backend is None or not backend.is_available():
        pytest.skip("MPS is unavailable; the MPS tier is experimental and observation-only.")

    inputs = np.random.default_rng(1010).normal(size=(12, 3)).astype(np.float32)
    targets = (inputs[:, 0] + inputs[:, 1]).astype(np.float32)
    model = psann.create_model(
        psann.ModelSpec(
            input_schema=psann.DataSchema(input_shape=(3,)),
            activation="relu",
            parameters={"hidden_layers": 1, "hidden_units": 8, "random_state": 1010},
        )
    )
    run = psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            device="mps",
            fallback_policy="error",
        ),
    )
    artifact = run.export(output_dir / "mps-observation.psann")
    runtime = psann.load_runtime(
        artifact,
        config=psann.InferenceConfig(device="mps", fallback_policy="error"),
    )
    result = runtime.predict(inputs)
    assert result.values.shape == (12,)
    assert np.isfinite(result.values).all()
    (output_dir / "mps-observation.json").write_text(
        json.dumps(
            {
                "status": "experimental",
                "device": str(runtime.device),
                "training": "passed",
                "artifact": "passed",
                "inference": "passed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
