from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from psann.platform import InferenceResult
from psann.platform.explain_contracts import BackgroundSummary, ExplainerConfig


def test_inference_result_preserves_typed_metadata():
    result = InferenceResult(
        values=[0.25, 0.75],
        task="binary",
        output_names=("negative", "positive"),
        artifact_version="1.0",
        model_id="model-1",
        metadata={"device": "cpu"},
    )

    assert result.task == "binary"
    assert result.output_names == ("negative", "positive")
    assert result.metadata["device"] == "cpu"


def test_inference_result_is_immutable():
    result = InferenceResult(values=[1.0], task="regression")

    with pytest.raises(FrozenInstanceError):
        result.task = "binary"  # type: ignore[misc]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_explainer_contracts_reject_nonfinite_json_values(invalid: float):
    with pytest.raises(TypeError, match="output must be"):
        ExplainerConfig(output=invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="finite JSON-serializable"):
        BackgroundSummary(
            values=np.ones((1, 2), dtype=np.float32),
            input_shape=(2,),
            metadata={"invalid": invalid},
        )
