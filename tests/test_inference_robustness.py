from __future__ import annotations

import numpy as np
import pytest

import psann


@pytest.fixture
def runtime():
    inputs = np.random.default_rng(951).normal(size=(12, 3)).astype(np.float32)
    targets = (inputs[:, 0] - inputs[:, 1]).astype(np.float32)
    model = psann.create_model(
        psann.ModelSpec(
            input_schema=psann.DataSchema(input_shape=(3,)),
            parameters={"hidden_layers": 1, "hidden_units": 4, "random_state": 951},
        )
    )
    psann.train(
        model,
        (inputs, targets),
        config=psann.TrainingConfig(epochs=1, batch_size=4),
    )
    return psann.create_inference_runtime(
        model,
        config=psann.InferenceConfig(batch_size=127),
    )


def test_empty_single_large_noncontiguous_and_mixed_dtype_inputs(runtime):
    with pytest.raises(ValueError, match="at least one sample"):
        runtime.predict(np.empty((0, 3), dtype=np.float32))

    single = runtime.predict(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert single.values.shape == (1,)

    large = np.arange(3 * 4097, dtype=np.float64).reshape(4097, 3)
    result = runtime.predict(large)
    assert result.values.shape == (4097,)
    assert result.metadata["chunks"] == 33

    base = np.arange(6 * 20, dtype=np.int16).reshape(20, 6)
    noncontiguous = base[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    observed = runtime.predict(noncontiguous)
    assert observed.values.shape == (20,)


def test_missing_and_reordered_named_columns(runtime):
    pandas = pytest.importorskip("pandas")
    core = runtime.model
    core.feature_names_in_ = np.asarray(["alpha", "beta", "gamma"], dtype=object)
    correct = pandas.DataFrame({"alpha": [1.0], "beta": [2.0], "gamma": [3.0]})
    assert runtime.predict(correct).values.shape == (1,)
    with pytest.raises(ValueError, match="missing=.*gamma"):
        runtime.predict(correct.drop(columns=["gamma"]))
    reordered = correct[["gamma", "alpha", "beta"]]
    with pytest.raises(ValueError, match="feature order"):
        runtime.predict(reordered)
    reordered_runtime = psann.create_inference_runtime(
        core,
        config=psann.InferenceConfig(feature_policy="reorder"),
    )
    np.testing.assert_allclose(
        reordered_runtime.predict(reordered).values,
        runtime.predict(correct).values,
    )


def test_malformed_and_missing_context_fail_clearly():
    inputs = np.random.default_rng(952).normal(size=(10, 3)).astype(np.float32)
    context = np.random.default_rng(953).normal(size=(10, 2)).astype(np.float32)
    model = psann.WaveResNetRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=5,
        context_dim=2,
        random_state=952,
    ).fit(inputs, inputs[:, 0], context=context)
    runtime = psann.create_inference_runtime(model)
    with pytest.raises(ValueError, match="matching context"):
        runtime.predict(inputs[:2])
    with pytest.raises(ValueError, match="samples"):
        runtime.predict(inputs[:2], context=context[:1])
    with pytest.raises(ValueError, match="feature dimension"):
        runtime.predict(inputs[:2], context=np.zeros((2, 3), dtype=np.float32))
