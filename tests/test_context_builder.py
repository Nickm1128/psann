import numpy as np
import pytest

torch = pytest.importorskip("torch")

from psann import PSANNRegressor


def make_dataset(n: int = 32, features: int = 3):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, features)).astype(np.float32)
    y = (X.sum(axis=1, keepdims=True) + 0.1).astype(np.float32)
    return X, y


def test_context_builder_set_params_resets_callable():
    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        context_builder="cosine",
        context_builder_params={"frequencies": 1, "include_sin": True, "include_cos": False},
    )
    builder_first = est._get_context_builder()
    assert builder_first is not None

    est.set_params(
        context_builder_params={"frequencies": 2, "include_sin": False, "include_cos": True}
    )
    assert est._context_builder_callable_ is None
    builder_second = est._get_context_builder()
    assert builder_second is not builder_first


def test_context_builder_disable_clears_context_dim():
    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=12,
        context_builder="cosine",
        context_builder_params={"frequencies": 1, "include_sin": False, "include_cos": True},
    )
    est._context_dim_ = 3
    assert est._context_builder_callable_ is None
    est._get_context_builder()
    assert est._context_builder_callable_ is not None
    est.set_params(context_builder=None)
    assert est._context_builder_callable_ is None
    assert est._context_dim_ is None


def test_context_builder_params_are_copied():
    params = {"frequencies": [1, 2], "include_sin": True, "include_cos": False}
    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        context_builder="cosine",
        context_builder_params=params,
    )
    params["frequencies"].append(3)
    assert est.context_builder_params["frequencies"] == [1, 2]

    new_input = {"frequencies": [5], "include_sin": False, "include_cos": True}
    est.set_params(context_builder_params=new_input)
    new_input["frequencies"].append(6)
    assert est.context_builder_params["frequencies"] == [5]
