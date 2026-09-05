"""Runtime invariants exposed while checking the public numerical utilities."""

import pytest
import torch
from torch import nn

from psann.state import StateController
from psann.utils.linear_probe import _unpack_batch


@pytest.mark.parametrize("route", ["explicit-none", "set-params", "clone"])
def test_legacy_wave_without_context_survives_parameter_reconstruction(route, tmp_path):
    import numpy as np
    from sklearn.base import clone
    from psann import PSANNRegressor, WaveResNetRegressor

    kwargs = dict(hidden_layers=2, hidden_units=8, epochs=2, random_state=31, device="cpu")
    if route == "explicit-none":
        kwargs["context_dim"] = None
    model = WaveResNetRegressor(**kwargs)
    if route == "set-params":
        model.set_params(lr=0.003)
    elif route == "clone":
        model = clone(model)
    x = np.random.default_rng(13).normal(size=(16, 4)).astype(np.float32)
    y = x[:, :1] * 0.3
    model.fit(x, y)
    expected = model.predict(x)
    assert model.architecture.context is None
    for generation in (1, 2):
        path = tmp_path / f"wave-{generation}.pt"
        model.save(path)
        model = PSANNRegressor.load(path)
        assert model.architecture.context is None
        np.testing.assert_array_equal(model.predict(x), expected)


@pytest.mark.parametrize(
    "name",
    [
        "ResPSANNRegressor",
        "ResConvPSANNRegressor",
        "SGRPSANNRegressor",
        "WaveResNetRegressor",
        "GeoSparseRegressor",
    ],
)
@pytest.mark.parametrize("lsm", [None, {"output_dim": 4, "hidden_units": 8, "hidden_layers": 1}])
@pytest.mark.parametrize("hidden_width", [None, 8])
def test_legacy_wrapper_with_flat_preprocessing_emits_one_caller_warning(name, lsm, hidden_width):
    import psann
    import warnings
    from pathlib import Path

    kwargs = {"lsm": lsm, "lsm_train": False, "lsm_pretrain_epochs": 0, "lsm_lr": None}
    if hidden_width is not None:
        kwargs["hidden_width"] = hidden_width
    if name == "GeoSparseRegressor":
        kwargs["shape"] = (2, 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        getattr(psann, name)(**kwargs)
    deprecations = [warning for warning in caught if warning.category is DeprecationWarning]
    assert len(deprecations) == 1
    assert Path(deprecations[0].filename).resolve() == Path(__file__).resolve()


def test_state_controller_preserves_recursive_module_apply_and_tensor_updates():
    controller = StateController(3, init=2.0)
    model = nn.Sequential(nn.Linear(3, 3), controller)
    visited = []
    assert model.apply(visited.append) is model
    assert visited == [model[0], controller, model]
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(controller.apply(values, feature_dim=1), 2 * values)
    controller.commit()
    assert not torch.equal(controller.state, torch.full((3,), 2.0))


@pytest.mark.parametrize("keys", [("x", "y", "c"), ("inputs", "targets", "context")])
def test_probe_dictionary_batches_preserve_multielement_tensors(keys):
    tensors = (torch.ones(3, 2), torch.arange(3), torch.zeros(3, 1))
    result = _unpack_batch(dict(zip(keys, tensors)))
    assert all(actual is expected for actual, expected in zip(result, tensors))


def test_probe_dictionary_short_keys_take_precedence():
    x, y = torch.zeros(3, 2), torch.zeros(3, dtype=torch.long)
    result = _unpack_batch({"x": x, "y": y, "inputs": torch.ones_like(x)})
    assert result[0] is x and result[1] is y and result[2] is None


@pytest.mark.parametrize("trainable", [False, True])
def test_geometric_preprocessor_width_controls_core_and_two_checkpoint_generations(
    tmp_path, trainable
):
    from psann import PSANNRegressor
    from psann.architectures import ArchitectureConfig, GeometryConfig
    from psann.preprocessing import LSMConfig, PreprocessorConfig, PreprocessorTrainingConfig
    import numpy as np

    estimator = PSANNRegressor(
        architecture=ArchitectureConfig.geometric_sparse(
            geometry=GeometryConfig(shape=(4, 4), k=4, seed=17)
        ),
        preprocessor=PreprocessorConfig(
            LSMConfig.dense(output_dim=16, hidden_layers=1, hidden_units=8, random_state=11),
            training=PreprocessorTrainingConfig(trainable=trainable),
        ),
        epochs=2,
        hidden_layers=2,
        batch_size=8,
        random_state=13,
        device="cpu",
    )
    x = np.random.default_rng(19).normal(size=(24, 4)).astype(np.float32)
    y = np.sin(x[:, :1])
    estimator.fit(x, y)
    expected = estimator.predict(x)
    for generation in (1, 2):
        path = tmp_path / f"geometry-{generation}.pt"
        estimator.save(path)
        estimator = PSANNRegressor.load(path, map_location="cpu")
        np.testing.assert_array_equal(estimator.predict(x), expected)


def test_geometric_builder_with_width_changing_module_matches_explicit_composition():
    from dataclasses import replace
    from psann.architectures import (
        ArchitectureBuildRequest,
        ArchitectureConfig,
        GeometryConfig,
        build_architecture,
    )
    import copy

    preprocessor = torch.nn.Linear(4, 16)
    reference_preprocessor = copy.deepcopy(preprocessor)
    request = ArchitectureBuildRequest(
        architecture=ArchitectureConfig.geometric_sparse(
            geometry=GeometryConfig(shape=(4, 4), k=4, seed=17)
        ),
        hidden_layers=2,
        hidden_units=16,
        input_shape=(4,),
        input_dim=4,
        output_dim=2,
        spatial_shape=None,
        spatial_ndim=None,
        in_channels=None,
        sequence_length=None,
        token_dim=None,
        per_element=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        preprocessor=preprocessor,
        preprocessor_output_dim=16,
        structure_metadata=None,
    )
    torch.manual_seed(23)
    actual = build_architecture(request).model
    torch.manual_seed(23)
    reference = build_architecture(
        replace(
            request,
            input_dim=16,
            input_shape=(16,),
            preprocessor=None,
            preprocessor_output_dim=None,
        )
    ).model
    x = torch.randn(8, 4)
    expected = reference(reference_preprocessor(x))
    result = actual(x)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    result.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(
        preprocessor.weight.grad, reference_preprocessor.weight.grad, rtol=0, atol=0
    )
