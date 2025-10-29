import pytest

pytest.importorskip("torch")

from psann import (
    PSANNRegressor,
    ResConvPSANNRegressor,
    ResPSANNRegressor,
    WaveResNetRegressor,
)


@pytest.mark.parametrize(
    ("estimator_cls", "kwargs"),
    [
        (PSANNRegressor, {"hidden_layers": 1, "hidden_units": 8, "epochs": 1}),
        (ResPSANNRegressor, {"hidden_layers": 2, "hidden_units": 8, "epochs": 1}),
        (
            ResConvPSANNRegressor,
            {"hidden_layers": 2, "hidden_units": 8, "epochs": 1, "conv_kernel_size": 3},
        ),
        (WaveResNetRegressor, {"hidden_layers": 2, "hidden_units": 16, "epochs": 1}),
    ],
)
def test_public_estimators_instantiate(estimator_cls, kwargs):
    estimator = estimator_cls(**kwargs)
    params = estimator.get_params()
    assert isinstance(params, dict)
    assert estimator.__class__.__name__ in {"PSANNRegressor", "ResPSANNRegressor", "ResConvPSANNRegressor", "WaveResNetRegressor"}
    assert params["hidden_layers"] == kwargs["hidden_layers"]
