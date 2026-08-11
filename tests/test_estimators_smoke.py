import numpy as np
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
    assert estimator.__class__.__name__ in {
        "PSANNRegressor",
        "ResPSANNRegressor",
        "ResConvPSANNRegressor",
        "WaveResNetRegressor",
    }
    assert params["hidden_layers"] == kwargs["hidden_layers"]


@pytest.mark.parametrize(
    "estimator_cls",
    [PSANNRegressor, ResPSANNRegressor],
)
def test_core_regressors_fit_predict_with_sigmoid(estimator_cls):
    rs = np.random.RandomState(3)
    X = rs.randn(48, 3).astype(np.float32)
    y = (X[:, :1] - 0.25 * X[:, 1:2]).astype(np.float32)

    estimator = estimator_cls(
        hidden_layers=1,
        hidden_units=8,
        activation_type="sigmoid",
        activation={"slope_init": 1.0, "slope_bounds": (1e-3, 5.0)},
        epochs=2,
        batch_size=16,
        random_state=0,
    )
    estimator.fit(X, y, verbose=0)
    preds = estimator.predict(X[:6])
    assert preds.shape == (6, 1)
