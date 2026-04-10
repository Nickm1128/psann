from __future__ import annotations

import numpy as np
import pytest

from psann import PSANNRegressor, ResConvPSANNRegressor, ResPSANNRegressor
from psann.hisso import HISSOTrainer, hisso_infer_series

pytestmark = pytest.mark.slow


def test_respsann_hisso_uses_shared_options() -> None:
    rng = np.random.default_rng(8)
    X = rng.standard_normal((24, 5)).astype(np.float32)
    y = rng.standard_normal((24, 1)).astype(np.float32)

    est = ResPSANNRegressor(
        hidden_layers=1,
        hidden_units=10,
        epochs=1,
        batch_size=6,
        lr=1e-3,
        random_state=8,
    )

    est.fit(X, y, hisso=True, hisso_window=6, hisso_primary_transform="tanh")

    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert options.primary_transform == "tanh"

    preds = est.predict(X)
    transformed = np.tanh(preds)
    inferred = hisso_infer_series(est, X)
    np.testing.assert_allclose(inferred, transformed, rtol=1e-6, atol=1e-6)


def test_resconv_hisso_preserve_shape_routes_options() -> None:
    rng = np.random.default_rng(9)
    X = rng.standard_normal((16, 2, 4, 4)).astype(np.float32)
    y = rng.standard_normal((16, 1)).astype(np.float32)

    est = ResConvPSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        conv_channels=4,
        epochs=1,
        batch_size=4,
        lr=5e-4,
        random_state=9,
    )

    est.fit(X, y, hisso=True, hisso_window=4, hisso_primary_transform="identity")

    options = getattr(est, "_hisso_options_", None)
    assert options is not None
    assert options.primary_transform == "identity"
    preds = est.predict(X)
    inferred = hisso_infer_series(est, X)
    np.testing.assert_allclose(inferred, preds, rtol=1e-6, atol=1e-6)


def test_resconv_hisso_rejects_per_element_mode() -> None:
    rng = np.random.default_rng(10)
    X = rng.standard_normal((12, 1, 4, 4)).astype(np.float32)
    y = rng.standard_normal((12, 1, 4, 4)).astype(np.float32)

    est = ResConvPSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        conv_channels=4,
        per_element=True,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        random_state=10,
    )

    with pytest.raises(ValueError, match="per_element=False"):
        est.fit(X, y, hisso=True, hisso_window=4)


def test_hisso_fit_without_reward_uses_trainer_history_on_base_regressor() -> None:
    rng = np.random.default_rng(22)
    X = rng.standard_normal((20, 3)).astype(np.float32)
    y = rng.standard_normal((20, 1)).astype(np.float32)

    est = PSANNRegressor(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=5,
        lr=5e-4,
        random_state=22,
    )
    est.fit(X, y, hisso=True, hisso_window=5)

    trainer = getattr(est, "_hisso_trainer_", None)
    assert isinstance(trainer, HISSOTrainer)
    assert trainer.history == est.history_
