from __future__ import annotations

import numpy as np
import torch

from psann.estimators import PSANNRegressor


def test_schema_v1_round_trip_does_not_store_final_module(tmp_path):
    X = np.ones((8, 2), dtype=np.float32)
    estimator = PSANNRegressor(epochs=1, batch_size=4, random_state=0).fit(
        X, np.ones(8, dtype=np.float32)
    )
    path = tmp_path / "regressor.pt"
    estimator.save(str(path))
    payload = torch.load(path, weights_only=False)
    assert payload["schema"] == "psann.regressor"
    assert payload["schema_version"] == 1
    assert "model" not in payload
    loaded = PSANNRegressor.load(str(path))
    np.testing.assert_allclose(loaded.predict(X[:2]), estimator.predict(X[:2]))
