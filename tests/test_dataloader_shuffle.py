from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from psann import PSANNRegressor
from psann.estimators import _fit_utils as fit_utils


def _toy_xy(seed: int = 0, batch: int = 12, features: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((batch, features)).astype(np.float32)
    y = rng.standard_normal((batch, 1)).astype(np.float32)
    return X, y


def test_fit_stateless_uses_shuffled_dataloader(monkeypatch) -> None:
    X, y = _toy_xy()
    estimator = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        random_state=1,
        stateful=False,
    )

    observed_shuffle: list[bool] = []

    def _capture_dataloader(dataset, *, batch_size, shuffle, num_workers):
        observed_shuffle.append(bool(shuffle))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    monkeypatch.setattr(fit_utils, "DataLoader", _capture_dataloader)

    estimator.fit(X, y, verbose=0)
    assert observed_shuffle and observed_shuffle[0] is True


def test_fit_stateful_epoch_reset_disables_shuffle(monkeypatch) -> None:
    X, y = _toy_xy(seed=1)
    estimator = PSANNRegressor(
        hidden_layers=1,
        hidden_units=8,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        random_state=2,
        stateful=True,
        state_reset="epoch",
    )

    observed_shuffle: list[bool] = []

    def _capture_dataloader(dataset, *, batch_size, shuffle, num_workers):
        observed_shuffle.append(bool(shuffle))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    monkeypatch.setattr(fit_utils, "DataLoader", _capture_dataloader)

    estimator.fit(X, y, verbose=0)
    assert observed_shuffle and observed_shuffle[0] is False
