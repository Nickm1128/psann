"""Bounded real-consumer checks for the canonical preprocessing examples."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from psann import PSANNRegressor

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    path = _EXAMPLES / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_14_canonical_preprocessor_runs_a_bounded_fit() -> None:
    example = _load("14_psann_with_vs_without_lsm.py")
    X, y = example.make_data(n=12, d=4, seed=1)
    estimator = PSANNRegressor(
        preprocessor=example.make_lsm_preprocessor(pretraining_epochs=0),
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=0,
        device="cpu",
    ).fit(X, y)
    assert estimator.predict(X[:2]).shape == (2, 1)


def test_example_21_canonical_preprocessor_runs_bounded_hisso_consumer() -> None:
    example = _load("21_psann_config_benchmark.py")
    prices = example.make_prices(T=30, seed=2)
    row = example.run_config(
        prices,
        seed=0,
        activation_type="psann",
        hidden_layers=1,
        hidden_width=4,
        hisso_window=2,
        trans_cost=1e-3,
        lsm_cfg={"output_dim": 4, "hidden_layers": 1, "hidden_width": 4, "epochs": 0},
        epochs=1,
        n_train=16,
        n_val=7,
    )
    assert row["lsm"] == "dict"
    assert np.isfinite(row["train_time_s"])
