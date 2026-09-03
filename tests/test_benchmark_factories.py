"""Maintained benchmark factories construct only the canonical estimator."""

from __future__ import annotations

import sys
from pathlib import Path

from psann import PSANNRegressor


def test_benchmark_factories_construct_canonical_architectures() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from benchmark_hisso_variants import _build_estimator
        from scripts._benchmark_regressor_ablations.models import MODELS
        from scripts._run_geosparse_vs_relu_benchmarks.models import build_geosparse_estimator

        assert type(_build_estimator("dense", "cpu", epochs=1, seed=0)) is PSANNRegressor
        assert type(_build_estimator("conv", "cpu", epochs=1, seed=0)) is PSANNRegressor
        assert all(type(spec.build()) is PSANNRegressor for spec in MODELS.values())
        assert (
            type(
                build_geosparse_estimator(
                    input_dim=4,
                    shape=(2, 2),
                    geo_depth=1,
                    geo_k=2,
                    activation_type="relu",
                    activation_config=None,
                    amp=False,
                    amp_dtype="float32",
                    compile=False,
                    compile_backend="inductor",
                    compile_mode="default",
                    compile_fullgraph=False,
                    compile_dynamic=False,
                    device="cpu",
                    seed=0,
                    epochs=1,
                    batch_size=2,
                    lr=1e-3,
                )
            )
            is PSANNRegressor
        )
    finally:
        sys.path.remove(str(scripts))
