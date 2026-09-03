"""Maintained benchmark factories construct only the canonical estimator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from psann import (
    PSANNRegressor,
    ResConvPSANNRegressor,
    ResPSANNRegressor,
    SGRPSANNRegressor,
    WaveResNetRegressor,
)


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


def test_ablation_runner_merges_overrides_and_fits_every_architecture() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from scripts._benchmark_regressor_ablations.data import DatasetBundle
        from scripts._benchmark_regressor_ablations.models import MODELS
        from scripts._benchmark_regressor_ablations.runner import _run_single

        for name, spec in MODELS.items():
            X = np.ones(
                (12, 2, 2) if spec.architecture.kind == "sequence" else (12, 4),
                dtype=np.float32,
            )
            y = X.reshape(len(X), -1).mean(axis=1, keepdims=True)
            dataset = DatasetBundle(
                name=name,
                task="regression",
                kind="tiny",
                X_train=X,
                y_train=y,
                X_test=X[:4],
                y_test=y[:4],
                meta={},
            )
            result = _run_single(
                spec,
                dataset,
                seed=0,
                device="cpu",
                epochs=1,
                batch_size=4,
                lr=1e-3,
                val_fraction=0.25,
                scale_y=False,
            )
            assert result["test_size"] == 4
            assert type(spec.build(**spec.params)) is PSANNRegressor
    finally:
        sys.path.remove(str(scripts))


def test_migrated_benchmark_configs_match_supported_legacy_facades() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from benchmark_hisso_variants import _build_estimator
        from scripts._benchmark_regressor_ablations.models import MODELS

        expected = {
            "res_base": ResPSANNRegressor(
                hidden_layers=4, hidden_units=64, norm="rms"
            ).architecture,
            "res_relu_sigmoid_psann": ResPSANNRegressor(
                hidden_layers=4,
                hidden_units=64,
                norm="rms",
                activation_type="relu_sigmoid_psann",
                activation={"slope_init": 1.0, "clip_max": 1.0},
            ).architecture,
            "res_drop_path": ResPSANNRegressor(
                hidden_layers=4, hidden_units=64, norm="rms", drop_path_max=0.1
            ).architecture,
            "res_no_norm": ResPSANNRegressor(
                hidden_layers=4, hidden_units=64, norm="none"
            ).architecture,
            "wrn_base": WaveResNetRegressor(
                hidden_layers=6, hidden_units=64, norm="rms"
            ).architecture,
            "wrn_no_phase": WaveResNetRegressor(
                hidden_layers=6, hidden_units=64, norm="rms", use_phase_shift=False
            ).architecture,
            "wrn_no_film": WaveResNetRegressor(
                hidden_layers=6, hidden_units=64, norm="rms", use_film=False
            ).architecture,
            "wrn_spec_gate_rfft": WaveResNetRegressor(
                hidden_layers=6,
                hidden_units=64,
                norm="rms",
                use_spectral_gate=True,
                k_fft=64,
                gate_type="rfft",
                gate_groups="depthwise",
                gate_strength=1.0,
            ).architecture,
            "wrn_spec_gate_feats": WaveResNetRegressor(
                hidden_layers=6,
                hidden_units=64,
                norm="rms",
                use_spectral_gate=True,
                k_fft=64,
                gate_type="fourier_features",
                gate_groups="depthwise",
                gate_strength=1.0,
            ).architecture,
            "sgr_base": SGRPSANNRegressor(hidden_layers=3, hidden_units=64).architecture,
            "sgr_no_gate": SGRPSANNRegressor(
                hidden_layers=3, hidden_units=64, use_spectral_gate=False
            ).architecture,
            "sgr_fourier_feats": SGRPSANNRegressor(
                hidden_layers=3, hidden_units=64, gate_type="fourier_features"
            ).architecture,
            "sgr_no_phase": SGRPSANNRegressor(
                hidden_layers=3, hidden_units=64, phase_trainable=False
            ).architecture,
        }
        assert {name: spec.architecture for name, spec in MODELS.items()} == expected
        assert (
            _build_estimator("dense", "cpu", epochs=1, seed=0).architecture
            == ResPSANNRegressor(hidden_layers=2, hidden_units=64).architecture
        )
        assert (
            _build_estimator("conv", "cpu", epochs=1, seed=0).architecture
            == ResConvPSANNRegressor(
                hidden_layers=2, hidden_units=32, conv_channels=32
            ).architecture
        )
    finally:
        sys.path.remove(str(scripts))


def test_hisso_benchmark_runner_fits_dense_and_convolutional_variants() -> None:
    """Exercise the HISSO benchmark's real fit path with its migrated factories."""

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from benchmark_hisso_variants import DatasetSpec, ScheduleSpec, _benchmark_variant

        def build(variant: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
            del seed
            if variant == "dense":
                X = np.ones((8, 3), dtype=np.float32)
            else:
                X = np.ones((8, 1, 3, 3), dtype=np.float32)
            return X, X.reshape(len(X), -1).mean(axis=1, keepdims=True)

        dataset = DatasetSpec(
            name="tiny",
            description="minimal real-consumer fixture",
            variants=("dense", "conv"),
            build_fn=build,
            metadata={},
        )
        schedule = ScheduleSpec(
            name="tiny",
            description="minimal real-consumer schedule",
            batch_episodes=1,
            updates_per_epoch=1,
        )
        for variant in dataset.variants:
            result = _benchmark_variant(
                variant,
                "cpu",
                schedule=schedule,
                epochs=1,
                repeats=1,
                window=2,
                transition_penalty=0.0,
                warmstart_epochs=1,
                base_seed=0,
                dataset=dataset,
            )
            assert result["runs"] == 1
            assert result["feature_shape"] == ([3] if variant == "dense" else [1, 3, 3])
    finally:
        sys.path.remove(str(scripts))
