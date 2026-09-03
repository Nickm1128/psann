"""Runtime contract for the GeoSparse-only phase/mixed activation extension."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("torch")

from psann import GeoSparseRegressor, PSANNRegressor
from psann.activations import MixedActivation, PhaseSineParam, ReLUSigmoidPSANN, SineParam
from psann.architectures import (
    ActivationConfig,
    ArchitectureConfig,
    GeometryConfig,
    normalize_activation_config,
    normalize_architecture,
)


def test_geosparse_activation_normalizer_equivalence_and_rejections() -> None:
    cases = [
        (
            {"kind": "respsann", "amp_init": 0.8, "freq_init": 1.2, "damp_init": 0.2},
            ActivationConfig(kind="psann", amplitude_init=0.8, frequency_init=1.2, decay_init=0.2),
        ),
        (
            {"kind": " PHASE_PSANN ", "phase_init": 0.25, "phase_trainable": False},
            ActivationConfig(kind="phase-psann", phase_init=0.25, phase_trainable=False),
        ),
        (
            {
                "kind": "mixed",
                "types": ["phasepsann", "rsp", "relu"],
                "ratios": [0.5, 0.25, 0.25],
                "layout": "contiguous",
                "seed": 7,
                "phase_init": 0.25,
            },
            ActivationConfig(
                kind="mixed",
                activation_types=("phase-psann", "relu-sigmoid-psann", "relu"),
                activation_ratios=(0.5, 0.25, 0.25),
                phase_init=0.25,
                mix_layout="contiguous",
                mix_seed=7,
            ),
        ),
        (
            {
                "kind": "clipped_psann",
                "relu_slope_init": 0.7,
                "slope_learnable": False,
                "clip_at": 2.0,
            },
            ActivationConfig(
                kind="relu-sigmoid-psann", slope_init=0.7, slope_trainable=False, clip_max=2.0
            ),
        ),
    ]
    for raw, expected in cases:
        assert normalize_activation_config(raw) == expected

    tagged = normalize_architecture(
        {
            "kind": "geo_sparse",
            "activation": {
                "kind": "mixed",
                "types": ["psann", "relu"],
                "ratios": [0.5, 0.5],
            },
            "residual": {},
            "geometry": {"shape": [2, 2]},
        }
    )
    assert tagged.activation == ActivationConfig(
        kind="mixed", activation_types=("psann", "relu"), activation_ratios=(0.5, 0.5)
    )
    assert (
        normalize_activation_config(
            {"kind": "mixed", "activation_types": ["psann", "relu"]}
        ).activation_ratios
        is None
    )

    with pytest.raises(ValueError, match="conflicting"):
        normalize_activation_config({"amp_init": 0.8, "amplitude_init": 0.8})
    with pytest.raises(ValueError, match="unknown"):
        normalize_activation_config({"unknown": 1})
    with pytest.raises(TypeError, match="trainable"):
        normalize_activation_config({"trainable": 1})
    with pytest.raises(TypeError, match="mix_seed"):
        normalize_activation_config(
            {"kind": "mixed", "activation_types": ["psann"], "mix_seed": True}
        )
    with pytest.raises(ValueError, match="phase"):
        ActivationConfig(kind="psann", phase_init=0.25)
    with pytest.raises(ValueError, match="mixed"):
        ArchitectureConfig.dense(
            activation=ActivationConfig(kind="mixed", activation_types=("psann",))
        )


@pytest.mark.parametrize(
    ("raw", "path"),
    [
        ({"kind": "phase-psann", "phase_init": "0.2"}, "activation.phase_init"),
        ({"kind": "phase-psann", "phase_init": True}, "activation.phase_init"),
        (
            {"kind": "mixed", "activation_types": ["psann"], "ratio_sum_tol": "0.1"},
            "activation.ratio_sum_tol",
        ),
        (
            {"kind": "mixed", "activation_types": ["psann"], "ratio_sum_tol": True},
            "activation.ratio_sum_tol",
        ),
        ({"kind": "mixed", "activation_types": "psann"}, "activation.activation_types"),
        ({"kind": "mixed", "activation_types": [1]}, "activation.activation_types[0]"),
        (
            {"kind": "mixed", "activation_types": ["psann"], "activation_ratios": "1"},
            "activation.activation_ratios",
        ),
        (
            {"kind": "mixed", "activation_types": ["psann"], "activation_ratios": [True]},
            "activation.activation_ratios[0]",
        ),
    ],
)
def test_phase_and_mixed_wrong_types_report_full_activation_paths(
    raw: dict[str, object], path: str
) -> None:
    with pytest.raises(TypeError, match=path.replace("[", r"\[").replace("]", r"\]")):
        normalize_activation_config(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"amp_init": 1.0, "amplitude_init": 1.0},
        {"freq_init": 1.0, "frequency_init": 1.0},
        {"damp_init": 0.1, "decay_init": 0.1},
        {"damping_init": 0.1, "decay_init": 0.1},
        {"trainable": True, "learnable": ()},
        {"kind": "mixed", "types": ["psann"], "activation_types": ["psann"]},
        {"kind": "mixed", "ratios": [1.0], "activation_ratios": [1.0]},
        {"kind": "mixed", "layout": "random", "mix_layout": "random"},
        {"kind": "mixed", "seed": 1, "mix_seed": 1},
        {"relu_slope_init": 1.0, "slope_init": 1.0},
        {"slope_learnable": True, "slope_trainable": True},
        {"clip_at": 1.0, "clip_max": 1.0},
        {"amp_bounds": (0.0, 1.0), "bounds": {"amplitude": (0.0, 1.0)}},
        {"freq_bounds": (0.0, 1.0), "bounds": {"frequency": (0.0, 1.0)}},
        {"damp_bounds": (0.0, 1.0), "bounds": {"decay": (0.0, 1.0)}},
    ],
)
def test_activation_alias_conflicts_are_never_silently_preferred(raw: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="conflicting"):
        normalize_activation_config(raw)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda activation: ArchitectureConfig.dense(activation=activation),
        lambda activation: ArchitectureConfig.convolutional(activation=activation),
        lambda activation: ArchitectureConfig.for_wave(activation=activation),
        lambda activation: ArchitectureConfig.for_sequence(activation=activation),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        ActivationConfig(kind="phase-psann", phase_init=0.2),
        ActivationConfig(kind="mixed", activation_types=("psann", "relu")),
    ],
)
def test_phase_and_mixed_are_rejected_for_every_non_geosparse_architecture(
    constructor: object, activation: ActivationConfig
) -> None:
    with pytest.raises(ValueError, match="geometric-sparse"):
        constructor(activation)  # type: ignore[operator]


def test_geosparse_facade_activation_adapter_retains_typed_policies_and_rejects_bad_values() -> (
    None
):
    from sklearn.base import clone

    typed = ActivationConfig(kind="phase-psann", phase_init=0.2, phase_trainable=False)
    with pytest.warns(DeprecationWarning):
        facade = GeoSparseRegressor(activation=typed, shape=(2, 2), k=2)
    assert facade.architecture.activation is typed
    assert clone(facade).architecture.activation == typed

    tagged = {"kind": "mixed", "types": ["psann", "relu"], "ratios": [0.5, 0.5]}
    with pytest.warns(DeprecationWarning):
        omitted_type = GeoSparseRegressor(activation=tagged, shape=(2, 2), k=2)
    assert omitted_type.architecture.activation.kind == "mixed"
    with pytest.warns(DeprecationWarning):
        matching_type = GeoSparseRegressor(
            activation_type="mixed", activation=tagged, shape=(2, 2), k=2
        )
    assert matching_type.architecture.activation.kind == "mixed"
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="conflicts"):
        GeoSparseRegressor(activation_type="relu", activation=tagged, shape=(2, 2), k=2)
    with pytest.warns(DeprecationWarning), pytest.raises(TypeError, match="activation"):
        GeoSparseRegressor(activation=7, shape=(2, 2), k=2)


def test_geosparse_facade_fully_tagged_mixed_payload_reaches_fit_and_predict() -> None:
    X = np.ones((8, 4), dtype=np.float32)
    y = X.mean(axis=1)
    with pytest.warns(DeprecationWarning):
        facade = GeoSparseRegressor(
            activation_type="mixed",
            activation={
                "kind": "mixed",
                "activation_types": ["phase-psann", "relu"],
                "activation_ratios": [0.5, 0.5],
                "phase_init": 0.2,
            },
            shape=(2, 2),
            k=2,
            hidden_layers=1,
            epochs=1,
            batch_size=4,
            random_state=0,
        )
    facade.fit(X, y)
    assert facade.predict(X[:2]).shape == (2,)


@pytest.mark.parametrize(
    ("activation_type", "activation", "expected_type"),
    [
        (
            "sine",
            {"amp_init": 0.8, "freq_init": 1.2, "damp_init": 0.2, "trainable": False},
            SineParam,
        ),
        (
            "phasepsann",
            {
                "amp_init": 0.8,
                "freq_init": 1.2,
                "damp_init": 0.2,
                "trainable": False,
                "phase_init": 0.25,
                "phase_trainable": False,
            },
            PhaseSineParam,
        ),
        (
            "mixed",
            {
                "types": ["phasepsann", "relu"],
                "ratios": [0.5, 0.5],
                "layout": "contiguous",
                "seed": 3,
                "phase_init": 0.25,
                "phase_trainable": False,
            },
            MixedActivation,
        ),
        (
            "rsp",
            {
                "amp_init": 0.8,
                "freq_init": 1.2,
                "damp_init": 0.2,
                "relu_slope_init": 0.7,
                "slope_learnable": False,
                "clip_at": 2.0,
            },
            ReLUSigmoidPSANN,
        ),
    ],
)
def test_geosparse_facade_matches_phase2_runtime(
    activation_type: str, activation: dict[str, object], expected_type: type[object]
) -> None:
    """The facade reaches the same legacy backbone behavior, not just its constructor."""

    from psann._sklearn.geosparse import GeoSparseRegressor as Phase2GeoSparseRegressor

    X = (np.arange(32, dtype=np.float32).reshape(8, 4) / 10.0).astype(np.float32)
    y = X.sum(axis=1)
    common = {
        "shape": (2, 2),
        "hidden_layers": 1,
        "k": 2,
        "epochs": 1,
        "batch_size": 4,
        "random_state": 0,
        "early_stopping": False,
        "activation_type": activation_type,
        "activation": activation,
    }
    legacy_common = dict(common)
    if activation_type == "mixed":
        legacy_common["activation"] = {
            **activation,
            "types": ["phase_psann", "relu"],
        }
    legacy = Phase2GeoSparseRegressor(**legacy_common).fit(X, y)
    facade = GeoSparseRegressor(**common).fit(X, y)
    np.testing.assert_allclose(facade.predict(X[:2]), legacy.predict(X[:2]), rtol=1e-6)

    act = facade.model_.blocks[0].act
    assert isinstance(act, expected_type)
    if activation_type == "sine":
        assert not act._A.requires_grad
    elif activation_type == "phasepsann":
        assert not act._A.requires_grad
        assert not act._phi.requires_grad
    elif activation_type == "mixed":
        assert act.layout == "contiguous"
        assert set(act.acts) == {"phase_psann", "relu"}
        assert not act.acts["phase_psann"]._phi.requires_grad
    else:
        np.testing.assert_allclose(
            torch.nn.functional.softplus(act._slope.detach()).cpu().numpy(), 0.7, rtol=1e-5
        )
        assert not act._slope.requires_grad
        assert act.clip_max == pytest.approx(2.0)


def test_geosparse_phase_and_mixed_clone_nested_updates_and_v1_round_trip(tmp_path: Path) -> None:
    from sklearn.base import clone

    X = np.ones((8, 4), dtype=np.float32)
    y = X.mean(axis=1)
    architecture = ArchitectureConfig.geometric_sparse(
        activation=ActivationConfig(
            kind="mixed",
            activation_types=("phase-psann", "relu-sigmoid-psann"),
            activation_ratios=(0.5, 0.5),
            phase_init=0.2,
            slope_init=0.7,
            clip_max=2.0,
        ),
        geometry=GeometryConfig(shape=(2, 2), k=2),
    )
    estimator = PSANNRegressor(
        architecture=architecture, hidden_layers=1, epochs=1, batch_size=4, random_state=0
    )
    assert clone(estimator).architecture == architecture
    assert estimator.get_params(deep=True)["architecture__activation__phase_init"] == 0.2
    estimator.set_params(
        architecture__activation__phase_init=0.3,
        architecture__activation__mix_layout="contiguous",
    )
    assert estimator.architecture.activation.phase_init == pytest.approx(0.3)
    assert estimator.architecture.activation.mix_layout == "contiguous"
    estimator.fit(X, y)
    path = tmp_path / "geosparse-mixed.pt"
    estimator.save(str(path))
    loaded = PSANNRegressor.load(str(path))
    assert loaded.architecture == estimator.architecture
    np.testing.assert_allclose(loaded.predict(X[:2]), estimator.predict(X[:2]), rtol=1e-6)

    phase = PSANNRegressor(
        architecture=ArchitectureConfig.geometric_sparse(
            activation=ActivationConfig(kind="phase-psann", phase_init=0.2),
            geometry=GeometryConfig(shape=(2, 2), k=2),
        ),
        hidden_layers=1,
        epochs=1,
        batch_size=4,
        random_state=0,
    )
    assert clone(phase).architecture == phase.architecture
    assert phase.get_params(deep=True)["architecture__activation__phase_init"] == 0.2
    phase.set_params(architecture__activation__phase_init=0.3)
    phase.fit(X, y)
    phase_path = tmp_path / "geosparse-phase.pt"
    phase.save(str(phase_path))
    loaded_phase = PSANNRegressor.load(str(phase_path))
    assert loaded_phase.architecture == phase.architecture
    np.testing.assert_allclose(loaded_phase.predict(X[:2]), phase.predict(X[:2]), rtol=1e-6)


def test_geosparse_benchmark_json_uses_shared_normalizer_and_real_consumer() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        from scripts._run_geosparse_vs_relu_benchmarks.bench import fit_with_timing
        from scripts._run_geosparse_vs_relu_benchmarks.models import (
            build_geosparse_estimator,
            count_geosparse_params,
        )

        payloads = [
            {
                "kind": "mixed",
                "activation_types": ["phase-psann", "relu"],
                "activation_ratios": [0.5, 0.5],
            },
            {"types": ["phasepsann", "relu"], "ratios": [0.5, 0.5], "layout": "contiguous"},
        ]
        X = np.ones((8, 4), dtype=np.float32)
        y = X.mean(axis=1)
        for payload in payloads:
            assert (
                count_geosparse_params(
                    4,
                    1,
                    depth=1,
                    k=2,
                    shape=(2, 2),
                    activation_type="mixed",
                    activation_config_json=json.dumps(payload),
                )
                > 0
            )

            def factory(
                epochs: int, *, activation_payload: dict[str, object] = payload
            ) -> PSANNRegressor:
                return build_geosparse_estimator(
                    input_dim=4,
                    shape=(2, 2),
                    geo_depth=1,
                    geo_k=2,
                    activation_type="mixed",
                    activation_config=activation_payload,
                    amp=False,
                    amp_dtype="float32",
                    compile=False,
                    compile_backend="inductor",
                    compile_mode="default",
                    compile_fullgraph=False,
                    compile_dynamic=False,
                    device="cpu",
                    seed=0,
                    epochs=epochs,
                    batch_size=4,
                    lr=1e-3,
                )

            result = fit_with_timing(
                factory,
                X,
                y,
                X[:2],
                y[:2],
                batch_size=4,
                target_steps=1,
                progress_every_steps=1,
                timing_warmup_epochs=0,
            )
            assert result["model"].predict(X[:2]).shape == (2,)
    finally:
        sys.path.remove(str(scripts))
