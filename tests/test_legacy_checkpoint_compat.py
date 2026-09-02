from __future__ import annotations

import inspect
import subprocess
import sys

import numpy as np
import pytest

from psann import (
    GeoSparseRegressor,
    PSANNRegressor,
    ResConvPSANNRegressor,
    ResPSANNRegressor,
    SGRPSANNRegressor,
    WaveResNetRegressor,
)
from psann._sklearn.serialization import _normalise_legacy_params
from psann.attention import AttentionConfig

ESTIMATOR_CLASSES = (
    PSANNRegressor,
    ResPSANNRegressor,
    ResConvPSANNRegressor,
    WaveResNetRegressor,
    SGRPSANNRegressor,
    GeoSparseRegressor,
)


def _constructor_names(cls: type) -> set[str]:
    return {
        name
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self"
        and parameter.kind not in {inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL}
    }


@pytest.mark.parametrize("estimator_cls", ESTIMATOR_CLASSES)
def test_constructor_parameters_match_get_params_and_sklearn_clone(estimator_cls) -> None:
    estimator = estimator_cls()
    assert set(estimator.get_params(deep=False)) == _constructor_names(estimator_cls)

    from sklearn.base import clone

    cloned = clone(estimator)
    assert cloned.__class__ is estimator_cls
    assert cloned.get_params(deep=False) == estimator.get_params(deep=False)


def test_no_sklearn_fallback_uses_constructor_parameters_in_subprocess() -> None:
    code = r"""
import builtins
import inspect

original_import = builtins.__import__
def block_sklearn(name, *args, **kwargs):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ImportError("blocked by characterization test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_sklearn

from psann import (
    GeoSparseRegressor, PSANNRegressor, ResConvPSANNRegressor,
    ResPSANNRegressor, SGRPSANNRegressor, WaveResNetRegressor,
)

for cls in (
    PSANNRegressor, ResPSANNRegressor, ResConvPSANNRegressor,
    WaveResNetRegressor, SGRPSANNRegressor, GeoSparseRegressor,
):
    expected = {
        name for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and parameter.kind not in {
            inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL
        }
    }
    estimator = cls()
    assert set(estimator.get_params(deep=False)) == expected, cls.__name__
    estimator.set_params(hidden_layers=estimator.hidden_layers + 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("estimator_cls", "drift"),
    [
        (ResPSANNRegressor, {"amp": False, "context_builder": None}),
        (ResConvPSANNRegressor, {"amp": False, "attention": None}),
        (WaveResNetRegressor, {"compile": False}),
        (SGRPSANNRegressor, {"amp": False, "context_builder_params": {}}),
        (GeoSparseRegressor, {"context_builder": None}),
    ],
)
def test_default_legacy_drift_is_filtered_only_for_affected_classes(estimator_cls, drift) -> None:
    params = _normalise_legacy_params(estimator_cls, drift)
    assert params == {}


def test_nondefault_or_unknown_legacy_drift_is_not_silently_discarded() -> None:
    with pytest.raises(ValueError, match="ResConvPSANNRegressor.*attention"):
        _normalise_legacy_params(ResConvPSANNRegressor, {"attention": {"kind": "mha"}})
    with pytest.raises(ValueError, match="WaveResNetRegressor.*amp"):
        _normalise_legacy_params(WaveResNetRegressor, {"amp": True})
    with pytest.raises(ValueError, match="PSANNRegressor.*unknown"):
        _normalise_legacy_params(PSANNRegressor, {"unknown": 1})


@pytest.mark.parametrize(
    "legacy_attention",
    [
        AttentionConfig(),
        {"kind": "none", "num_heads": 8, "dropout": 0.2},
    ],
)
def test_resconv_filters_phase1_disabled_attention_fallback(legacy_attention) -> None:
    # Phase 1's no-sklearn BaseEstimator fallback exposed this inherited default
    # as AttentionConfig(kind="none", ...), not as the simplified None literal.
    assert _normalise_legacy_params(ResConvPSANNRegressor, {"attention": legacy_attention}) == {}


@pytest.mark.parametrize(
    "legacy_attention",
    [AttentionConfig(kind="mha"), {"kind": "mha", "num_heads": 2}],
)
def test_resconv_rejects_enabled_legacy_attention(legacy_attention) -> None:
    with pytest.raises(ValueError, match="ResConvPSANNRegressor.*attention"):
        _normalise_legacy_params(ResConvPSANNRegressor, {"attention": legacy_attention})


def test_clone_preserves_nondefault_mutable_constructor_values() -> None:
    from sklearn.base import clone

    offsets = [(0, 1), (1, 0)]
    context_params = {"frequencies": 2, "include_cos": False}
    activation = {"amp_init": 0.75, "freq_init": 1.25}
    estimators = (
        GeoSparseRegressor(offsets=offsets),
        PSANNRegressor(
            activation=activation, context_builder="cosine", context_builder_params=context_params
        ),
        WaveResNetRegressor(
            activation=activation, context_builder="cosine", context_builder_params=context_params
        ),
    )
    for estimator in estimators:
        cloned = clone(estimator)
        assert cloned.get_params(deep=False) == estimator.get_params(deep=False)


def _roundtrip_case(estimator_cls):
    rng = np.random.default_rng(17)
    if estimator_cls is ResConvPSANNRegressor:
        X = rng.standard_normal((12, 1, 3, 3)).astype(np.float32)
        y = X.mean(axis=(1, 2, 3)).astype(np.float32)
        kwargs = {"conv_channels": 4}
    elif estimator_cls is SGRPSANNRegressor:
        X = rng.standard_normal((12, 3, 2)).astype(np.float32)
        y = X.mean(axis=(1, 2)).astype(np.float32)
        kwargs = {}
    elif estimator_cls is GeoSparseRegressor:
        X = rng.standard_normal((12, 4)).astype(np.float32)
        y = X.mean(axis=1).astype(np.float32)
        kwargs = {"shape": (2, 2), "k": 2}
    else:
        X = rng.standard_normal((12, 4)).astype(np.float32)
        y = X.mean(axis=1).astype(np.float32)
        kwargs = {}
    return X, y, kwargs


@pytest.mark.parametrize("estimator_cls", ESTIMATOR_CLASSES)
def test_cpu_save_load_roundtrip_for_every_current_estimator(tmp_path, estimator_cls) -> None:
    X, y, kwargs = _roundtrip_case(estimator_cls)
    estimator = estimator_cls(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=17,
        device="cpu",
        **kwargs,
    ).fit(X[:8], y[:8], verbose=0)
    expected = estimator.predict(X[8:])
    checkpoint = tmp_path / f"{estimator_cls.__name__}.pt"
    estimator.save(str(checkpoint))
    actual = estimator_cls.load(str(checkpoint), map_location="cpu").predict(X[8:])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_no_sklearn_save_load_roundtrips_every_current_estimator() -> None:
    code = r"""
import builtins
import os
import tempfile
import numpy as np

original_import = builtins.__import__
def block_sklearn(name, *args, **kwargs):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ImportError("blocked by characterization test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = block_sklearn

from psann import GeoSparseRegressor, PSANNRegressor, ResConvPSANNRegressor, ResPSANNRegressor, SGRPSANNRegressor, WaveResNetRegressor
rng = np.random.default_rng(29)
for cls in (PSANNRegressor, ResPSANNRegressor, ResConvPSANNRegressor, WaveResNetRegressor, SGRPSANNRegressor, GeoSparseRegressor):
    if cls is ResConvPSANNRegressor:
        X = rng.standard_normal((12, 1, 3, 3)).astype(np.float32); y = X.mean(axis=(1, 2, 3)); extra = {"conv_channels": 4}
    elif cls is SGRPSANNRegressor:
        X = rng.standard_normal((12, 3, 2)).astype(np.float32); y = X.mean(axis=(1, 2)); extra = {}
    elif cls is GeoSparseRegressor:
        X = rng.standard_normal((12, 4)).astype(np.float32); y = X.mean(axis=1); extra = {"shape": (2, 2), "k": 2}
    else:
        X = rng.standard_normal((12, 4)).astype(np.float32); y = X.mean(axis=1); extra = {}
    model = cls(hidden_layers=1, hidden_units=4, epochs=1, batch_size=4, random_state=29, device="cpu", **extra).fit(X[:8], y[:8], verbose=0)
    descriptor, path = tempfile.mkstemp(suffix=".pt")
    os.close(descriptor)
    try:
        model.save(path)
        np.testing.assert_allclose(cls.load(path).predict(X[8:]), model.predict(X[8:]), rtol=1e-6, atol=1e-6)
    finally:
        os.unlink(path)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_current_attention_state_lsm_and_hisso_capability_matrix() -> None:
    """Pin legacy behavior that Phase 3 must replace with canonical validation."""
    from psann import StateConfig

    attention = AttentionConfig(kind="mha")
    state = StateConfig()

    # Supported constructor paths retain enabled attention. ResConv intentionally
    # has no public attention argument; its legacy payload-only fallback is covered
    # above instead.
    for cls in (PSANNRegressor, ResPSANNRegressor, WaveResNetRegressor):
        assert cls(attention=attention).attention.is_enabled()
    with pytest.raises(TypeError, match="attention"):
        ResConvPSANNRegressor(attention=attention)

    # SGR and GeoSparse document ignored attention; SGR emits its warning at model
    # build while GeoSparse reports it directly at construction.
    assert SGRPSANNRegressor(attention=attention).attention.is_enabled()
    with pytest.warns(RuntimeWarning, match="ignores attention"):
        GeoSparseRegressor(attention=attention)

    # Dense retains state. The remaining legacy variants either normalize it away at
    # construction or retain it only until their builder emits its documented ignore.
    assert PSANNRegressor(stateful=True, state=state).stateful is True
    for cls in (WaveResNetRegressor, SGRPSANNRegressor):
        with pytest.warns(RuntimeWarning, match="does not support stateful"):
            assert cls(stateful=True, state=state).stateful is False
    assert ResPSANNRegressor(stateful=True, state=state).stateful is True
    assert ResConvPSANNRegressor(stateful=True, state=state).stateful is True
    assert GeoSparseRegressor(stateful=True, state=state).stateful is True

    # Every legacy estimator still exposes the HISSO fit route. LSM is supported by
    # all except SGR, which both warns and clears the ignored value.
    for cls in ESTIMATOR_CLASSES:
        fit_parameters = inspect.signature(cls.fit).parameters
        assert "hisso" in fit_parameters or "kwargs" in fit_parameters
    sentinel_lsm = object()
    for cls in (
        PSANNRegressor,
        ResPSANNRegressor,
        ResConvPSANNRegressor,
        WaveResNetRegressor,
        GeoSparseRegressor,
    ):
        assert cls(lsm=sentinel_lsm).lsm is sentinel_lsm
    with pytest.warns(RuntimeWarning, match="does not support LSM"):
        assert SGRPSANNRegressor(lsm=sentinel_lsm).lsm is None


@pytest.mark.parametrize(
    ("estimator_cls", "parameter", "value", "warning"),
    [
        (ResPSANNRegressor, "state", "state", "does not currently support stateful"),
        (SGRPSANNRegressor, "attention", "attention", "ignores attention"),
        (GeoSparseRegressor, "state", "state", "does not support stateful"),
    ],
)
def test_ignored_capability_cells_warn_when_the_legacy_builder_runs(
    estimator_cls, parameter, value, warning
) -> None:
    """Exercise the build-time ignored cells recorded in the capability matrix."""
    from psann import StateConfig

    X, y, kwargs = _roundtrip_case(estimator_cls)
    config = StateConfig() if value == "state" else AttentionConfig(kind="mha")
    if parameter == "state":
        kwargs.update({"stateful": True, "state": config})
    else:
        kwargs[parameter] = config
    estimator = estimator_cls(
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        batch_size=4,
        random_state=43,
        device="cpu",
        **kwargs,
    )
    with pytest.warns(RuntimeWarning, match=warning):
        estimator.fit(X[:8], y[:8], verbose=0)


def test_geosparse_legacy_keys_are_renamed_and_conflicts_fail() -> None:
    params = _normalise_legacy_params(
        GeoSparseRegressor,
        {"geo_shape": (2, 3), "geo_k": 4, "geo_compute_mode": "masked"},
    )
    assert params == {"shape": (2, 3), "k": 4, "compute_mode": "masked"}

    with pytest.raises(ValueError, match="conflicting 'geo_k' and 'k'"):
        _normalise_legacy_params(GeoSparseRegressor, {"geo_k": 4, "k": 5})
