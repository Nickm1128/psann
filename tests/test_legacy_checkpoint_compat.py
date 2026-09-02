from __future__ import annotations

import inspect
import subprocess
import sys

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
        _normalise_legacy_params(ResConvPSANNRegressor, {"attention": {"num_heads": 2}})
    with pytest.raises(ValueError, match="WaveResNetRegressor.*amp"):
        _normalise_legacy_params(WaveResNetRegressor, {"amp": True})
    with pytest.raises(ValueError, match="PSANNRegressor.*unknown"):
        _normalise_legacy_params(PSANNRegressor, {"unknown": 1})


def test_geosparse_legacy_keys_are_renamed_and_conflicts_fail() -> None:
    params = _normalise_legacy_params(
        GeoSparseRegressor,
        {"geo_shape": (2, 3), "geo_k": 4, "geo_compute_mode": "masked"},
    )
    assert params == {"shape": (2, 3), "k": 4, "compute_mode": "masked"}

    with pytest.raises(ValueError, match="conflicting 'geo_k' and 'k'"):
        _normalise_legacy_params(GeoSparseRegressor, {"geo_k": 4, "k": 5})
