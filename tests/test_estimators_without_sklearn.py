from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from psann._sklearn.fallback import BaseEstimator, ClassifierMixin, r2_score

ROOT = Path(__file__).resolve().parents[1]


def test_fallback_parameter_contract_is_covered_in_process():
    class Empty(BaseEstimator):
        pass

    class Variadic(BaseEstimator):
        def __init__(self, *values):
            self.values = values

    class Leaf(BaseEstimator):
        def __init__(self, value=1):
            self.value = value

    class Parent(BaseEstimator):
        def __init__(self, child=None, **metadata):
            self.child = Leaf() if child is None else child
            self.metadata = metadata

    assert Empty().get_params() == {}
    with pytest.raises(RuntimeError, match="must declare constructor parameters explicitly"):
        Variadic().get_params()

    parent = Parent(Leaf(2), ignored=True)
    assert parent.get_params(deep=False) == {"child": parent.child}
    assert parent.get_params()["child__value"] == 2
    assert parent.set_params() is parent
    parent.set_params(child__value=3)
    assert parent.child.value == 3
    parent.set_params(child=Leaf(4), child__value=5)
    assert parent.child.value == 5
    with pytest.raises(ValueError, match="Invalid parameter 'unknown'"):
        parent.set_params(unknown=1)

    assert ClassifierMixin._estimator_type == "classifier"
    assert r2_score([1.0, 2.0], [1.0, 2.0]) == 1.0
    assert np.isnan(r2_score([1.0, 1.0], [0.0, 0.0]))


def test_estimator_fallback_matches_sklearn_parameter_and_serialization_contract(tmp_path: Path):
    program = textwrap.dedent("""
        import builtins
        import inspect
        import tempfile
        import warnings
        from pathlib import Path

        import numpy as np

        real_import = builtins.__import__

        def block_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 0 and (name == "sklearn" or name.startswith("sklearn.")):
                raise ImportError("scikit-learn blocked for fallback contract test")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = block_sklearn

        from psann import ResConvPSANNRegressor, WaveResNetRegressor
        from psann._sklearn.shared import BaseEstimator

        assert BaseEstimator.__module__ == "psann._sklearn.fallback"

        class Leaf(BaseEstimator):
            def __init__(self, value=1):
                self.value = value

        class Parent(BaseEstimator):
            def __init__(self, child=None):
                self.child = Leaf() if child is None else child

        parent = Parent(Leaf(2))
        assert parent.get_params()["child__value"] == 2
        parent.set_params(child__value=3)
        assert parent.child.value == 3
        parent.set_params(child=Leaf(4), child__value=5)
        assert parent.child.value == 5
        try:
            parent.set_params(unknown=1)
        except ValueError as exc:
            assert "Invalid parameter 'unknown'" in str(exc)
        else:
            raise AssertionError("Fallback set_params accepted an unknown parameter")

        rng = np.random.default_rng(2026)
        conv_inputs = rng.normal(size=(8, 1, 4, 4)).astype(np.float32)
        conv_targets = conv_inputs.mean(axis=(1, 2, 3)).astype(np.float32)
        dense_inputs = rng.normal(size=(8, 4)).astype(np.float32)
        dense_targets = dense_inputs.mean(axis=1).astype(np.float32)

        estimators = (
            (
                ResConvPSANNRegressor(
                    hidden_layers=1,
                    hidden_units=4,
                    conv_channels=4,
                    epochs=1,
                    batch_size=4,
                    random_state=7,
                ),
                conv_inputs,
                conv_targets,
            ),
            (
                WaveResNetRegressor(
                    hidden_layers=1,
                    hidden_units=4,
                    epochs=1,
                    batch_size=4,
                    w0_warmup_epochs=0,
                    random_state=11,
                ),
                dense_inputs,
                dense_targets,
            ),
        )

        warnings.simplefilter("ignore")
        with tempfile.TemporaryDirectory() as directory:
            for index, (estimator, inputs, targets) in enumerate(estimators):
                constructor_names = set(inspect.signature(estimator.__class__.__init__).parameters)
                constructor_names.discard("self")
                assert set(estimator.get_params()) <= constructor_names
                estimator.fit(inputs, targets, verbose=0)
                predictions = estimator.predict(inputs)
                checkpoint = Path(directory) / f"fallback-{index}.pt"
                estimator.save(str(checkpoint))
                restored = estimator.__class__.load(str(checkpoint))
                np.testing.assert_allclose(
                    restored.predict(inputs),
                    predictions,
                    rtol=1e-6,
                    atol=1e-6,
                )
        """)
    env = os.environ.copy()
    source_paths = [str(ROOT / "src"), str(ROOT)]
    if env.get("PYTHONPATH"):
        source_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(source_paths)

    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
