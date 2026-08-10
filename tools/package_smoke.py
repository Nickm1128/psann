#!/usr/bin/env python
"""Install built wheels into a temporary environment and smoke-test imports.

By default pip resolves wheel dependencies, which is the mode used in CI. Local
contributors can reuse already installed NumPy/Torch dependencies:

  python tools/package_smoke.py --system-site-packages --no-deps
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

CORE_SMOKE = """
import importlib
import importlib.metadata
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import psann
from psann import (
    DataSchema,
    ExplainerConfig,
    ModelSpec,
    OperationalHooks,
    PerformanceBaseline,
    PSANNClassifier,
    PSANNRegressor,
    RetentionPolicy,
    StreamingSupervisedData,
    TaskSpec,
    TrainingCheckpointError,
    TrainingConfig,
    TrainingEvent,
    WaveResNetRegressor,
    create_inference_runtime,
    create_model,
    fingerprint_data,
    inspect_artifact,
    load_model,
    load_runtime,
    train,
)

assert PSANNRegressor is not None
assert PSANNClassifier is not None
assert WaveResNetRegressor is not None
assert TrainingEvent is not None
assert TrainingCheckpointError is not None
assert ExplainerConfig(seed=3).seed == 3
assert RetentionPolicy().redact_raw_inputs is True
assert OperationalHooks().error_policy == "raise"
assert PerformanceBaseline(name="smoke", metrics={}).name == "smoke"
assert StreamingSupervisedData is not None
assert "shap" not in sys.modules
spec = ModelSpec(
    task=TaskSpec(kind="binary"),
    input_schema=DataSchema(input_shape=(3,)),
)
model = create_model(spec)
assert isinstance(model, PSANNClassifier)
assert TrainingConfig(epochs=1).to_dict()["epochs"] == 1
inputs = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.5, 0.25], [0.25, 0.5]], dtype=np.float32)
targets = inputs[:, 0] - inputs[:, 1]
assert fingerprint_data(inputs, targets).startswith("sha256:")
regression = create_model(
    ModelSpec(
        input_schema=DataSchema(input_shape=(2,)),
        parameters={"hidden_layers": 1, "hidden_units": 4, "random_state": 3},
    )
)
run = train(
    regression,
    (inputs, targets),
    config=TrainingConfig(epochs=1, batch_size=2, deterministic=True),
)
with tempfile.TemporaryDirectory() as directory:
    artifact = run.export(Path(directory) / "smoke.psann")
    assert inspect_artifact(artifact).backbone == "psann_mlp"
    loaded = load_model(artifact, device="cpu")
    np.testing.assert_allclose(loaded.predict(inputs), regression.predict(inputs))
    runtime = load_runtime(
        artifact,
        config=psann.InferenceConfig(batch_size=3, device="cpu"),
    )
    result = runtime.predict(inputs)
    np.testing.assert_allclose(result.values, regression.predict(inputs), rtol=1e-6, atol=1e-6)
    assert result.metadata["chunks"] == 2
    direct_runtime = create_inference_runtime(regression)
    assert direct_runtime.predict(inputs).task == "regression"
    assert callable(direct_runtime.make_explainer)
assert importlib.metadata.version("psann") == psann.__version__
assert importlib.util.find_spec("psannlm") is None
try:
    importlib.import_module("psann.lm")
except ImportError as exc:
    assert "psannlm" in str(exc)
else:
    raise AssertionError("psann.lm should require the separate psannlm wheel")
"""

LM_SMOKE = """
import importlib.metadata

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

import psann
import psannlm
import psannlm.train
from psannlm import psannLM, psannLMDataPrep

core_version = importlib.metadata.version("psann")
lm_version = importlib.metadata.version("psannlm")
assert psann.__version__ == core_version
assert psannlm.__version__ == lm_version
assert SpecifierSet(">=1.1.0rc1,<1.2").contains(core_version, prereleases=True)
lm_requirements = [Requirement(value) for value in importlib.metadata.requires("psannlm") or ()]
core_requirement = next(item for item in lm_requirements if item.name == "psann")
assert core_requirement.specifier.contains("1.1.0rc1", prereleases=True)
assert core_requirement.specifier.contains("1.1.9", prereleases=True)
assert not core_requirement.specifier.contains("1.0.9", prereleases=True)
assert not core_requirement.specifier.contains("1.2.0", prereleases=True)
assert psannLM is not None
assert psannLMDataPrep is not None
"""


def _single_wheel(directory: Path, project: str) -> Path:
    wheels = sorted(directory.glob(f"{project.replace('-', '_')}-*.whl"))
    if len(wheels) != 1:
        names = ", ".join(path.name for path in wheels) or "none"
        raise RuntimeError(f"Expected one {project} wheel in {directory}; found {names}")
    return wheels[0].resolve()


def _venv_python(root: Path) -> Path:
    if sys.platform == "win32":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def _run(python: Path, args: Sequence[str], *, cwd: Path) -> None:
    command = [str(python), *args]
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.check_call(command, cwd=cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--core-dist",
        type=Path,
        default=REPO_ROOT / "dist",
        help="Directory containing exactly one psann wheel.",
    )
    parser.add_argument(
        "--lm-dist",
        type=Path,
        default=REPO_ROOT / "psannlm" / "dist",
        help="Directory containing exactly one psannlm wheel.",
    )
    parser.add_argument(
        "--system-site-packages",
        action="store_true",
        help="Let the temporary environment reuse locally installed dependencies.",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Install the built wheels without resolving dependencies.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    core_wheel = _single_wheel(args.core_dist, "psann")
    lm_wheel = _single_wheel(args.lm_dist, "psannlm")

    with tempfile.TemporaryDirectory(prefix="psann-wheel-smoke-") as temp_text:
        temp_root = Path(temp_text)
        env_root = temp_root / "venv"
        work_root = temp_root / "work"
        work_root.mkdir()
        venv.EnvBuilder(
            with_pip=True,
            clear=True,
            system_site_packages=args.system_site_packages,
        ).create(env_root)
        python = _venv_python(env_root)

        install_args = ("-m", "pip", "install")
        if args.no_deps:
            install_args += ("--no-deps",)

        _run(python, (*install_args, str(core_wheel)), cwd=work_root)
        _run(python, ("-c", CORE_SMOKE), cwd=work_root)
        _run(python, (*install_args, str(lm_wheel)), cwd=work_root)
        _run(python, ("-c", LM_SMOKE), cwd=work_root)

    print("Built-wheel package smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
