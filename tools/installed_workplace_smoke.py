"""Exercise the installed workplace lifecycle without importing the source tree."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import tempfile
from pathlib import Path

import numpy as np

import psann

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Also exercise the installed optional SHAP integration.",
    )
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="Fail if psann resolves from this checkout instead of site-packages.",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    package_path = Path(psann.__file__).resolve()
    if arguments.require_installed and SOURCE_ROOT.resolve() in package_path.parents:
        raise RuntimeError(f"Smoke test imported the source checkout: {package_path}")

    inputs = np.asarray(
        [
            [-1.0, 0.5],
            [-0.5, -0.25],
            [0.0, 0.0],
            [0.25, 0.75],
            [0.5, -0.5],
            [0.75, 0.25],
            [1.0, -0.75],
            [1.25, 0.5],
        ],
        dtype=np.float32,
    )
    targets = (0.8 * inputs[:, 0] - 0.2 * inputs[:, 1]).astype(np.float32)
    spec = psann.ModelSpec(
        input_schema=psann.DataSchema(
            input_shape=(2,),
            feature_names=("signal", "context"),
            output_names=("forecast",),
        ),
        parameters={"hidden_layers": 1, "hidden_units": 4, "random_state": 311},
    )
    run = psann.train(
        psann.create_model(spec),
        (inputs, targets),
        config=psann.TrainingConfig(
            epochs=1,
            batch_size=4,
            deterministic=True,
            fallback_policy="error",
        ),
    )

    with tempfile.TemporaryDirectory(prefix="psann-installed-smoke-") as temporary:
        artifact = run.export(Path(temporary) / "installed-smoke.psann")
        info = psann.inspect_artifact(artifact)
        runtime = psann.load_runtime(
            artifact,
            config=psann.InferenceConfig(batch_size=3, device="cpu"),
        )
        result = runtime.predict(inputs)
        np.testing.assert_allclose(
            result.values,
            run.model.predict(inputs),
            rtol=1e-6,
            atol=1e-6,
        )

        from fastapi.testclient import TestClient

        with TestClient(psann.create_app(runtime=runtime)) as client:
            assert client.get("/health").json() == {"status": "ok"}
            assert client.get("/ready").json() == {"status": "ready"}
            response = client.post("/predict", json={"inputs": inputs[:3].tolist()})
            response.raise_for_status()
            np.testing.assert_allclose(
                response.json()["values"],
                runtime.predict(inputs[:3]).values,
                rtol=1e-6,
                atol=1e-6,
            )

        explanation_shape = None
        if arguments.explain:
            explanation = psann.explain(
                runtime,
                inputs[:1],
                background=inputs[1:5],
                config=psann.ExplainerConfig(
                    algorithm="permutation",
                    max_evaluations=12,
                    seed=311,
                ),
            )
            explanation_shape = list(explanation.values.shape)
            if explanation.values.shape[0] != 1:
                raise AssertionError("Installed SHAP smoke returned an invalid sample axis.")

    print(
        json.dumps(
            {
                "artifact_format": info.artifact_format_version,
                "explanation_shape": explanation_shape,
                "numpy": np.__version__,
                "package_path": str(package_path),
                "psann": importlib.metadata.version("psann"),
                "status": "passed",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
