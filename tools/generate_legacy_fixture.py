"""Rebuild the retained public PSANN 0.12.7 legacy checkpoint fixture.

The fixture is deliberately produced in an isolated environment from the exact
public wheel recorded below. Tests never download packages; they consume the
retained checkpoint and its generated provenance sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
import venv
from pathlib import Path

PRODUCER_VERSION = "0.12.7"
WHEEL_NAME = "psann-0.12.7-py3-none-any.whl"
WHEEL_SHA256 = "43e6bc16a06a27b72e9073d1f80dbac70e07634df4dd01459ab949032997699b"
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/a1/4c/"
    "a67991f14275e426a7773b8623f0d1dd5a2a036d14d332926b0122cde275/"
    f"{WHEEL_NAME}"
)
PYPI_URL = f"https://pypi.org/project/psann/{PRODUCER_VERSION}/"
FIXTURE_NAME = "psann-0.12.7-regressor.pt"
SIDECAR_NAME = "psann-0.12.7-regressor.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_in(environment: Path) -> Path:
    if sys.platform == "win32":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _obtain_wheel(workspace: Path, supplied_wheel: Path | None) -> Path:
    if supplied_wheel is not None:
        wheel = supplied_wheel.resolve()
    else:
        wheel = workspace / WHEEL_NAME
        request = urllib.request.Request(
            WHEEL_URL,
            headers={"User-Agent": "psann-legacy-fixture-builder/1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            wheel.write_bytes(response.read())
    if not wheel.is_file():
        raise FileNotFoundError(f"Producer wheel does not exist: {wheel}")
    actual = _sha256(wheel)
    if actual != WHEEL_SHA256:
        raise RuntimeError(f"Producer wheel SHA256 mismatch: expected {WHEEL_SHA256}, got {actual}")
    return wheel


def _producer_program() -> str:
    return textwrap.dedent(
        f"""
        import importlib.metadata
        import json
        import platform
        import sys

        import numpy as np
        import psann
        import torch
        from psann import PSANNRegressor

        expected_version = {PRODUCER_VERSION!r}
        installed_version = importlib.metadata.version("psann")
        if installed_version != expected_version or psann.__version__ != expected_version:
            raise RuntimeError(
                f"Expected public PSANN {{expected_version}}, got distribution "
                f"{{installed_version}} and package {{psann.__version__}}"
            )

        inputs = np.asarray(
            [
                [-1.00, -0.50],
                [-0.50, 0.25],
                [0.00, 0.00],
                [0.25, 0.50],
                [0.75, -0.25],
                [1.00, 0.75],
                [1.25, -1.00],
                [1.50, 0.50],
            ],
            dtype=np.float32,
        )
        targets = (0.75 * inputs[:, 0] - 0.25 * inputs[:, 1] + 0.10).astype(np.float32)
        training = {{
            "hidden_layers": 1,
            "hidden_units": 4,
            "epochs": 3,
            "batch_size": 4,
            "lr": 0.01,
            "random_state": 1729,
            "device": "cpu",
        }}
        estimator = PSANNRegressor(**training).fit(inputs, targets)
        output_path = sys.argv[1]
        estimator.save(output_path)
        expected = np.asarray(estimator.predict(inputs), dtype=np.float32)
        restored = PSANNRegressor.load(output_path, map_location="cpu")
        np.testing.assert_allclose(restored.predict(inputs), expected, rtol=0.0, atol=0.0)

        print(
            json.dumps(
                {{
                    "inputs": inputs.tolist(),
                    "targets": targets.tolist(),
                    "expected_predictions": expected.tolist(),
                    "training": training,
                    "producer_environment": {{
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "torch": torch.__version__,
                    }},
                }},
                sort_keys=True,
            )
        )
        """
    )


def build_fixture(output_dir: Path, supplied_wheel: Path | None) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="psann-0127-fixture-") as temporary:
        workspace = Path(temporary)
        wheel = _obtain_wheel(workspace, supplied_wheel)
        environment = workspace / "producer-venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = _python_in(environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-deps",
                str(wheel),
            ],
            cwd=workspace,
            check=True,
        )

        fixture = output_dir / FIXTURE_NAME
        producer = workspace / "produce.py"
        producer.write_text(_producer_program(), encoding="utf-8")
        result = subprocess.run(
            [str(python), str(producer), str(fixture.resolve())],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        )
        generated = json.loads(result.stdout.strip().splitlines()[-1])
        sidecar = output_dir / SIDECAR_NAME
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "fixture": {
                        "filename": FIXTURE_NAME,
                        "sha256": _sha256(fixture),
                        "format": "torch-pickle legacy checkpoint",
                    },
                    "producer": {
                        "distribution": "psann",
                        "version": PRODUCER_VERSION,
                        "pypi_url": PYPI_URL,
                        "wheel": {
                            "filename": WHEEL_NAME,
                            "url": WHEEL_URL,
                            "sha256": WHEEL_SHA256,
                        },
                        **generated["producer_environment"],
                    },
                    "case": {
                        "estimator": "PSANNRegressor",
                        "training": generated["training"],
                        "inputs": generated["inputs"],
                        "targets": generated["targets"],
                        "expected_predictions": generated["expected_predictions"],
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return fixture, sidecar


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "tests" / "fixtures" / "legacy",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        help="Use a local wheel after verifying it against the pinned public SHA256.",
    )
    arguments = parser.parse_args()
    fixture, sidecar = build_fixture(arguments.output_dir.resolve(), arguments.wheel)
    print(f"Wrote {fixture} ({_sha256(fixture)})")
    print(f"Wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
