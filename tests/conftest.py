import json
import os
from datetime import datetime
from pathlib import Path

import pytest


def _default_outdir() -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / "gpu_tests" / ts


@pytest.fixture(scope="session")
def output_dir() -> Path:
    out = Path(os.environ.get("PSANN_OUTPUT_DIR", _default_outdir()))
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def torch_available():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def cuda_available(torch_available):
    if not torch_available:
        return False
    import torch

    return bool(torch.cuda.is_available())


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

