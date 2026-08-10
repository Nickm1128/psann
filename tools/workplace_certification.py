"""Run the installed PSANN workplace certification module."""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "src"
if _SOURCE.is_dir():
    sys.path.insert(0, str(_SOURCE))

from psann.platform.certification import main

if __name__ == "__main__":
    raise SystemExit(main())
