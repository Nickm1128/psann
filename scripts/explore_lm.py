"""Prepare, inspect, probe, and compare bounded PSANN language-model trials."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts._explore_lm.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
