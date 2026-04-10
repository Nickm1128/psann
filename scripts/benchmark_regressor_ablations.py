#!/usr/bin/env python
"""Compatibility wrapper for the split regressor ablation benchmark CLI."""

from __future__ import annotations

try:
    from _benchmark_regressor_ablations.io import parse_args
    from _benchmark_regressor_ablations.main import main
except ImportError:  # pragma: no cover - import path differs under tests
    from scripts._benchmark_regressor_ablations.io import parse_args
    from scripts._benchmark_regressor_ablations.main import main

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    raise SystemExit(main())
