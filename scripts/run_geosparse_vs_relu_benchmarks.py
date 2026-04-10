#!/usr/bin/env python3
"""Compatibility wrapper for the split GeoSparse-vs-ReLU benchmark CLI."""

from __future__ import annotations

try:
    from _run_geosparse_vs_relu_benchmarks.cli import main, parse_args
except ImportError:  # pragma: no cover - import path differs under tests
    from scripts._run_geosparse_vs_relu_benchmarks.cli import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    main()
