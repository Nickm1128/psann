#!/usr/bin/env python
"""Compatibility wrapper for the split GeoSparse-vs-dense benchmark CLI."""

from __future__ import annotations

try:
    from _benchmark_geo_sparse_vs_dense.main import main, parse_args
except ImportError:  # pragma: no cover - import path differs under tests
    from scripts._benchmark_geo_sparse_vs_dense.main import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    main()
