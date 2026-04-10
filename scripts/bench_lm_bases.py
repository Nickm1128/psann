#!/usr/bin/env python
"""Compatibility wrapper for the split LM base benchmark CLI."""

from __future__ import annotations

try:
    from _bench_lm_bases.main import main
except ImportError:  # pragma: no cover - import path differs under tests
    from scripts._bench_lm_bases.main import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
