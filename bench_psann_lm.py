#!/usr/bin/env python
"""Compatibility wrapper for the split PSANN-LM scaling benchmark CLI."""

from __future__ import annotations

from _bench_psann_lm.cli import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    raise SystemExit(main())
