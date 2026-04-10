#!/usr/bin/env python
"""Compatibility wrapper for the split light-probe benchmark CLI."""

from __future__ import annotations

try:
    from _run_light_probes.cli import main, parse_args
except ImportError:  # pragma: no cover - import path differs under tests
    from scripts._run_light_probes.cli import main, parse_args

__all__ = ["main", "parse_args"]

if __name__ == "__main__":
    main()
