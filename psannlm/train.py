#!/usr/bin/env python
"""Compatibility facade for the split PSANN-LM training CLI."""

from __future__ import annotations

from ._train import build_parser, main, str2bool

__all__ = ["build_parser", "main", "str2bool"]

if __name__ == "__main__":
    raise SystemExit(main())
