#!/usr/bin/env python
"""Backward-compatible shim for the psannlm.train entrypoint.

Usage (after installing the LM add-on alongside psann, e.g. ``pip install psann psannlm``):

    python scripts/train_psann_lm.py [args...]

This simply forwards to :mod:`psannlm.train`.
"""

from psannlm.train import main


if __name__ == "__main__":
    raise SystemExit(main())
