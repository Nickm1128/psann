#!/usr/bin/env python
"""Training script delegating to the canonical PSANN-LM CLI.

Usage (after installing the LM add-on alongside psann, e.g. ``pip install psann psannlm``):

    python scripts/train_psann_lm.py [args...]

This forwards to ``python -m psannlm train``.
"""

import sys
from psannlm.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
