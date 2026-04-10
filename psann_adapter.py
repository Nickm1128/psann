"""Compatibility shim for the PSANN-LM lm-eval adapter.

The active implementation lives in `psannlm.eval_adapter`. This file remains at
repo root so existing lm-eval commands using `psann_adapter.PSANNLM` keep
working while docs and scripts move to the package-local import path.
"""

from __future__ import annotations

from psannlm.eval_adapter import PSANNLM

__all__ = ["PSANNLM"]
