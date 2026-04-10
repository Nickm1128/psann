"""PSANN Language Modeling (LM) module package.

Exposes the public API entry points `psannLM` and `psannLMDataPrep`.
Training, data, and generation internals live under this package now; any
remaining roadmap work is tracked in `docs/backlog/psann_lm_todo.md`.
"""

from .api import psannLM, psannLMDataPrep

__all__ = ["psannLM", "psannLMDataPrep"]
