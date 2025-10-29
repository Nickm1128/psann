"""
PSANN utility scripts exposed as Python modules.

The logging CLI lives in ``psann.scripts.hisso_log_run`` and can be
invoked via ``python -m psann.scripts.hisso_log_run``.  The package is
kept intentionally lightweight so the modules can run inside the same
environment as the estimators without extra dependencies.
"""

from __future__ import annotations

__all__ = [
    "hisso_log_run",
]
