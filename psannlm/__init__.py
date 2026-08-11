"""PSANN-LM package (language modeling utilities)."""

from psann import __version__ as _psann_core_version

from ._compat import PSANNCoreCompatibilityError, ensure_core_compatibility
from ._version import __version__

ensure_core_compatibility(_psann_core_version)

from .lm import psannLM, psannLMDataPrep  # noqa: E402

__all__ = [
    "PSANNCoreCompatibilityError",
    "__version__",
    "psannLM",
    "psannLMDataPrep",
]
