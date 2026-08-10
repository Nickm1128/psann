"""Runtime compatibility contract between PSANN-LM and the PSANN core."""

from __future__ import annotations

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

PSANN_CORE_REQUIREMENT = ">=1.1.0rc1,<1.2"
_SUPPORTED_CORE = SpecifierSet(PSANN_CORE_REQUIREMENT)


class PSANNCoreCompatibilityError(ImportError):
    """Raised when PSANN-LM is imported with an unsupported PSANN core."""


def ensure_core_compatibility(core_version: str) -> None:
    """Fail clearly unless *core_version* is in the supported 1.1 release line."""

    try:
        parsed = Version(core_version)
    except InvalidVersion as exc:
        raise PSANNCoreCompatibilityError(
            "PSANN-LM is incompatible with the installed psann core: "
            f"found unparseable version {core_version!r}, required {PSANN_CORE_REQUIREMENT}."
        ) from exc

    if not _SUPPORTED_CORE.contains(parsed, prereleases=True):
        raise PSANNCoreCompatibilityError(
            "PSANN-LM is incompatible with the installed psann core: "
            f"found {core_version}, required {PSANN_CORE_REQUIREMENT}. "
            f'Install a compatible core with `python -m pip install "psann{PSANN_CORE_REQUIREMENT}"`.'
        )
