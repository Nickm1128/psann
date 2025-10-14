from __future__ import annotations

"""Helpers for normalising deprecated API parameter aliases.

Several public entry points still accept legacy keyword names (e.g. the
``hidden_width`` alias for ``hidden_units``).  Historically each class handled
those shims independently which made it easy for behaviours to drift.

This module provides a small helper for reconciling primary parameters with
their aliases in one place so we can emit consistent warnings and validation
errors across the codebase.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeVar
import warnings

IntLike = TypeVar("IntLike", bound=int)


@dataclass(frozen=True)
class AliasResolution:
    """Result of reconciling a parameter alias with its primary name."""

    value: Optional[int]
    provided_primary: bool
    provided_alias: bool
    used_primary: bool
    used_alias: bool


def resolve_int_alias(
    *,
    primary_value: Optional[Any],
    alias_value: Optional[Any],
    primary_name: str,
    alias_name: str,
    context: str,
    default: Optional[int] = None,
    prefer: Literal["primary", "alias"] = "primary",
    alias_deprecated: bool = True,
    alias_warning_category: type[Warning] = DeprecationWarning,
    mismatch_strategy: Literal["warn", "error"] = "warn",
    mismatch_message: Optional[str] = None,
    mismatch_warning_category: type[Warning] = UserWarning,
) -> AliasResolution:
    """Coerce paired integer parameters into a single canonical value.

    Parameters
    ----------
    primary_value:
        Value supplied for the canonical keyword (e.g. ``hidden_units``).
    alias_value:
        Value supplied for the legacy alias (e.g. ``hidden_width``).
    primary_name:
        Canonical keyword name for error message construction.
    alias_name:
        Alias keyword name for warning construction.
    context:
        Short description of the caller used to prefix warning/error messages.
    default:
        Optional default applied when neither parameter was provided.
    prefer:
        Which value wins when both are given (only relevant when they differ).
    alias_deprecated:
        Whether using the alias alone should trigger a deprecation warning.
    alias_warning_category:
        Warning category for alias-only usage (defaults to ``DeprecationWarning``).
    mismatch_strategy:
        Behaviour when both parameters are provided but differ.  ``"warn"``
        mirrors prior behaviour in estimators/LSM classes, while ``"error"``
        keeps the stricter conv modules behaviour.
    mismatch_message:
        Optional override for the mismatch warning/error message.
    mismatch_warning_category:
        Warning category emitted when ``mismatch_strategy=="warn"``.

    Returns
    -------
    AliasResolution
        Dataclass describing the reconciled value and which inputs were used.
    """

    provided_primary = primary_value is not None
    provided_alias = alias_value is not None

    def _coerce_int(raw: Any, *, name: str) -> int:
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{context}: `{name}` must be an integer (got {raw!r}).")

    primary_int = _coerce_int(primary_value, name=primary_name) if provided_primary else None
    alias_int = _coerce_int(alias_value, name=alias_name) if provided_alias else None

    used_primary = False
    used_alias = False

    if provided_primary and provided_alias:
        values_match = alias_int == primary_int
        if not values_match:
            if mismatch_message is None:
                if prefer == "primary":
                    mismatch_message = (
                        f"{context}: `{primary_name}` overrides `{alias_name}` because the values differ."
                    )
                else:
                    mismatch_message = (
                        f"{context}: `{alias_name}` overrides `{primary_name}` because the values differ."
                    )
            if mismatch_strategy == "error":
                raise ValueError(mismatch_message)
            warnings.warn(mismatch_message, mismatch_warning_category, stacklevel=3)
        if prefer == "primary" or values_match:
            resolved = primary_int
            used_primary = True
            used_alias = values_match
        else:
            resolved = alias_int
            used_primary = values_match
            used_alias = True
    elif provided_primary:
        resolved = primary_int
        used_primary = True
    elif provided_alias:
        if alias_deprecated:
            warnings.warn(
                f"{context}: `{alias_name}` is deprecated; use `{primary_name}` instead.",
                alias_warning_category,
                stacklevel=3,
            )
        resolved = alias_int
        used_alias = True
    else:
        resolved = default

    if resolved is not None:
        resolved = _coerce_int(resolved, name=primary_name if provided_primary else alias_name)

    return AliasResolution(
        value=resolved,
        provided_primary=provided_primary,
        provided_alias=provided_alias,
        used_primary=used_primary,
        used_alias=used_alias,
    )

