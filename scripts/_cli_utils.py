"""Small CLI helpers shared by scripts.

These are intentionally kept lightweight and local to `scripts/` so they don't
expand the supported surface area of the `psann` library package.
"""

from __future__ import annotations


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(
    value: str,
    *,
    colon: str = "_",
    replace_commas: bool = False,
    replace_equals: bool = False,
) -> str:
    out = (
        value.replace(":", colon)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    if replace_commas:
        out = out.replace(",", "_")
    if replace_equals:
        out = out.replace("=", "_")
    return out

