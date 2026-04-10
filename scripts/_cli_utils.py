"""Small CLI helpers shared by scripts.

These are intentionally kept lightweight and local to `scripts/` so they don't
expand the supported surface area of the `psann` library package.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def slugify(
    value: str,
    *,
    colon: str = "_",
    replace_commas: bool = False,
    replace_equals: bool = False,
) -> str:
    out = value.replace(":", colon).replace("/", "_").replace("\\", "_").replace(" ", "_")
    if replace_commas:
        out = out.replace(",", "_")
    if replace_equals:
        out = out.replace("=", "_")
    return out


def ensure_src_dir(repo_root: Path) -> Path:
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    src_dir = repo_root / "src"
    if src_dir.exists():
        src_text = str(src_dir)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)
    return src_dir


def utc_timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
