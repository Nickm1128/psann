"""Helpers for keeping the Hugging Face datasets cache bounded."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Tuple


def _resolve_cache_root() -> Path:
    env = os.environ.get("HF_DATASETS_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "huggingface" / "datasets"


def _iter_cache_files(root: Path, subdirs: Iterable[str]) -> list[tuple[float, int, Path]]:
    entries: list[tuple[float, int, Path]] = []
    for sub in subdirs:
        base = root / sub
        if not base.exists():
            continue
        try:
            iterator = base.rglob("*")
        except Exception:
            continue
        for path in iterator:
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
    return entries


def _prune_empty_dirs(root: Path, subdirs: Iterable[str]) -> None:
    for sub in subdirs:
        base = root / sub
        if not base.exists():
            continue
        # Walk deepest directories first
        for path in sorted(base.rglob("*"), reverse=True):
            if not path.is_dir():
                continue
            try:
                has_children = any(path.iterdir())
            except OSError:
                continue
            if not has_children:
                try:
                    path.rmdir()
                except OSError:
                    pass


def cleanup_hf_cache(
    max_bytes: int,
    *,
    subdirs: Iterable[str] = ("downloads", "extracted"),
) -> Tuple[int, int]:
    """Trim HF datasets cache directories down to at most `max_bytes`.

    Returns a tuple of (freed_bytes, total_bytes_after_cleanup).
    """

    if max_bytes <= 0:
        return (0, 0)

    root = _resolve_cache_root()
    entries = _iter_cache_files(root, subdirs)
    total = sum(size for _, size, _ in entries)
    if total <= max_bytes:
        return (0, total)

    entries.sort()  # oldest first
    freed = 0
    for _, size, path in entries:
        try:
            path.unlink()
        except OSError:
            continue
        freed += size
        total -= size
        if total <= max_bytes:
            break

    _prune_empty_dirs(root, subdirs)
    return (freed, total)


__all__ = ["cleanup_hf_cache"]
