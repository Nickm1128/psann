#!/usr/bin/env python
"""Build a newline-separated manifest of text shards.

Each line in the manifest is a file path to a text file containing
newline-delimited documents. The trainer will stream and pack tokens
across files.

Examples:
  # All .txt files under a root directory (recursively)
  python tools/build_manifest.py --roots /data/en_corpus \
    --pattern "*.txt" --recurse --absolute --output en_manifest.txt

  # Multiple roots and patterns, shuffled
  python tools/build_manifest.py --roots /data/wiki /data/books \
    --pattern "*.txt" --pattern "*.jsonl" --recurse --shuffle \
    --output shards.txt
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List


def _collect(root: Path, patterns: List[str], recurse: bool) -> List[Path]:
    files: List[Path] = []
    if recurse:
        for pat in patterns:
            files.extend(root.rglob(pat))
    else:
        for pat in patterns:
            files.extend(root.glob(pat))
    # Keep only files
    return [p for p in files if p.is_file()]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build newline-separated manifest of text shard paths")
    p.add_argument("--roots", nargs="+", type=str, help="One or more root directories (or files)")
    p.add_argument("--pattern", action="append", default=["*.txt"], help="Glob pattern(s) to include")
    p.add_argument("--recurse", action="store_true", help="Search directories recursively")
    p.add_argument("--absolute", action="store_true", help="Write absolute paths in manifest")
    p.add_argument("--min-bytes", type=int, default=1, help="Skip files smaller than this size in bytes")
    p.add_argument("--shuffle", action="store_true", help="Shuffle file order for stochastic ingestion")
    p.add_argument("--seed", type=int, default=1337, help="Shuffle seed")
    p.add_argument("--output", required=True, type=str, help="Output manifest path")
    args = p.parse_args(argv)

    patterns: List[str] = [str(x) for x in (args.pattern or ["*.txt"])]
    paths: List[Path] = []
    for r in args.roots:
        root = Path(r)
        if root.is_file():
            if root.stat().st_size >= int(args.min_bytes):
                paths.append(root)
            continue
        if not root.exists():
            continue
        paths.extend(_collect(root, patterns, bool(args.recurse)))

    # Filter and optionally shuffle
    uniq = []
    seen = set()
    for p in paths:
        try:
            if p.stat().st_size < int(args.min_bytes):
                continue
        except Exception:
            continue
        key = str(p.resolve()) if args.absolute else str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(key)

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(uniq)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in uniq:
            fh.write(p + "\n")
    print(f"Wrote manifest with {len(uniq)} files -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

