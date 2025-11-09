#!/usr/bin/env python
"""Lightweight text deduplication utility.

Reads newline-delimited text from input files and writes unique lines
to stdout (or a file). Optionally uses MinHash LSH if `datasketch` is
installed to deduplicate near-duplicates; otherwise falls back to exact
hash dedup.

Usage:
  python tools/dedupe.py --input shards.txt --output unique.txt
  python tools/dedupe.py --inputs file1.txt file2.txt > unique.txt
  python tools/dedupe.py --inputs file1.txt --minhash --threshold 0.9
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set


def read_sources(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                s = ln.rstrip("\n")
                if s:
                    yield s


def exact_dedupe(lines: Iterable[str]) -> List[str]:
    seen: Set[int] = set()
    out: List[str] = []
    for s in lines:
        h = hash(s)
        if h in seen:
            continue
        seen.add(h)
        out.append(s)
    return out


def minhash_dedupe(lines: List[str], threshold: float = 0.9) -> List[str]:
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore
    except Exception:
        return exact_dedupe(lines)

    def shingles(s: str, k: int = 5) -> List[str]:
        s = s.lower()
        return [s[i : i + k] for i in range(max(0, len(s) - k + 1))]

    lsh = MinHashLSH(threshold=float(threshold), num_perm=128)
    keep: List[str] = []
    for s in lines:
        mh = MinHash(num_perm=128)
        for g in shingles(s):
            mh.update(g.encode("utf-8"))
        dup = False
        for _ in lsh.query(mh):
            dup = True
            break
        if dup:
            continue
        lsh.insert(str(len(keep)), mh)
        keep.append(s)
    return keep


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Deduplicate newline-delimited text inputs")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=str, help="Path to a manifest of text files (one per line)")
    g.add_argument("--inputs", nargs="*", type=str, help="One or more text files")
    p.add_argument("--output", type=str, default=None, help="Output file (default: stdout)")
    p.add_argument("--minhash", action="store_true", help="Use MinHash LSH if available")
    p.add_argument("--threshold", type=float, default=0.9, help="Jaccard threshold for MinHash LSH")
    args = p.parse_args(argv)

    if args.input:
        manifest = Path(args.input)
        if not manifest.exists():
            raise SystemExit(f"Manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8") as fh:
            files = [ln.strip() for ln in fh if ln.strip()]
    else:
        files = list(args.inputs or [])

    lines = list(read_sources(files))
    if args.minhash:
        uniq = minhash_dedupe(lines, threshold=float(args.threshold))
    else:
        uniq = exact_dedupe(lines)

    if args.output:
        outp = Path(args.output)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            for s in uniq:
                fh.write(s + "\n")
    else:
        for s in uniq:
            sys.stdout.write(s + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

