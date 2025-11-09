#!/usr/bin/env python
"""Crude decontamination tool for text shards.

Removes lines that appear to contain long substrings from provided
reference corpora (e.g., evaluation datasets) to reduce leakage.

This is a heuristic line-level filter and not a substitute for full
document matching or n-gram decontamination.

Usage:
  python tools/decontaminate.py \
    --input shards.txt \
    --refs wt2.txt lambada.txt hellaswag_questions.txt \
    --min-substr 32 \
    --output shards_clean.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Set


def read_manifest(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def read_lines(paths: Iterable[str]) -> Iterable[str]:
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for ln in fh:
                s = ln.rstrip("\n")
                if s:
                    yield s


def build_substrings(refs: Iterable[str], min_len: int) -> Set[str]:
    bank: Set[str] = set()
    for s in refs:
        ss = s.strip()
        if len(ss) < min_len:
            continue
        # take a rolling window of min_len to keep memory small
        for i in range(0, len(ss) - min_len + 1, max(1, min_len // 2)):
            bank.add(ss[i : i + min_len])
    return bank


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heuristic decontamination: remove lines matching ref substrings")
    p.add_argument("--input", required=True, type=str, help="Manifest of training text files (one per line)")
    p.add_argument("--refs", nargs="+", required=True, type=str, help="Reference text files to avoid")
    p.add_argument("--min-substr", type=int, default=32, help="Minimum substring length to match")
    p.add_argument("--output", type=str, required=True, help="Output file for filtered training lines")
    args = p.parse_args(argv)

    shards = read_manifest(args.input)
    ref_lines = list(read_lines(args.refs))
    bank = build_substrings(ref_lines, min_len=int(args.min_substr))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with out_path.open("w", encoding="utf-8") as out:
        for line in read_lines(shards):
            s = line.strip()
            if not s:
                continue
            # Drop if any bank substring appears in the line
            if any(sub in s for sub in bank):
                dropped += 1
                continue
            out.write(s + "\n")
            kept += 1
    print(f"Decontamination complete: kept={kept} dropped={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

