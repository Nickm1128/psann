from __future__ import annotations

import argparse
import os
from typing import Any, Iterator, Optional

from psannlm.lm.data.dataset import build_text_filter


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def _read_manifest(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh.readlines() if ln.strip()]


def _iter_manifest_texts(paths: list[str], limit: Optional[int]) -> Iterator[str]:
    yielded = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                yield s
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def _iter_hf_texts(args: argparse.Namespace, limit: Optional[int]) -> Iterator[str]:
    from datasets import load_dataset  # type: ignore

    text_filter = build_text_filter(
        ascii_only=bool(args.hf_keep_ascii_only),
        languages=[s for s in (args.hf_lang or [])],
        lang_threshold=float(args.hf_lang_threshold),
    )
    stream = load_dataset(
        args.hf_dataset,
        name=args.hf_name,
        split=args.hf_split,
        streaming=True,
        revision=args.hf_revision,
    )
    if args.hf_shuffle:
        try:
            stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.hf_shuffle_buffer))
        except Exception:
            pass
    yielded = 0
    for row in stream:
        try:
            text = str(row.get(args.hf_text_key, "")).strip()
        except Exception:
            text = ""
        if not text:
            continue
        if not text_filter(text):
            continue
        yield text
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def _tokenizer_sample_iterator(
    args: argparse.Namespace, shard_paths: list[str], limit: Optional[int]
) -> Iterator[str]:
    if args.hf_dataset:
        yield from _iter_hf_texts(args, limit)
    else:
        yield from _iter_manifest_texts(shard_paths, limit)
