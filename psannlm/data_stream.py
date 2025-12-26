"""
Streaming token iterator utilities for PSANN-LM.

Builds an on-the-fly token stream from Hugging Face datasets (streaming=True)
without ever storing the full corpus on disk. Text rows are tokenized lazily,
buffered into a bounded queue, and emitted as fixed-length chunks suitable for
next-token prediction.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Callable

import torch
from datasets import load_dataset


def _build_encode_fn(tokenizer):
    if callable(getattr(tokenizer, "__call__", None)):

        def _encode(text: str) -> List[int]:
            out = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                truncation=False,
            )
            if isinstance(out, dict):
                ids = out.get("input_ids")
            else:
                ids = getattr(out, "input_ids", out)
            if ids is None:
                return []
            if isinstance(ids, torch.Tensor):
                return ids.tolist()
            return list(ids)

        return _encode

    if hasattr(tokenizer, "encode"):

        def _encode(text: str) -> List[int]:
            return list(tokenizer.encode(text, add_specials=False))

        return _encode

    raise TypeError("tokenizer must be callable or expose an encode() method")


def streamed_token_iterator(
    *,
    dataset_name: str,
    split: str,
    tokenizer,
    dataset_config: Optional[str] = None,
    text_field: str = "text",
    seq_len: int = 2048,
    shuffle_seed: int = 42,
    shuffle_buffer: int = 10_000,
    pack_buffer_tokens: int = 2_000_000,
    skip_sequences: int = 0,
    worker_id: Optional[int] = None,
    num_workers: Optional[int] = None,
    process_rank: Optional[int] = None,
    process_world_size: Optional[int] = None,
    text_filter: Optional[Callable[[str], bool]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield dicts of {"input_ids": Tensor[T], "labels": Tensor[T]} produced from a
    streaming HF dataset without materializing shards locally.
    """

    encode = _build_encode_fn(tokenizer)
    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        split=split,
        streaming=True,
    )

    if process_rank is None or process_world_size is None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                process_rank = dist.get_rank()
                process_world_size = dist.get_world_size()
        except Exception:
            pass

    if process_rank is None:
        process_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if process_world_size is None:
        process_world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if process_world_size > 1:
        try:
            ds = ds.shard(num_shards=process_world_size, index=process_rank)
        except Exception:
            pass

    effective_seed = int(shuffle_seed) + int(process_rank or 0)
    if num_workers and worker_id is not None:
        effective_seed += int(worker_id)
        try:
            ds = ds.shard(num_shards=int(num_workers), index=int(worker_id))
        except Exception:
            pass

    if shuffle_buffer and shuffle_buffer > 0:
        try:
            ds = ds.shuffle(seed=effective_seed, buffer_size=int(shuffle_buffer))
        except Exception:
            pass

    seq_len = int(seq_len)
    pack_cap = max(int(pack_buffer_tokens), seq_len * 2)
    buffer: List[int] = []
    remaining_skip = max(0, int(skip_sequences))

    for row in ds:
        try:
            text = str(row.get(text_field, "")).strip()
        except Exception:
            text = ""
        if not text:
            continue
        if text_filter is not None and not text_filter(text):
            continue
        try:
            ids = encode(text)
        except Exception:
            continue
        if not ids:
            continue
        buffer.extend(int(t) for t in ids)

        while len(buffer) >= seq_len + 1:
            chunk = buffer[: seq_len + 1]
            if remaining_skip > 0:
                remaining_skip -= 1
            else:
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}
            del buffer[:seq_len]

        if len(buffer) > pack_cap:
            drop = len(buffer) - pack_cap
            del buffer[:drop]
