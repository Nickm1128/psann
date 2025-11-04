"""Datasets and collation for PSANN-LM.

This module includes a minimal character-level LM dataset built from
tokenized texts. It supports basic sequence chunking and a simple
sequence packing mode across documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, IterableDataset

from .tokenizer import Tokenizer


@dataclass
class PackingConfig:
    max_length: int = 1024
    pack_sequences: bool = True


class LMDataset(Dataset):
    """Language modeling dataset with next-token prediction targets.

    Builds fixed-length examples from tokenized texts. Each item returns:
      - input_ids: (T,) tensor
      - labels:    (T,) tensor (shifted targets)
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: Tokenizer,
        cfg: PackingConfig = PackingConfig(),
    ) -> None:
        self.cfg = cfg
        self.tok = tokenizer
        # Tokenize all texts
        encoded: List[List[int]] = [self.tok.encode(t, add_specials=True) for t in texts]

        # Build a contiguous stream if packing is enabled; else keep per-doc
        self._examples: List[List[int]] = []
        T = int(self.cfg.max_length)
        if self.cfg.pack_sequences:
            stream: List[int] = []
            for ids in encoded:
                stream.extend(ids)
            # Slide window with stride T to create non-overlapping chunks of length T+1
            for i in range(0, max(0, len(stream) - 1 - T), T):
                chunk = stream[i : i + T + 1]
                if len(chunk) == T + 1:
                    self._examples.append(chunk)
        else:
            for ids in encoded:
                # Per-doc chunking
                for i in range(0, max(0, len(ids) - 1 - T), T):
                    chunk = ids[i : i + T + 1]
                    if len(chunk) == T + 1:
                        self._examples.append(chunk)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = self._examples[idx]
        # Shift for labels
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def collate_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # Fixed-length items; simple stack
    input_ids = torch.stack([it["input_ids"] for it in items], dim=0)
    labels = torch.stack([it["labels"] for it in items], dim=0)
    return {"input_ids": input_ids, "labels": labels}


class StreamingLMDataset(IterableDataset):
    """Streaming dataset reading from text files line-by-line.

    Yields fixed-length examples constructed from a contiguous token stream
    across file boundaries. Deterministic document-order shuffling is supported.
    """

    def __init__(
        self,
        paths: Iterable[str],
        tokenizer: Tokenizer,
        cfg: PackingConfig = PackingConfig(),
        *,
        shuffle_docs: bool = False,
        seed: int = 1337,
    ) -> None:
        super().__init__()
        self.paths = [p for p in paths]
        self.tok = tokenizer
        self.cfg = cfg
        self.shuffle_docs = bool(shuffle_docs)
        self.seed = int(seed)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        import os
        import random as _random

        paths = [p for p in self.paths if os.path.exists(p)]
        if self.shuffle_docs:
            rng = _random.Random(self.seed)
            rng.shuffle(paths)

        T = int(self.cfg.max_length)
        stream: List[int] = []

        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    for ln in fh:
                        s = ln.strip()
                        if not s:
                            continue
                        ids = self.tok.encode(s, add_specials=True)
                        stream.extend(ids)
                        while len(stream) >= T + 1:
                            chunk = stream[: T + 1]
                            # drop consumed tokens (non-overlapping windows)
                            del stream[:T]
                            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                            labels = torch.tensor(chunk[1:], dtype=torch.long)
                            yield {"input_ids": input_ids, "labels": labels}
            except Exception:
                continue
