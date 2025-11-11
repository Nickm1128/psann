"""Datasets and collation for PSANN-LM.

This module includes a minimal character-level LM dataset built from
tokenized texts. It supports basic sequence chunking and a simple
sequence packing mode across documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple, Dict, Any, Sequence, Callable
import os

import torch
from torch.utils.data import Dataset, IterableDataset

from .tokenizer import Tokenizer


def build_text_filter(
    *,
    ascii_only: bool = False,
    languages: Optional[Sequence[str]] = None,
    lang_threshold: float = 0.8,
) -> Callable[[str], bool]:
    """Return a predicate that checks whether a text sample passes filters."""

    languages_norm = [s.lower() for s in languages or [] if s]
    langdetect_state: Dict[str, Any] = {
        "fn": None,
        "unavailable": False,
        "warned": False,
    }

    def _maybe_load_langdetect() -> Optional[Callable[[str], Any]]:
        if langdetect_state["unavailable"]:
            return None
        if langdetect_state["fn"] is None:
            try:
                from langdetect import detect_langs  # type: ignore

                langdetect_state["fn"] = detect_langs
            except Exception:
                langdetect_state["unavailable"] = True
                if not langdetect_state["warned"]:
                    print("[text_filter] langdetect not available; skipping language filter")
                    langdetect_state["warned"] = True
                return None
        return langdetect_state["fn"]  # type: ignore[return-value]

    def _filter(text: str) -> bool:
        if not text:
            return False
        if ascii_only and not text.isascii():
            return False
        if languages_norm:
            if len(text) < 5:
                return True
            detect_fn = _maybe_load_langdetect()
            if detect_fn is None:
                return True
            try:
                candidates = detect_fn(text)
            except Exception:
                return False
            for cand in candidates:
                try:
                    code = cand.lang.lower()
                    prob = float(cand.prob)
                except Exception:
                    continue
                if code in languages_norm and prob >= float(lang_threshold):
                    return True
            return False
        return True

    return _filter


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


class HFTextStreamingLMDataset(IterableDataset):
    """Streaming dataset backed by Hugging Face `datasets` with streaming=True.

    Loads a text dataset from the Hub and yields fixed-length examples
    constructed from a contiguous token stream across rows.

    Parameters
    ----------
    dataset:
        HF dataset repo id, e.g. "allenai/c4" or "wikitext".
    split:
        Split name (e.g., "train").
    text_key:
        Column name containing raw text.
    name:
        Optional dataset configuration name (subset).
    revision:
        Optional branch/tag/commit on the Hub.
    shuffle:
        If True, apply streaming shuffle with a buffer.
    seed:
        RNG seed for shuffling.
    shuffle_buffer:
        Buffer size for streaming shuffle.
    """

    def __init__(
        self,
        *,
        dataset: str,
        tokenizer: Tokenizer,
        cfg: PackingConfig = PackingConfig(),
        split: str = "train",
        text_key: str = "text",
        name: str | None = None,
        revision: str | None = None,
        shuffle: bool = False,
        seed: int = 1337,
        shuffle_buffer: int = 10000,
        ascii_only: bool = False,
        languages: Optional[Sequence[str]] = None,
        lang_threshold: float = 0.8,
    ) -> None:
        super().__init__()
        self.dataset = str(dataset)
        self.split = str(split)
        self.text_key = str(text_key)
        self.name = name
        self.revision = revision
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.tok = tokenizer
        self.cfg = cfg
        self._filter = build_text_filter(
            ascii_only=ascii_only,
            languages=languages,
            lang_threshold=lang_threshold,
        )

    def __iter__(self):
        from datasets import load_dataset  # type: ignore

        stream = load_dataset(
            self.dataset,
            name=self.name,
            split=self.split,
            streaming=True,
            revision=self.revision,
        )
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                raise RuntimeError
        except Exception:
            rank = int(os.environ.get("RANK", "0"))
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            try:
                stream = stream.shard(num_shards=world_size, index=rank)
            except Exception:
                pass
        if self.shuffle:
            try:
                stream = stream.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
            except Exception:
                pass

        T = int(self.cfg.max_length)
        buffer: list[int] = []
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and worker_info.num_workers > 1:
            try:
                stream = stream.shard(num_shards=worker_info.num_workers, index=worker_info.id)
            except Exception:
                pass

        for row in stream:
            try:
                s = str(row.get(self.text_key, "")).strip()
            except Exception:
                s = ""
            if not s:
                continue
            if not self._filter(s):
                continue
            ids = self.tok.encode(s, add_specials=True)
            buffer.extend(ids)
            while len(buffer) >= T + 1:
                chunk = buffer[: T + 1]
                del buffer[:T]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }
