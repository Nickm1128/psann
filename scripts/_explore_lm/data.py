"""Bounded corpus preparation with document-disjoint, reproducible holdouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from torch.utils.data import Dataset, IterableDataset

from .config import fingerprint


class Blocks(Dataset):
    def __init__(self, blocks):
        self.blocks = blocks

    def __len__(self):
        return self.blocks.shape[0]

    def __getitem__(self, index):
        row = self.blocks[index]
        return {"input_ids": row[:-1], "labels": row[1:]}


class RepeatBlocks(IterableDataset):
    """Identical seeded order across architectures; explicit cycling of small caches."""

    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, seed

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed)
        while True:
            for index in torch.randperm(len(self.dataset), generator=generator).tolist():
                yield self.dataset[index]


def documents(data):
    yield from data.get("texts", [])
    for filename in data.get("files", []):
        path = Path(filename)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield (
                        json.loads(line)[data.get("text_key", "text")]
                        if path.suffix == ".jsonl"
                        else line
                    )


def content_hash(cache):
    digest = hashlib.sha256()
    for split in ("train", "validation"):
        digest.update(cache[split].contiguous().numpy().tobytes())
    digest.update(fingerprint(cache["tokenizer"]).encode())
    return digest.hexdigest()


def prepare(config):
    data = config["data"]
    length = int(data.get("sequence_length", 256))
    fraction = float(data.get("validation_fraction", 0.05))
    limits = {
        "train": int(data.get("train_tokens", 2097152)),
        "validation": int(data.get("validation_tokens", 131072)),
    }
    max_documents = int(data.get("max_documents", 10000))
    if (
        length <= 0
        or not 0 < fraction < 0.5
        or max_documents <= 0
        or min(limits.values()) < length + 1
    ):
        raise ValueError(
            "Invalid sequence length, holdout fraction, document limit, or token budget."
        )
    texts, hashes = {"train": [], "validation": []}, {"train": [], "validation": []}
    seen, duplicates, scanned = set(), 0, 0
    for text in documents(data):
        scanned += 1
        if not isinstance(text, str):
            raise TypeError("Corpus documents must be strings.")
        text = text.strip()
        if text:
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)
                bucket = hashlib.sha256((str(data.get("split_seed", 731)) + key).encode()).digest()
                split = (
                    "validation"
                    if int.from_bytes(bucket[:8], "big") / 2**64 < fraction
                    else "train"
                )
                texts[split].append(text)
                hashes[split].append(key)
        if scanned >= max_documents:
            break
    if not all(texts.values()):
        raise ValueError("The bounded sample must contain both train and validation documents.")
    tok_config = dict(config.get("tokenizer", {"backend": "simple"}))
    tok_config.setdefault("backend", "simple")
    if tok_config["backend"] not in {"simple", "tokenizers", "sentencepiece"}:
        raise ValueError(
            "Use an explicit tokenizer backend; auto selection is unsuitable for comparisons."
        )
    if tok_config.get("backend", "simple") != "simple" and not tok_config.get("model_path"):
        raise ValueError(
            "Exploration requires an existing fitted tokenizer for non-simple backends."
        )
    tokenizer = Tokenizer(TokenizerConfig(**tok_config))
    tokenizer.fit(texts["train"] if tok_config.get("backend", "simple") == "simple" else [])
    tensors = {}
    for split, split_texts in texts.items():
        tokens = []
        for text in split_texts:
            tokens.extend(tokenizer.encode(text, add_specials=True))
            if len(tokens) >= limits[split]:
                break
        tokens = tokens[: limits[split]]
        starts = range(0, len(tokens) - length, length)
        blocks = [tokens[start : start + length + 1] for start in starts]
        if not blocks:
            raise ValueError(f"{split} did not yield a full token block.")
        tensors[split] = torch.tensor(blocks, dtype=torch.long)
    metadata = {
        "config_signature": fingerprint({"data": config["data"], "tokenizer": tok_config}),
        "documents_scanned": scanned,
        "duplicate_documents_removed": duplicates,
        "document_hashes": hashes,
        "sequence_length": length,
        "train_tokens": int(tensors["train"].shape[0] * length),
        "validation_tokens": int(tensors["validation"].shape[0] * length),
        "tokenizer_sha256": fingerprint(tokenizer.to_state()),
    }
    cache = {
        "schema": "psannlm.exploration.tokens",
        "schema_version": 1,
        "metadata": metadata,
        "tokenizer": tokenizer.to_state(),
        **tensors,
    }
    metadata["content_sha256"] = content_hash(cache)
    return cache


def load_cache(path, config):
    cache = torch.load(path, map_location="cpu", weights_only=True)
    if cache.get("schema") != "psannlm.exploration.tokens" or cache.get("schema_version") != 1:
        raise ValueError("Unsupported exploration token cache.")
    tok_config = dict(config.get("tokenizer", {}))
    tok_config.setdefault("backend", "simple")
    expected = fingerprint({"data": config["data"], "tokenizer": tok_config})
    if cache["metadata"]["config_signature"] != expected:
        raise ValueError(
            "Token cache conflicts with data/tokenizer configuration; prepare a new cache."
        )
    if set(cache["metadata"]["document_hashes"]["train"]) & set(
        cache["metadata"]["document_hashes"]["validation"]
    ):
        raise ValueError("Training and validation document hashes overlap.")
    tokenizer = Tokenizer.from_state(cache["tokenizer"])
    for split in ("train", "validation"):
        tensor = cache[split]
        if (
            tensor.ndim != 2
            or tensor.shape[0] == 0
            or tensor.shape[1] != cache["metadata"]["sequence_length"] + 1
            or tensor.dtype != torch.long
        ):
            raise ValueError(f"Invalid {split} token block shape or dtype.")
        if int(tensor.min()) < 0 or int(tensor.max()) >= tokenizer.vocab_size:
            raise ValueError(f"Invalid {split} token IDs.")
    if cache["metadata"].get("content_sha256") != content_hash(cache):
        raise ValueError("Token cache content hash does not match its metadata.")
    return cache
