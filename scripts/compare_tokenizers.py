#!/usr/bin/env python
"""Utility script to compare LM tokenizer backends.

Records basic fit/encode timings plus rough quality/footprint proxies for
SentencePiece and Hugging Face `tokenizers` on a shared corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_CORPUS = Path("examples/lm/sample_texts.txt")


def load_texts(path: Path) -> List[str]:
    data = path.read_text(encoding="utf-8").splitlines()
    texts = [ln.strip() for ln in data if ln.strip()]
    if not texts:
        raise ValueError(f"corpus '{path}' does not contain any non-empty lines")
    return texts


def replicate_texts(texts: Iterable[str], repeat: int) -> List[str]:
    items = list(texts)
    return items * max(1, repeat)


def measure_sentencepiece(
    texts: List[str],
    *,
    vocab_size: int,
    character_coverage: float,
) -> Dict[str, float]:
    import sentencepiece as spm  # type: ignore
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile("w", delete=False, encoding="utf-8") as fh:
        for line in texts:
            fh.write(line.replace("\n", " ") + "\n")
        corpus_path = fh.name
    prefix_file = NamedTemporaryFile(delete=False)
    model_prefix = prefix_file.name
    prefix_file.close()

    try:
        start = time.perf_counter()
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=int(vocab_size),
            model_type="unigram",
            character_coverage=float(character_coverage),
            bos_id=1,
            eos_id=2,
            unk_id=3,
            pad_id=0,
            hard_vocab_limit=False,
        )
        fit_seconds = time.perf_counter() - start

        model_path = model_prefix + ".model"
        model_bytes = os.path.getsize(model_path)

        sp = spm.SentencePieceProcessor()
        sp.load(model_path)

        total_tokens = 0
        total_chars = sum(len(t) for t in texts)
        encode_start = time.perf_counter()
        for t in texts:
            ids = sp.encode(t, out_type=int)
            total_tokens += len(ids) + 2  # add BOS/EOS
        encode_seconds = time.perf_counter() - encode_start
    finally:
        for ext in ("", ".model", ".vocab"):
            try:
                os.remove(model_prefix + ext)
            except OSError:
                pass
        try:
            os.remove(corpus_path)
        except OSError:
            pass

    tokens_per_second = total_tokens / encode_seconds if encode_seconds > 0 else float("inf")
    tokens_per_char = total_tokens / max(total_chars, 1)
    chars_per_token = total_chars / max(total_tokens, 1)

    return {
        "fit_seconds": fit_seconds,
        "encode_seconds": encode_seconds,
        "tokens_per_second": tokens_per_second,
        "tokens_per_char": tokens_per_char,
        "chars_per_token": chars_per_token,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "vocab_size": int(vocab_size),
        "model_bytes": model_bytes,
    }


def measure_hf_tokenizers(
    texts: List[str],
    *,
    vocab_size: int,
    min_frequency: int,
) -> Dict[str, float]:
    from tokenizers import Tokenizer as HFTokenizer  # type: ignore
    from tokenizers import models, normalizers, pre_tokenizers, trainers

    model = models.BPE(unk_token="[UNK]")
    tokenizer = HFTokenizer(model)
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"],
    )

    start = time.perf_counter()
    tokenizer.train_from_iterator(texts, trainer=trainer)
    fit_seconds = time.perf_counter() - start

    config_bytes = len(tokenizer.to_str().encode("utf-8"))

    total_tokens = 0
    total_chars = sum(len(t) for t in texts)
    encode_start = time.perf_counter()
    for t in texts:
        encoded = tokenizer.encode(t)
        total_tokens += len(encoded.ids) + 2  # add BOS/EOS
    encode_seconds = time.perf_counter() - encode_start

    tokens_per_second = total_tokens / encode_seconds if encode_seconds > 0 else float("inf")
    tokens_per_char = total_tokens / max(total_chars, 1)
    chars_per_token = total_chars / max(total_tokens, 1)

    return {
        "fit_seconds": fit_seconds,
        "encode_seconds": encode_seconds,
        "tokens_per_second": tokens_per_second,
        "tokens_per_char": tokens_per_char,
        "chars_per_token": chars_per_token,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "vocab_size": tokenizer.get_vocab_size(),
        "model_bytes": config_bytes,
    }


def run_measurement(fn) -> Dict[str, float | str]:
    try:
        result = fn()
        result["status"] = "ok"
        return result
    except ImportError as exc:  # dependency missing
        return {"status": "missing_dependency", "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": str(exc)}


def format_row(name: str, data: Dict[str, float | str]) -> str:
    status = data.get("status", "n/a")
    if status != "ok":
        return f"{name:14} | {status}"
    return (
        f"{name:14} | "
        f"fit {data['fit_seconds']:.3f}s | "
        f"enc {data['encode_seconds']:.3f}s | "
        f"tok/s {data['tokens_per_second']:.1f} | "
        f"tok/char {data['tokens_per_char']:.3f} | "
        f"model {data['model_bytes']/1024:.1f} KiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help="Path to a newline-delimited text corpus",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=128,
        help="Replication factor applied to corpus lines for timing",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Target vocabulary size for both backends",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for Hugging Face tokenizers",
    )
    parser.add_argument(
        "--character-coverage",
        type=float,
        default=0.9995,
        help="SentencePiece character coverage",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to store metrics.json (defaults to reports/tokenizers/<ts>)",
    )
    args = parser.parse_args()

    corpus_path = args.corpus.resolve()
    texts = load_texts(corpus_path)
    bench_texts = replicate_texts(texts, max(1, args.repeat))

    sp_result = run_measurement(
        lambda: measure_sentencepiece(
            bench_texts,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
        ),
    )
    hf_result = run_measurement(
        lambda: measure_hf_tokenizers(
            bench_texts,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        ),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (
        args.out_dir if args.out_dir is not None else Path("reports") / "tokenizers" / timestamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "corpus": str(corpus_path),
            "lines": len(texts),
            "repeat": int(args.repeat),
            "vocab_size": int(args.vocab_size),
            "timestamp_utc": timestamp,
        },
        "results": {
            "sentencepiece": sp_result,
            "tokenizers": hf_result,
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote tokenizer comparison to {metrics_path}")
    print(format_row("sentencepiece", sp_result))
    print(format_row("tokenizers", hf_result))


if __name__ == "__main__":
    main()
