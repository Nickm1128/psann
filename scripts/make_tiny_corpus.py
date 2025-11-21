"""Create a local tiny text corpus if missing.

By default writes datasets/lm/tiny_books.txt with ~50MB of synthetic paragraphs
composed from a small seed set. This avoids external downloads on pods without
network access.

Usage:
  python scripts/make_tiny_corpus.py --out datasets/lm/tiny_books.txt --mb 50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


SEED_PARAS = [
    "Once upon a time there was a tiny model that learned to predict the next token.",
    "The quick brown fox jumps over the lazy dog in the moonlight.",
    "WaveResNet residual paths whisper as sine activations sing in the night.",
    "Packing sequences boosts throughput for language modeling on long documents.",
    "Parameter-efficient designs often trade depth for width to match memory budgets.",
    "Autoregressive decoders require careful KV-cache handling for fast generation.",
    "Distributed data parallel can be made deterministic with careful seeding.",
    "Tokenizer choice affects fragmentation, vocabulary size, and downstream quality.",
    "Gradient checkpointing saves memory at the cost of additional compute.",
    "Hello world from the PSANN-LM tiny corpus generator.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="datasets/lm/tiny_books.txt")
    ap.add_argument("--mb", type=int, default=50, help="Target size in megabytes (approx)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    target_bytes = int(args.mb) * 1024 * 1024
    written = 0
    with outp.open("w", encoding="utf-8") as fh:
        while written < target_bytes:
            base = random.choice(SEED_PARAS)
            # Jitter a bit by shuffling words and appending small variations
            words = base.split()
            random.shuffle(words)
            extra = " ".join(random.choice(SEED_PARAS).split()[: random.randint(4, 10)])
            para = " ".join(words) + ". " + extra + ".\n"
            fh.write(para)
            written += len(para.encode("utf-8"))

    print(f"Wrote ~{written/ (1024*1024):.1f} MB to {outp}")


if __name__ == "__main__":
    main()
