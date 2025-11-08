"""Minimal end-to-end training example for PSANN-LM.

This script:
1. Loads `examples/lm/sample_texts.txt`.
2. Prepares the dataset/tokenizer via `psannLMDataPrep`.
3. Trains a tiny WaveResNet transformer on CPU.
4. Generates a few completions and optionally stores them under `reports/examples/`.

Run:
    python examples/lm/minimal_train.py --out reports/examples/<run_dir>
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json

from psann.lm import psannLM, psannLMDataPrep

PROMPTS = [
    "hello world",
    "wave networks",
    "sine activations",
]
TOKENIZER_MODEL = Path(__file__).with_name("tokenizer").joinpath("sample_texts.model")


def _load_corpus() -> tuple[list[str], Path]:
    corpus_path = Path(__file__).with_name("sample_texts.txt")
    texts = [
        line.strip()
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return texts, corpus_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train + generate with psannLM on a toy corpus.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional directory to store metadata + generation samples (created if missing).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Epochs for the tiny training run (default: 12).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=64,
        help="How many times to repeat the sample corpus to create a few thousand tokens (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_texts, corpus_path = _load_corpus()
    repeats = max(1, int(args.repeat))
    texts = base_texts * repeats
    data = psannLMDataPrep(
        texts,
        tokenizer="sentencepiece",
        tokenizer_model_path=str(TOKENIZER_MODEL),
        max_length=64,
        pack_sequences=True,
        val_split=0.2,
    )
    model = psannLM(
        base="waveresnet",
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_mlp=1024,
        vocab_size=data.vocab_size,
    )
    print(f"Loaded {len(base_texts)} unique documents from {corpus_path.name} (repeat x{repeats} -> {len(texts)} total)")
    print(f"Dataset batches: {len(data)} (max_length={data.max_length}, pack_sequences={data.pack_sequences})")
    print(f"Tokenizer model: {TOKENIZER_MODEL}")
    print(f"Vocab size: {data.vocab_size}")

    train_cfg = dict(epochs=args.epochs, batch_tokens=512, lr=2e-4, amp="fp32", ddp="off")
    print(f"Training config: {train_cfg}")
    model.fit(data, **train_cfg)

    generations = {}
    print("\n=== Sample generations ===")
    for prompt in PROMPTS:
        completion = model.generate(
            prompt,
            max_new_tokens=48,
            top_k=None,
            top_p=0.6,
            temperature=0.0,
        ).strip()
        generations[prompt] = completion
        print(f"[prompt] {prompt!r}")
        print(f"[output] {completion}\n")

    out_dir = args.out
    if out_dir is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("reports/examples") / f"{timestamp}_minimal_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus": str(corpus_path),
        "unique_docs": len(base_texts),
        "repeat_factor": repeats,
        "num_texts": len(texts),
        "train_config": train_cfg,
        "prompts": PROMPTS,
        "generations": generations,
    }
    (out_dir / "generations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                f"# Minimal Train Run ({datetime.utcnow().isoformat(timespec='seconds')}Z)",
                "",
                f"- Corpus: {corpus_path}",
                f"- Repeat factor: {repeats}",
                f"- Epochs: {train_cfg['epochs']}",
                f"- batch_tokens: {train_cfg['batch_tokens']}",
                "",
                "See `generations.json` for prompts and outputs.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved generation samples to: {out_dir}")


if __name__ == "__main__":
    main()
