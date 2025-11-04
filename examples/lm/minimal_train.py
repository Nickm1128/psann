"""Minimal training example for PSANN-LM.

Trains a tiny char-level model on a few toy lines and prints a sample.
"""

from psann.lm import psannLM, psannLMDataPrep


def main() -> None:
    texts = [
        "hello world",
        "goodnight moon",
        "the quick brown fox jumps over the lazy dog",
    ]
    # Keep max_length small for tiny demo texts
    data = psannLMDataPrep(texts, tokenizer="auto", max_length=16)
    model = psannLM(base="waveresnet", d_model=256, n_layers=4, n_heads=4, vocab_size=data.vocab_size)
    print("Data size:", len(data))
    print("Vocab size:", data.vocab_size)
    # CPU-only minimal training (1 epoch)
    model.fit(data, epochs=1, batch_tokens=8192, lr=1e-3)
    print(model.generate("Once upon a time", max_new_tokens=32, top_p=0.9))


if __name__ == "__main__":
    main()
