"""Minimal generation example for PSANN-LM."""

from psannlm.lm import psannLM, psannLMDataPrep


def main() -> None:
    texts = ["hello world", "goodnight moon"]
    data = psannLMDataPrep(texts, tokenizer="simple", max_length=32)
    model = psannLM(base="waveresnet", d_model=128, n_layers=2, n_heads=4, vocab_size=data.vocab_size)
    model.fit(data, epochs=1, batch_tokens=1024, lr=1e-3)
    out = model.generate("Once upon a time", max_new_tokens=32, top_p=0.9)
    print(out)


if __name__ == "__main__":
    main()
