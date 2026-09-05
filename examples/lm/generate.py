"""Minimal generation example for PSANN-LM."""

from psannlm import LMConfig, LMArchitectureConfig, PSANNLM, PSANNLMDataPrep, TrainConfig


def main() -> None:
    texts = ["hello world", "goodnight moon"] * 8
    data = PSANNLMDataPrep(texts, tokenizer="simple", max_length=32)
    model = PSANNLM(
        config=LMConfig(
            architecture=LMArchitectureConfig.wave(),
            d_model=128,
            n_layers=2,
            n_heads=4,
            vocab_size=data.vocab_size,
        ),
        device="cpu",
    )
    model.fit(data, train=TrainConfig(epochs=1, batch_tokens=1024, lr=1e-3))
    out = model.generate("Once upon a time", max_new_tokens=32, top_p=0.9)
    print(out)


if __name__ == "__main__":
    main()
