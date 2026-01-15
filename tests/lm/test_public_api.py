from psannlm.lm import psannLM, psannLMDataPrep


def test_public_api_fit_generate_and_checkpoint_roundtrip(tmp_path):
    texts = ["hello world", "goodnight moon"]
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=8, pack_sequences=True, val_split=0.0
    )
    model = psannLM(
        base="waveresnet",
        d_model=64,
        n_layers=2,
        n_heads=4,
        vocab_size=dp.vocab_size,
        positional_encoding="rope",
    )

    model.fit(dp, epochs=1, batch_tokens=512, lr=1e-3, amp="fp32")

    single = model.generate("hello", max_new_tokens=4, top_p=0.9)
    batch = model.generate_batch(["hello", "moon"], max_new_tokens=4, top_p=0.9)

    assert isinstance(single, str)
    assert isinstance(batch, list) and len(batch) == 2
    assert all(isinstance(item, str) for item in batch)

    ckpt_path = tmp_path / "public_api.pt"
    model.save(str(ckpt_path))
    loaded = psannLM.load(str(ckpt_path))
    assert loaded.base == "waveresnet"
    assert loaded.positional_encoding == model.positional_encoding
