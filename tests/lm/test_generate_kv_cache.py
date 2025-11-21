
from psann.lm import psannLM, psannLMDataPrep


def test_generate_batch_kv_cache_cpu():
    # Tiny char-level model; CPU
    texts = ["hello world", "goodnight moon", "abc def ghi", "lorem ipsum"]
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=16, pack_sequences=True, val_split=0.0
    )
    model = psannLM(
        base="waveresnet", d_model=64, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True
    )
    model.fit(dp, epochs=1, batch_tokens=256, lr=1e-3)
    outs = model.generate_batch(["hello", "goodnight"], max_new_tokens=8, top_p=0.9)
    assert isinstance(outs, list) and len(outs) == 2
    assert all(isinstance(o, str) for o in outs)
