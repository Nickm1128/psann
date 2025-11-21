import os
import torch

from psann.lm import psannLM, psannLMDataPrep


def test_save_load_roundtrip_and_seeded_generation(tmp_path):
    texts = ["hello world", "goodnight moon", "abc def ghi", "lorem ipsum"]
    dp = psannLMDataPrep(
        texts, tokenizer="simple", max_length=16, pack_sequences=True, val_split=0.0
    )
    lm = psannLM(
        base="respsann", d_model=64, n_layers=2, n_heads=4, vocab_size=dp.vocab_size, rope=True
    )
    lm.fit(dp, epochs=1, batch_tokens=256, lr=1e-3)

    ckpt = os.path.join(tmp_path, "lm.pt")
    lm.save(ckpt)
    assert os.path.exists(ckpt)

    loaded = psannLM.load(ckpt)
    # Compare parameters equal after roundtrip
    assert lm._model is not None and loaded._model is not None
    for (n1, p1), (n2, p2) in zip(
        lm._model.state_dict().items(), loaded._model.state_dict().items()
    ):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)

    # Seeded determinism for sampling on CPU
    prompt = "Once upon a time"
    # Attach tokenizer for loaded instance for generation convenience
    loaded._tokenizer = dp.tokenizer  # type: ignore[attr-defined]

    torch.manual_seed(123)
    out1 = lm.generate(prompt, max_new_tokens=8, top_p=0.9, temperature=0.8)

    torch.manual_seed(123)
    out2 = loaded.generate(prompt, max_new_tokens=8, top_p=0.9, temperature=0.8)

    assert isinstance(out1, str) and isinstance(out2, str)
    assert out1 == out2
