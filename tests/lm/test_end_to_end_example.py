from __future__ import annotations

from pathlib import Path

from psannlm.lm import psannLM, psannLMDataPrep


SAMPLE_TEXTS = Path(__file__).resolve().parents[2] / "examples" / "lm" / "sample_texts.txt"
TOKENIZER_MODEL = (
    Path(__file__).resolve().parents[2] / "examples" / "lm" / "tokenizer" / "sample_texts.model"
)


def _load_texts() -> list[str]:
    return [
        ln.strip() for ln in SAMPLE_TEXTS.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]


def test_minimal_end_to_end_flow(tmp_path) -> None:
    texts = _load_texts() * 16  # keep runtime tiny but ensure enough tokens
    data = psannLMDataPrep(
        texts,
        tokenizer="sentencepiece",
        tokenizer_model_path=str(TOKENIZER_MODEL),
        max_length=48,
        pack_sequences=True,
        val_split=0.1,
    )
    model = psannLM(
        base="waveresnet",
        d_model=128,
        n_layers=2,
        n_heads=2,
        d_mlp=256,
        vocab_size=data.vocab_size,
    )
    model.fit(data, epochs=3, batch_tokens=256, lr=5e-4, amp="fp32", ddp="off")
    single_out = model.generate("hello world", max_new_tokens=24, temperature=0.0).strip()
    assert single_out, "generation should produce text"

    ckpt_path = tmp_path / "mini_example.pt"
    model.save(str(ckpt_path))

    restored = psannLM.load(str(ckpt_path))
    # Reattach tokenizer for inference convenience
    restored._tokenizer = data.tokenizer  # type: ignore[attr-defined]
    reload_out = restored.generate("wave networks", max_new_tokens=24, temperature=0.0).strip()
    assert reload_out, "restored model should also generate text"
