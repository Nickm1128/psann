import importlib.util
import os
import tempfile
import pytest

from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psannlm.lm.data.dataset import LMDataset, PackingConfig, StreamingLMDataset


def test_simple_tokenizer_roundtrip():
    texts = ["hello", "world"]
    tok = Tokenizer(TokenizerConfig(backend="simple"))
    tok.fit(texts)
    ids = tok.encode("hello", add_specials=True)
    s = tok.decode(ids, skip_specials=True)
    assert "hello" in s


def test_lm_dataset_packing_and_shapes():
    texts = ["abcde", "fghij"]
    tok = Tokenizer(TokenizerConfig(backend="simple"))
    tok.fit(texts)
    ds = LMDataset(texts, tok, PackingConfig(max_length=4, pack_sequences=True))
    assert len(ds) > 0
    sample = ds[0]
    assert tuple(sample["input_ids"].shape) == (4,)
    assert tuple(sample["labels"].shape) == (4,)


def test_lm_dataset_without_packing_preserves_doc_boundaries():
    texts = [
        "abcdefghij klmnop",
        "qrstuvwx yzabcdef",
    ]
    tok = Tokenizer(TokenizerConfig(backend="simple"))
    tok.fit(texts)
    packed = LMDataset(texts, tok, PackingConfig(max_length=4, pack_sequences=True))
    unpacked = LMDataset(texts, tok, PackingConfig(max_length=4, pack_sequences=False))
    assert len(packed) > 0 and len(unpacked) > 0

    def has_mid_sequence_bos(ds):
        for sample in ds:
            # reconstruct the original (T+1) chunk
            chunk = sample["input_ids"].tolist()
            chunk.append(int(sample["labels"][-1]))
            if tok.bos_id in chunk[1:]:
                return True
        return False

    # Packing joins docs so BOS tokens can appear mid-chunk; non-packing should not.
    assert has_mid_sequence_bos(packed) is True
    assert has_mid_sequence_bos(unpacked) is False


def test_streaming_lm_dataset(tmp_path: tempfile.TemporaryDirectory):
    p = os.path.join(tmp_path, "text.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("hello world\n")
        fh.write("goodnight moon\n")
    tok = Tokenizer(TokenizerConfig(backend="simple"))
    tok.fit(["hello world", "goodnight moon"])  # fitting corpus
    ds = StreamingLMDataset([p], tok, PackingConfig(max_length=6), shuffle_docs=False)
    it = iter(ds)
    ex = next(it)
    assert tuple(ex["input_ids"].shape) == (6,)
    assert tuple(ex["labels"].shape) == (6,)


@pytest.mark.skipif(
    importlib.util.find_spec("sentencepiece") is None,
    reason="sentencepiece not installed",
)
def test_tokenizer_auto_prefers_sentencepiece_when_available():
    texts = ["hello world", "goodnight moon"]
    cfg = TokenizerConfig(
        backend="auto",
        vocab_size=64,
        sp_character_coverage=1.0,
        sp_input_sentence_size=0,
    )
    tok = Tokenizer(cfg)
    tok.fit(texts)
    assert tok.backend_name == "sentencepiece"
    ids = tok.encode("hello", add_specials=True)
    assert isinstance(ids, list) and len(ids) > 0


def test_tokenizer_auto_falls_back_to_hf_when_sentencepiece_missing(monkeypatch):
    def _boom(*_, **__):
        raise ImportError("sentencepiece missing for test")

    monkeypatch.setattr("psannlm.lm.data.tokenizer._make_sentencepiece_tokenizer", _boom)
    texts = ["auto fallback is healthy"]
    tok = Tokenizer(TokenizerConfig(backend="auto", vocab_size=128, min_frequency=1))
    tok.fit(texts)
    assert tok.backend_name == "tokenizers"


def test_tokenizer_auto_falls_back_to_simple_when_no_external(monkeypatch):
    def _boom(*_, **__):
        raise ImportError("missing dependency")

    monkeypatch.setattr("psannlm.lm.data.tokenizer._make_sentencepiece_tokenizer", _boom)
    monkeypatch.setattr("psannlm.lm.data.tokenizer._make_hf_tokenizers", _boom)

    tok = Tokenizer(TokenizerConfig(backend="auto"))
    tok.fit(["chars only"])
    assert tok.backend_name == "simple"
    ids = tok.encode("chars only")
    assert len(ids) > 0


@pytest.mark.skipif(
    importlib.util.find_spec("tokenizers") is None,
    reason="tokenizers not installed",
)
def test_hf_tokenizers_backend_import():
    texts = ["hello tokenizers"]
    tok = Tokenizer(TokenizerConfig(backend="tokenizers", vocab_size=256, min_frequency=1))
    tok.fit(texts)
    ids = tok.encode("hello", add_specials=True)
    assert isinstance(ids, list) and len(ids) > 0
