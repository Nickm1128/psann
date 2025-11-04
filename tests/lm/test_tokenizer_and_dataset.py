import os
import tempfile
import pytest

from psann.lm.data.tokenizer import Tokenizer, TokenizerConfig
from psann.lm.data.dataset import LMDataset, PackingConfig, StreamingLMDataset


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


@pytest.mark.skipif(pytest.importorskip, reason="conditional import check placeholder")
def test_hf_tokenizers_backend_import():
    try:
        import tokenizers  # noqa: F401
    except Exception:
        pytest.skip("tokenizers not installed")
    texts = ["hello tokenizers"]
    tok = Tokenizer(TokenizerConfig(backend="tokenizers", vocab_size=256, min_frequency=1))
    tok.fit(texts)
    ids = tok.encode("hello", add_specials=True)
    assert isinstance(ids, list) and len(ids) > 0

