import pytest
import torch

from psann.embeddings import SineTokenEmbedder
from psann.tokenizer import SimpleWordTokenizer


def test_simple_word_tokenizer_encode_decode_bos_eos_and_unknown():
    corpus = ["Hello world", "hello psann world"]
    tok = SimpleWordTokenizer(lowercase=True)
    tok.fit(corpus)

    ids = tok.encode("HELLO unseen WORLD", add_bos=True, add_eos=True)

    bos_id = tok._tok2id[SimpleWordTokenizer.BOS]  # type: ignore[attr-defined]
    eos_id = tok._tok2id[SimpleWordTokenizer.EOS]  # type: ignore[attr-defined]
    unk_id = tok._tok2id[SimpleWordTokenizer.UNK]  # type: ignore[attr-defined]

    assert ids[0] == bos_id
    assert ids[-1] == eos_id
    assert ids[2] == unk_id

    decoded = tok.decode(ids)
    assert decoded == "hello <UNK> world"
    assert tok.vocab_size >= 4  # specials always present


def test_simple_word_tokenizer_respects_max_vocab_and_case_flag():
    tok = SimpleWordTokenizer(lowercase=False, max_vocab=6)
    tok.fit(["Foo bar baz Foo", "foo BAR"])

    assert tok.vocab_size == 6  # 4 specials + 2 most frequent case-sensitive tokens
    encoded = tok.encode("Foo bar foo BAR baz qux")
    unk_id = tok._tok2id[SimpleWordTokenizer.UNK]  # type: ignore[attr-defined]
    assert encoded.count(unk_id) >= 1
    decoded = tok.decode(encoded)
    # Lower/upper cases preserved since lowercase=False
    tokens = decoded.split()
    assert tokens[0] == "Foo"
    assert "<UNK>" in tokens


def test_sine_token_embedder_requires_vocab_size_before_forward():
    embedder = SineTokenEmbedder(embedding_dim=4)
    with pytest.raises(RuntimeError, match="set_vocab_size"):
        embedder(torch.tensor([0, 1]))


def test_sine_token_embedder_matches_frequency_schedule_and_lazy_table():
    dim = 4
    embedder = SineTokenEmbedder(embedding_dim=dim, base=10000.0, scale=1.0, trainable=False)
    embedder.set_vocab_size(8)

    ids = torch.tensor([0, 1, 5], dtype=torch.long)
    output = embedder(ids)
    assert output.shape == (ids.shape[0], dim)
    assert embedder.A.requires_grad is False

    omega = embedder._frequencies(embedder.A.device)  # type: ignore[attr-defined]
    expected = torch.sin(omega.view(1, -1) * ids.unsqueeze(-1))
    assert torch.allclose(output, expected, atol=1e-5)

    table = embedder.embedding_matrix()
    assert torch.allclose(table[ids], output)


def test_sine_token_embedder_trainable_path_enables_gradients():
    embedder = SineTokenEmbedder(embedding_dim=6, trainable=True)
    embedder.set_vocab_size(4)

    ids = torch.arange(4, dtype=torch.long)
    output = embedder(ids)
    loss = output.sum()
    loss.backward()

    assert embedder.A.requires_grad is True
    assert embedder.A.grad is not None
    assert embedder.phi.grad is not None
    assert embedder.offset.grad is not None
