from __future__ import annotations

import torch

from psannlm.lm.models.transformer_geosparse import GeoSparseTransformer, GeoSparseTransformerConfig


def test_geosparse_transformer_forward_shapes() -> None:
    cfg = GeoSparseTransformerConfig(
        vocab_size=101,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_mlp=64,
        positional_encoding="rope",
        geosparse_depth=1,
        geosparse_k=4,
        geosparse_chunk_size=16,
    )
    model = GeoSparseTransformer(cfg)
    inp = torch.randint(0, cfg.vocab_size, (2, 7), dtype=torch.long)
    out = model(inp)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 7, cfg.vocab_size)


def test_geosparse_transformer_kv_cache_shapes() -> None:
    cfg = GeoSparseTransformerConfig(
        vocab_size=97,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_mlp=64,
        positional_encoding="rope",
        geosparse_depth=1,
        geosparse_k=4,
        geosparse_chunk_size=16,
    )
    model = GeoSparseTransformer(cfg)
    inp = torch.randint(0, cfg.vocab_size, (2, 5), dtype=torch.long)
    logits, past = model(inp[:, :3], use_cache=True)
    assert logits.shape == (2, 3, cfg.vocab_size)
    assert isinstance(past, list)
    assert len(past) == cfg.n_layers

    # One more token with cache should extend the KV sequence length by 1.
    logits2, past2 = model(inp[:, 3:4], use_cache=True, past_kvs=past)
    assert logits2.shape == (2, 1, cfg.vocab_size)
    assert len(past2) == cfg.n_layers
    for (k, v), (k2, v2) in zip(past, past2):
        assert k2.size(-2) == k.size(-2) + 1
        assert v2.size(-2) == v.size(-2) + 1


def test_geosparse_transformer_mixed_activation_smoke() -> None:
    cfg = GeoSparseTransformerConfig(
        vocab_size=97,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_mlp=64,
        positional_encoding="rope",
        geosparse_depth=1,
        geosparse_k=4,
        geosparse_chunk_size=16,
        geosparse_activation="mixed",
        geosparse_activation_types=["psann", "relu"],
        geosparse_activation_ratios=[0.5, 0.5],
    )
    model = GeoSparseTransformer(cfg)
    inp = torch.randint(0, cfg.vocab_size, (2, 5), dtype=torch.long)
    out = model(inp)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 5, cfg.vocab_size)
