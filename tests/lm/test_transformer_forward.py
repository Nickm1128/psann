import torch

from psann.lm.models.registry import get_base


def _tiny_model(base: str, vocab_size: int = 64):
    factory = get_base(base)
    return factory(vocab_size=vocab_size, d_model=64, n_layers=2, n_heads=4, d_mlp=128, dropout=0.0, rope=True)


def test_forward_shapes_respsann():
    model = _tiny_model("respsann", 64)
    x = torch.randint(0, 64, (2, 5), dtype=torch.long)
    y = model(x)
    assert y.shape == (2, 5, 64)


def test_forward_shapes_waveresnet():
    model = _tiny_model("waveresnet", 64)
    x = torch.randint(0, 64, (2, 5), dtype=torch.long)
    y = model(x)
    assert y.shape == (2, 5, 64)


def test_kv_cache_step_waveresnet():
    model = _tiny_model("waveresnet", 64)
    x = torch.randint(0, 64, (2, 4), dtype=torch.long)
    logits, past = model(x, use_cache=True)
    assert logits.shape == (2, 4, 64)
    step_tok = torch.randint(0, 64, (2, 1), dtype=torch.long)
    logits2, past2 = model(step_tok, use_cache=True, past_kvs=past)
    assert logits2.shape == (2, 1, 64)


def test_waveresnet_wave_interleave_forward():
    from psann.lm.models.transformer_waveresnet import build_waveresnet_transformer

    model = build_waveresnet_transformer(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_mlp=128,
        rope=True,
        wave_interleave=True,
        wave_kernel_size=3,
        wave_dilation_growth=1,
    )
    x = torch.randint(0, 64, (2, 5), dtype=torch.long)
    y = model(x)
    assert y.shape == (2, 5, 64)


def test_waveresnet_wave_replace_forward():
    from psann.lm.models.transformer_waveresnet import build_waveresnet_transformer

    model = build_waveresnet_transformer(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_mlp=128,
        rope=True,
        wave_replace=True,
        wave_kernel_size=3,
        wave_dilation_growth=1,
    )
    x = torch.randint(0, 64, (2, 5), dtype=torch.long)
    y = model(x)
    assert y.shape == (2, 5, 64)


def test_kv_cache_step_waveresnet_with_wave():
    from psann.lm.models.transformer_waveresnet import build_waveresnet_transformer

    model = build_waveresnet_transformer(
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_mlp=128,
        rope=True,
        wave_interleave=True,
        wave_kernel_size=3,
        wave_dilation_growth=1,
    )
    x = torch.randint(0, 64, (2, 4), dtype=torch.long)
    logits, past = model(x, use_cache=True)
    assert logits.shape == (2, 4, 64)
    step_tok = torch.randint(0, 64, (2, 1), dtype=torch.long)
    logits2, past2 = model(step_tok, use_cache=True, past_kvs=past)
    assert logits2.shape == (2, 1, 64)
