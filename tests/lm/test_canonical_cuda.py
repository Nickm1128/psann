"""Real CUDA training, cache, generation and cross-device artifact closure."""

from dataclasses import replace

import pytest
import torch

from psann.architectures import GeometryConfig, ResidualConfig, SpectralConfig
from psannlm import LMArchitectureConfig, LMConfig, PSANNLM, PSANNLMDataPrep, TrainConfig
from psannlm.architectures import to_mapping
from psannlm.persistence import load_lm_checkpoint

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize(
    "kind", ["transformer", "residual", "spectral", "wave", "geometric-sparse"]
)
@pytest.mark.parametrize("amp", ["fp32", "fp16", "bf16"])
def test_cuda_train_cached_tokens_gradient_checkpoint_and_two_device_generations(
    kind, amp, tmp_path
):
    data = PSANNLMDataPrep(
        ["abc def ghij klmn opq rst uvw xyz " * 5], tokenizer="simple", max_length=7
    )
    if kind == "spectral":
        architecture = LMArchitectureConfig.residual(
            spectral=SpectralConfig(k_fft=5, init=0.23, strength=0.4)
        )
    elif kind == "wave":
        architecture = LMArchitectureConfig.wave(
            temporal={"mode": "interleave", "kernel_size": 5, "dilation_growth": 2}
        )
    elif kind == "geometric-sparse":
        architecture = LMArchitectureConfig.geometric_sparse(
            geometry=GeometryConfig(shape=(4, 9), k=5, pattern="random", seed=31),
            residual=ResidualConfig(alpha_init=0.61),
            geometry_execution={"depth": 2, "chunk_size": 7},
        )
    else:
        architecture = getattr(LMArchitectureConfig, kind)()
    config = LMConfig(
        architecture,
        d_model=24,
        n_layers=2,
        n_heads=3,
        d_mlp=36,
        vocab_size=data.vocab_size,
        attention_implementation="sdpa",
    )
    torch.manual_seed(131)
    lm = PSANNLM(config=config, device="cuda")
    model = lm._ensure_model(data.vocab_size)
    before = model.lm_head.weight.detach().clone()
    train = TrainConfig(
        steps_per_epoch=3,
        batch_tokens=14,
        lr=0.004,
        warmup_steps=0,
        amp=amp,
        ddp="off",
        grad_checkpoint=True,
        checkpoint_dir=str(tmp_path),
        dataloader_num_workers=0,
    )
    lm.fit(data, train=train)
    assert not torch.equal(model.lm_head.weight, before)
    payload = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=True)
    assert payload["state"]["step"] == 3 and payload["config"] == to_mapping(config)
    assert bool(payload["scaler"]) == (amp == "fp16")
    assert all(v["step"].item() == 3 for v in payload["optim"]["state"].values())
    model.eval()
    tokens = torch.tensor([[1, 4, 7, 10, 13, 16, 19], [2, 5, 8, 11, 14, 17, 20]], device="cuda")
    with torch.no_grad():
        full = model(tokens)
        prefix, cache = model(tokens[:, :-1], use_cache=True)
        last, cache = model(tokens[:, -1:], use_cache=True, past_kvs=cache)
    reference = load_lm_checkpoint(tmp_path / "final.pt").model.eval()
    with torch.no_grad():
        _, reference_cache = reference(tokens[:, :-1].cpu(), use_cache=True)
        reference_last, _ = reference(
            tokens[:, -1:].cpu(), use_cache=True, past_kvs=reference_cache
        )
    torch.testing.assert_close(last.cpu(), reference_last, rtol=3e-4, atol=3e-5)
    if kind in {"spectral", "wave"}:
        # Retained spectral/temporal sequence operations differ between the full
        # sequence and cached continuation; compare the same executed route.
        assert not torch.allclose(last[:, -1], full[:, -1], rtol=2e-4, atol=2e-5)
    else:
        torch.testing.assert_close(last[:, -1], full[:, -1], rtol=2e-4, atol=2e-5)
    assert all(k.shape == v.shape == (2, 3, 7, 8) for k, v in cache)
    torch.manual_seed(773)
    expected = lm.generate("abc", max_new_tokens=13, temperature=0.73, top_p=0.87)
    original_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for generation, destination in [(1, "cpu"), (2, "cuda")]:
        path = tmp_path / f"model{generation}.pt"
        lm.save(path)
        lm = PSANNLM.load(path, map_location=destination)
        assert next(lm._model.parameters()).device.type == destination
        assert to_mapping(lm.config) == to_mapping(config)
        for key, value in lm._model.state_dict().items():
            torch.testing.assert_close(value.cpu(), original_state[key], rtol=0, atol=0)
        lm._model.eval()
        torch.testing.assert_close(
            lm._model(tokens.to(destination)).cpu(), full.cpu(), rtol=3e-4, atol=3e-5
        )
    torch.manual_seed(773)
    assert lm.generate("abc", max_new_tokens=13, temperature=0.73, top_p=0.87) == expected
    continued = replace(train, steps_per_epoch=6)
    lm.fit(data, train=continued, resume_checkpoint=str(tmp_path / "final.pt"))
    resumed = torch.load(tmp_path / "final.pt", map_location="cpu", weights_only=True)
    assert resumed["state"]["step"] == 6
    assert not torch.equal(resumed["model"]["lm_head.weight"], payload["model"]["lm_head.weight"])
    if amp == "fp16":
        assert resumed["scaler"]["_growth_tracker"] == payload["scaler"]["_growth_tracker"] + 3
