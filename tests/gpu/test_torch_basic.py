import math
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu


@pytest.mark.gpu
def test_cuda_available(torch_available, cuda_available, output_dir: Path):
    if not torch_available:
        pytest.skip("torch not installed")
    import torch

    summary = {
        "torch_version": torch.__version__,
        "cuda_available": bool(cuda_available),
        "device_count": int(torch.cuda.device_count() if cuda_available else 0),
    }
    (output_dir / "test_cuda_available.json").write_text(str(summary))
    if not cuda_available:
        pytest.skip("CUDA not available on this system")


@pytest.mark.gpu
def test_cuda_matmul_matches_cpu(cuda_available, output_dir: Path):
    if not cuda_available:
        pytest.skip("CUDA not available")
    import torch

    torch.manual_seed(42)
    n = 512
    a_cpu = torch.randn((n, n), dtype=torch.float32)
    b_cpu = torch.randn((n, n), dtype=torch.float32)
    c_cpu = a_cpu @ b_cpu

    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    c_gpu = a_gpu @ b_gpu
    c_gpu_cpu = c_gpu.cpu()

    max_abs = (c_cpu - c_gpu_cpu).abs().max().item()
    mean_abs = (c_cpu - c_gpu_cpu).abs().mean().item()

    # CPU/GPU FP32 matmul should match closely
    assert math.isfinite(max_abs)
    assert max_abs < 1e-3
    assert mean_abs < 1e-5

    stats = {"max_abs": max_abs, "mean_abs": mean_abs, "n": n}
    (output_dir / "test_cuda_matmul_matches_cpu.json").write_text(str(stats))


@pytest.mark.gpu
def test_reproducible_random_on_cuda(cuda_available, output_dir: Path):
    if not cuda_available:
        pytest.skip("CUDA not available")
    import torch

    # use generator to ensure per-device determinism
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    a1 = torch.randn((4, 4), generator=gen, device="cuda")
    gen.manual_seed(1234)
    a2 = torch.randn((4, 4), generator=gen, device="cuda")
    assert torch.equal(a1, a2)

    (output_dir / "test_reproducible_random_on_cuda.pt").write_bytes(
        a1.cpu().numpy().tobytes()
    )


@pytest.mark.gpu
def test_amp_fp16_matmul(cuda_available, output_dir: Path):
    if not cuda_available:
        pytest.skip("CUDA not available")
    import torch

    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    y = torch.randn((1024, 1024), device="cuda", dtype=torch.float32)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        z = x @ y
    assert z.is_cuda and z.shape == (1024, 1024)
    (output_dir / "test_amp_fp16_matmul.shape").write_text(str(list(z.shape)))


@pytest.mark.gpu
def test_amp_bf16_if_supported(cuda_available, output_dir: Path):
    if not cuda_available:
        pytest.skip("CUDA not available")
    import torch

    bf16_supported = False
    try:
        bf16_supported = torch.cuda.is_bf16_supported()
    except Exception:
        # fallback heuristic via capability
        try:
            bf16_supported = max(torch.cuda.get_device_capability()) >= 8
        except Exception:
            bf16_supported = False

    if not bf16_supported:
        pytest.skip("BF16 not supported on this GPU/stack")

    x = torch.randn((256, 256), device="cuda", dtype=torch.float32)
    y = torch.randn((256, 256), device="cuda", dtype=torch.float32)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        z = x @ y
    assert z.is_cuda and z.shape == (256, 256)
    (output_dir / "test_amp_bf16_if_supported.shape").write_text(str(list(z.shape)))

