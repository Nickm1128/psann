# Performance Tips

This page collects quick, practical tips for getting reliable performance on CPU and GPU. It is intentionally brief and focused on the common pain points.

## CPU runs

- Use smaller `batch_size` values (32-128) for quick iteration.
- Prefer `float32` NumPy inputs to avoid extra casts.
- Start with `examples/01_basic_regression.py` to validate the environment.

## GPU runs

- Install a CUDA-enabled PyTorch build via the official selector:
  - https://pytorch.org/get-started/locally/
- PSANN does not install CUDA wheels for you; install PyTorch first, then `pip install psann`.
- Enable TF32 for matrix-heavy runs when accuracy allows:
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`

## Mixed precision

- For modern GPUs, prefer BF16 (`amp="bf16"` in LM scripts or `--amp --amp-dtype bfloat16` in benchmarks).
- If BF16 is unavailable, FP16 can work but may require smaller learning rates.

## torch.compile

- `torch.compile` can speed up steady-state training but adds compile overhead.
- Use it for long runs or sweeps; disable it for quick debugging.
- Keep shapes static where possible to avoid graph breaks.

## Memory and allocator tuning

- If GPU memory fragmentation is high, try:
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- For recent PyTorch builds, you can opt into the async allocator:
  - `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync`
- Prefer leaving headroom for other processes; the LM trainer supports
  `--cuda-memory-fraction` to cap VRAM usage per process.
- Prefer gradient checkpointing when memory is tight (trade throughput for memory).

## Data handling

- Use `num_workers > 0` when dataset preprocessing is CPU-heavy.
- Avoid large eval shards inside the training loop unless you explicitly want frequent validation passes.

## Reproducibility

- Set `random_state` on estimators and call `seed_all(...)` for script-level tests.
- Keep `torch.backends.cudnn.deterministic = True` for strict determinism (at the cost of speed).

## GPU generation notes

- **Ampere (A100/RTX 30)**: TF32 is usually safe for training; BF16 is supported on most SKUs.
- **Hopper (H100)**: BF16 + `torch.compile` tends to deliver strong throughput; keep shapes static.
- **Blackwell (B100/GB10)**: prefer CUDA 13 + BF16, and use SDPA/FlashAttention when available.

## Profiling and microbenching

- Use `scripts/profile_psann.py` to capture a chrome trace and a summary table:
  - `python scripts/profile_psann.py --model psann --device cuda --out reports/profiles/psann_gpu`
- Use `scripts/microbench_psann.py` to record throughput and memory for PSANN vs baselines:
  - `python scripts/microbench_psann.py --device cpu --out reports/benchmarks/microbench_cpu.json`
- Use `scripts/benchmark_geo_sparse_micro.py` to compare GeoSparse compute modes:
  - `python scripts/benchmark_geo_sparse_micro.py --device cuda --compute-mode auto`
