PSANN-LM GPU Benchmark Bundle (2025-11-07)
==========================================

Artifacts in this directory back-fill the "GPU block completion" requirements from
`psann_lm_todo.md`. All runs were captured on a dual NVIDIA H200 box (139.7 GB each) running
Python 3.12.3 + PyTorch 2.8.0 (cu128, bf16 enabled). The same system metadata is stored in
`reports/tests/20251107_172133/system.json` and `reports/gpu/20251107_172205/summary.json`.

Throughput Sweep (GPU-03)
------------------------
The `throughput.csv` file aggregates every GPU-03 measurement. The best numbers per base/batch
after the latest sweep are:

| base      | batch_tokens | tokens/sec | timestamp      |
|-----------|--------------|-----------:|----------------|
| respsann  | 131,072      | 614,749.60 | 20251107_015048 |
| respsann  | 262,144      | 613,210.62 | 20251107_015103 |
| waveresnet| 131,072      | 612,637.03 | 20251107_015048 |
| waveresnet| 262,144      | 612,338.01 | 20251107_015103 |

Lower-batch results (20,480 tokens) are also logged for regression tracking, along with the
earlier 65k/131k sweeps (`throughput.csv` retains the entire history).

Gradient-Checkpoint Memory Snapshot (GPU-04)
--------------------------------------------
`memory.json` records the GPU-04 measurement from the same bundle:

- Model: WaveResNet (d_model=512, n_layers=8, bf16)
- Gradient checkpointing: enabled
- `torch.cuda.max_memory_allocated`: 71.76 MiB
- `torch.cuda.max_memory_reserved`: 296.0 MiB
- Wall-clock per forward/backward micro-batch: 0.1049 s

GPU Validation Status (GPU-01..08)
----------------------------------
The corresponding validation run lives at `reports/gpu/20251107_172205/`. Highlights:

- AMP parity: bf16 vs fp32 `rel_diff=1.57e-4`
- Throughput sanity (B=4, T=256) ~299k–302k tok/s across bases
- Gradient checkpointing sample matches the memory snapshot above
- DDP and FSDP both match single-GPU loss (rel_diff=0.0, world_size=2)
- Generation smoke sample: `hqixqtqixqqxxzqxqut`
- Checkpoint save/load parity confirmed (`params_equal=True`, `gen_equal=True`)

Use this README plus the `throughput.csv`/`memory.json` files when citing the GPU numbers in docs
or acceptance notes. The `scripts/run_gpu_validation.py --out reports/gpu` and
`scripts/run_cuda_suite.sh` commands reproduce the same flow end-to-end.
