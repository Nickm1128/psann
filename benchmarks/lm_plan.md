PSANN-LM Benchmark Plan
=======================

This document captures the remaining benchmark tasks referenced in `psann_lm_todo.md`.
Each section lists the target inputs, commands, and expected artifacts so we can execute
them quickly once GPUs are available.

Pre-flight Script
-----------------
Use `scripts/next_gpu_batch.sh` to queue the standard validation run plus step-specific
benchmarks. The script wraps:

1. Full GPU validation (`python scripts/run_gpu_validation.py --out reports/gpu`)
2. Throughput-only run (`--only GPU-03`)
3. Gradient-checkpointing/memory run (`--only GPU-04`)
4. Tiny corpus training (`python -m psannlm.lm.train.cli --config examples/lm/configs/tiny_corpus_benchmark.yaml`)

Export `OUT_DIR`, `BENCH_OUT`, or `TINY_CONFIG` to customize destinations before running.

BMRK-01 – Tiny Corpus Baseline
------------------------------
- **Goal:** Fit a ~50 MB text shard end-to-end and record the training + validation curves
  along with a perplexity target.
- **Dataset:** Concatenate any public-domain corpus into `datasets/lm/tiny_books.txt`
  (UTF-8, one line per paragraph). Keep the raw source committed to git-ignore and store
  the processed shard locally.
- **Command:** Use the dedicated YAML config for reproducibility:
  ```
  python -m psannlm.lm.train.cli --config examples/lm/configs/tiny_corpus_benchmark.yaml
  ```
- **Metrics:** Capture
  - Training loss per epoch (CSV + plot).
  - Validation loss + perplexity (target: <= 8.0 on the shard after 2 epochs).
- **Artifacts:** Write to `reports/benchmarks/<timestamp>/`:
  - `loss_curve.png`
  - `metrics.json` with `{train_loss, val_loss, perplexity, corpus_path}`
  - `config_used.yaml`

BMRK-02 – Throughput Table
--------------------------
- **Goal:** Produce a table of tokens/sec for both bases across multiple `batch_tokens`
  settings.
- **Procedure:** Reuse `scripts/run_gpu_validation.py --out reports/gpu --only GPU-03`
  to gather synthetic throughput numbers. Repeat for:
  - Base: `respsann` | `waveresnet`
  - `batch_tokens`: `65_536`, `131_072`, `262_144`
- **Recording:** Collect the `tokens_per_s` entries from `summary.json` and aggregate into
  `reports/benchmarks/<timestamp>/throughput.csv` with columns
  `timestamp,base,batch_tokens,tokens_per_s`.
- **Presentation:** Add a short Markdown table snippet under `reports/benchmarks/<timestamp>/README.md`
  summarizing the best numbers.

BMRK-03 – Memory Snapshot
-------------------------
- **Goal:** Measure the memory impact of `grad_checkpoint=True` under AMP on a medium config.
- **Procedure:**
  - Launch `scripts/run_gpu_validation.py --out reports/gpu --only GPU-04`.
  - Capture `torch.cuda.max_memory_allocated()` before/after the checkpointed run plus elapsed time.
- **Outputs:** Store in `reports/benchmarks/<timestamp>/memory.json`:
  ```json
  {
    "base": "waveresnet",
    "d_model": 512,
    "n_layers": 8,
    "grad_checkpoint": true,
    "amp": "bf16",
    "max_memory_mb": 1234,
    "elapsed_s": 0.061
  }
  ```
- **Comparison:** Include both checkpointed and non-checkpointed entries whenever possible.

BMRK-04 ƒ?" Base Estimator Shootout
---------------------------------
- **Goal:** Compare PSANN-LM base estimators (e.g., `respsann`, `sgrpsann`, `waveresnet`) on a small but
  representative WikiText-103 budget.
- **Command:** Use the quick baseline YAML to keep runs consistent:
  ```
  python scripts/bench_lm_bases.py --config examples/lm/configs/base_compare_quick.yaml
  ```
- **Metrics:** `val_loss`, `val_ppl`, `val_top1_acc`, `train_tokens_per_s`,
  `peak_cuda_mem_gb` (CUDA only).
- **Optional:** Add `--with-lm-eval --lm-eval-tasks lambada_openai,hellaswag --lm-eval-limit 256`
  when lm-eval is available and extra runtime is acceptable.
- **Artifacts:** Stored under `reports/benchmarks/<timestamp>_base_shootout/`:
  - `summary.json` + `summary.csv`
  - `leaderboard.md`
  - `runs/<base>_seed<seed>/metrics.json` (per-base details)

Artifact Layout Summary
-----------------------
```
reports/
  benchmarks/
    YYYYMMDD_HHMMSS/
      loss_curve.png
      metrics.json
      throughput.csv
      memory.json
      README.md          # optional narrative for the run
```

With these outlines in place we can complete the GPU runs as soon as hardware frees up,
without having to reinvent the procedure each time.
