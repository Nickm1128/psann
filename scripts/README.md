# PSANN Utility Scripts

The utilities under `scripts/` assume the project is installed as a package, so
you no longer need to manually edit `sys.path` when running them.

## Quick Start

1. Create or activate a virtual environment.
2. Install the project in editable mode:
   ```bash
   pip install -e .
   ```
3. Optional: install profiling dependencies as needed (for example,
   `pip install torch torchvision` for GPU runs).

Once the package is available, run scripts directly:

```bash
python scripts/profile_hisso.py --epochs 4
```

Set `CUDA_VISIBLE_DEVICES` or `PYTORCH_CUDA_ALLOC_CONF` if you need to target
specific GPUs; no additional `PYTHONPATH` modifications are required.

## Available Scripts

- `profile_hisso.py` - quick sanity check for the lightweight HISSO trainer with
  a synthetic MLP.
- `benchmark_hisso_variants.py` - benchmarks residual dense vs. convolutional
  HISSO estimators, reporting wall-clock time and reward trends for each device
  (CPU/GPU if available). Supports `--dataset` (`synthetic` or `portfolio`) and
  `--output` to persist JSON summaries for docs/CI:
  ```bash
  python -m scripts.benchmark_hisso_variants --dataset portfolio --epochs 8 --devices cpu --output docs/benchmarks/hisso_variants_portfolio_cpu.json
  ```
- `compare_hisso_benchmarks.py` - compares two benchmark payloads with configurable tolerances; used by CI to detect HISSO performance regressions.
- `fetch_benchmark_data.py` - downloads the trimmed AAPL price series (or any
  other ticker/date range) and writes a `date,open,close` CSV for HISSO runs.
- `run_light_probes.py` - executes the lightweight Colab probes locally. Use
  `--results-dir` to redirect metric dumps away from the repository if desired.
- `run_gpu_validation.py` - unified GPU validation (GPU-01..08). Writes
  timestamped reports under `reports/gpu/`. Accepts `--only GPU-03` etc.
- `run_cuda_suite.sh` - convenience wrapper that sequentially runs `run_cuda_tests.py`,
  `run_gpu_tests.py`, and `run_gpu_validation.py` so a single command exercises the
  full CUDA + GPU validation battery. Emits artifacts under `reports/tests/`,
  `reports/tests/gpu_smoke/`, and `reports/gpu/`.
- `next_gpu_batch.sh` - one-sweep batch runner for RunPod/local GPUs. Runs the
  full validation, throughput sweeps at three batch token sizes, gradient
  checkpoint/memory step, and a tiny-corpus training. Produces benchmark
  artifacts in `reports/benchmarks/<timestamp>/`.
- `aggregate_benchmarks.py` - aggregates GPU validation outputs into
  `throughput.csv` and `memory.json` under a benchmark directory.
- `finalize_bmrk01.py` - builds `metrics.json` for the tiny-corpus benchmark by
  parsing training metrics and computing validation loss/perplexity.
- `parse_trainer_log.py` - parses trainer stdout to `metrics.csv` and, if
  `matplotlib` is available, `loss_curve.png`.
- `make_tiny_corpus.py` - synthesizes a ~50MB `datasets/lm/tiny_books.txt` if a
  real corpus is not available in the pod.
- `run_bmrk01.sh` - one-shot runner for the BMRK-01 tiny-corpus benchmark. Emits
  `metrics.csv`, `metrics.json`, and optionally `loss_curve.png` into
  `reports/benchmarks/<timestamp>/`.

## Current Limitations

- GPU runs depend on local PyTorch CUDA support; the benchmarking script
  auto-skips unavailable devices.
- The tiny-corpus benchmark uses a synthetic corpus by default if
  `datasets/lm/tiny_books.txt` is missing. Replace with a public-domain shard
  for meaningful perplexity numbers.
- The portfolio dataset ships a trimmed AAPL open/close series at
  `benchmarks/hisso_portfolio_prices.csv`; point `--dataset-path` at a custom CSV
  (columns: `open,close,...`) to benchmark other assets.

## TODO

- Expand the benchmarking harness to support custom datasets and export
  structured reports.
