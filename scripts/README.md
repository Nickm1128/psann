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

## Current Limitations

- GPU runs depend on local PyTorch CUDA support; the benchmarking script
  auto-skips unavailable devices.
- The portfolio dataset ships a trimmed AAPL open/close series at
  `benchmarks/hisso_portfolio_prices.csv`; point `--dataset-path` at a custom CSV
  (columns: `open,close,...`) to benchmark other assets.

## TODO

- Expand the benchmarking harness to support custom datasets and export
  structured reports.

