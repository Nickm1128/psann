## HISSO Variant Benchmarks

Baseline runs captured the residual dense vs. convolutional HISSO estimators on the trimmed
portfolio dataset (`benchmarks/hisso_portfolio_prices.csv`, AAPL open/close log returns).
Command used:

```bash
python -m scripts.benchmark_hisso_variants \
  --dataset portfolio \
  --epochs 4 \
  --devices cpu \
  --variants dense,conv \
  --modes compat,fast \
  --compat-batch-episodes 1 \
  --compat-updates-per-epoch 32 \
  --fast-batch-episodes 8 \
  --fast-updates-per-epoch 4 \
  --output docs/benchmarks/hisso_variants_portfolio_cpu.json
```

| Mode   | Variant | Device | B/U | Series Length | Feature Shape | Mean Wall Time (s) | Wall Throughput (eps/s) | Profile Throughput (eps/s) | Final Reward |
|--------|---------|--------|-----|---------------|---------------|--------------------|-------------------------|----------------------------|--------------|
| compat | Dense   | CPU    | 1/32 | 506         | `[2]`         | 2.457              | 52.10                   | 124.32                     | -4.52e-07    |
| compat | Conv    | CPU    | 1/32 | 490         | `[2, 4, 4]`   | 2.628              | 48.71                   | 57.78                      | -3.47e-06    |
| fast   | Dense   | CPU    | 8/4  | 506         | `[2]`         | 0.240              | 533.84                  | 745.53                     | -3.95e-02    |
| fast   | Conv    | CPU    | 8/4  | 490         | `[2, 4, 4]`   | 1.571              | 81.47                   | 94.58                      | -6.25e-04    |

The JSON payload written by the command above (`docs/benchmarks/hisso_variants_portfolio_cpu.json`)
archives the full reward trajectories, episode counts, schedule metadata (`mode`,
`configured_batch_episodes`, `configured_updates_per_epoch`), and timing/throughput statistics so
future sessions or CI runs can diff regressions.

GitHub Actions (`.github/workflows/hisso-benchmark.yml`) replays a shorter CPU run on every PR and
compares the output against this snapshot via `scripts.compare_hisso_benchmarks.py`:
- `compat` mode is a blocking stability check.
- `fast` mode is tracked as a non-blocking trend signal.

Adjust tolerances or refresh the baseline JSON when intentional HISSO performance changes land.

GPU baselines are still pending; capture them manually once hardware is available and extend the
workflow with an additional job mirroring the CPU flow.
