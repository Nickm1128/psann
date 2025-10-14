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
  --output docs/benchmarks/hisso_variants_portfolio_cpu.json
```

| Variant | Device | Series Length | Feature Shape | Mean Wall Time (s) | Final Reward |
|---------|--------|---------------|---------------|--------------------|--------------|
| Dense   | CPU    | 506           | `[2]`         | 4.901              | -4.52e-07    |
| Conv    | CPU    | 490           | `[2, 4, 4]`   | 9.783              | -3.47e-06    |

The JSON payload written by the command above (`docs/benchmarks/hisso_variants_portfolio_cpu.json`)
archives the full reward trajectories, episode counts, and timing statistics so future sessions or CI
runs can diff regressions.

GitHub Actions (`.github/workflows/hisso-benchmark.yml`) replays a shorter CPU run on every PR and
compares the output against this snapshot via `scripts.compare_hisso_benchmarks.py`. Adjust the
tolerances or baseline JSON when intentional changes to HISSO performance land.

GPU baselines are still pending; capture them manually once hardware is available and extend the
workflow with an additional job mirroring the CPU flow.
