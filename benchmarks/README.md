# HISSO Benchmark Data

The HISSO examples and docs rely on a compact daily price slice to avoid heavy
dataset downloads at runtime.

- `hisso_portfolio_prices.csv` &mdash; 506 rows (16&nbsp;KiB), Apple Inc. (AAPL)
  open/close prices from 2015-02-17 through 2017-02-16. The CSV keeps only the
  `date`, `open`, and `close` columns to minimise repository weight.

## Provenance and Licensing

Data points were retrieved from Yahoo Finance and remain subject to Yahoo's
terms of service. The trimmed CSV is provided for reproducibility and quick
checks only; redistribute or refresh it according to your own compliance
requirements.

## Regenerating the Dataset

To download a fresh copy (or target a different symbol), run:

```bash
python scripts/fetch_benchmark_data.py \
  --symbol AAPL \
  --start 2015-02-17 \
  --end 2017-02-16 \
  --out benchmarks/hisso_portfolio_prices.csv \
  --overwrite
```

The helper script pulls data directly from Yahoo Finance, trims the columns to
match downstream expectations, and writes the result to the requested path. Use
the `--symbol`, `--start`, and `--end` flags to benchmark other tickers or time
windows.

> Tip: If you prefer to keep benchmark data out of git entirely, add your custom
> CSV path to `.gitignore` and reference it through `--dataset-path` when running
> `scripts/benchmark_hisso_variants.py`.
