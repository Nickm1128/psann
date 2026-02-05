# reports/

This directory is for **generated benchmark outputs** (tables, plots, JSON summaries, CSVs).

- This folder is ignored by git by default, except `reports/full_suite/**` (used by the one-command
  experiment runner) and this README.
- Create a new timestamped subdirectory per run, e.g. `reports/full_suite/20260115_120000_my_sweep/`.
- If you need versioned "golden" results for regression testing, store a small JSON/CSV under
  `docs/benchmarks/` instead.
