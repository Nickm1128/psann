# Benchmark Promotion Guide

Use this guide when a local benchmark run under `reports/`, `runs/`, `outputs/`, or `logs/` needs a checked-in summary under `docs/benchmarks/`.

## Keep raw outputs local

- Do not commit full raw run trees, checkpoints, stdout/stderr captures, profiler traces, or ad hoc plots from `reports/` or `runs/`.
- Treat checked-in benchmark artifacts as compact summaries, not as the source-of-truth run directory.

## What to promote

Promote only the smallest artifact set that preserves the claim:

- one short Markdown summary (`.md`) with the headline result and reproduction command
- one compact table payload (`.json` or `.csv`) containing the metrics used in the summary
- optional tiny static figures only when the table is not sufficient

## Required provenance

Every promoted summary should include:

- command or config used to produce the run
- run date
- git commit if known
- hardware/device context when performance numbers matter
- dependency or package context when it materially affects the result

## Promotion checklist

1. Start from a local run directory under `reports/` or `runs/`.
2. Extract only the metrics and config fields needed for the checked-in summary.
3. Remove raw logs, checkpoints, duplicated per-step traces, and any environment dumps that could contain secrets or machine-specific noise.
4. Write the compact summary into `docs/benchmarks/`.
5. Link the summary from `docs/README.md` if it becomes part of the maintained docs set.
6. Run `python tools/repo_hygiene_audit.py --json` before committing to confirm no raw output trees slipped into git.

## Recommended file shape

- `docs/benchmarks/<topic>.md` for the narrative
- `docs/benchmarks/<topic>.json` or `.csv` for the reproducible compact metrics

When in doubt, prefer less. The local run directory can stay rich; the checked-in artifact should stay small.
