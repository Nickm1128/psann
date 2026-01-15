# Repo Structure & Conventions

This document defines **what belongs where** in the PSANN repository and where scripts should write outputs. The goal is to keep the repo easy to navigate and prevent large generated artifacts from being committed to git.

## Directory Map

| Path | What it contains | Notes |
| --- | --- | --- |
| `src/psann/` | Core library code shipped by `pip install psann` | Keep this lean and stable; avoid importing heavyweight optional deps at import time. |
| `psannlm/` | Separate LM tooling distribution | Contains LM training/CLI code and heavier dependencies. |
| `tests/` | Unit + integration tests | Keep “fast” tests default; GPU/slow tests opt-in. |
| `docs/` | Maintained documentation | Link new docs from `docs/README.md`. |
| `scripts/` | Operational CLIs and benchmark runners | Should have `--help`, log config + output directory, and write outputs outside tracked source directories. |
| `examples/` | Runnable examples and config snippets | Prefer small, focused YAML configs and short “how to run” notes. |
| `configs/` | Shared configuration files used by scripts | Prefer stable relative paths; keep configs small and documented. |
| `benchmarks/` | Small, versioned benchmark inputs + benchmark writeups | Keep datasets small; include provenance + regeneration instructions. |
| `datasets/` | Small, versioned fixtures | Mostly ignored by git; see `datasets/README.md`. |
| `notebooks/` | Exploratory notebooks | Keep outputs stripped; prefer `scripts/` + docs for reproducible runs. |
| `tools/` | One-off utilities (data prep, conversions, etc.) | Prefer deterministic, well-documented tools. |

## Benchmarks: What Goes Where

- **Benchmark scripts / runners**: `scripts/` (e.g., `scripts/benchmark_*.py`, `scripts/*_sweep.py`).
- **Benchmark configs**: `examples/` (or `configs/` if the config is used by multiple subsystems).
- **Benchmark inputs that must be versioned** (small): `benchmarks/` + a short README describing provenance.
- **Benchmark outputs**: `reports/` (ignored by git; see below).

If a benchmark needs to ship “golden” reference numbers for regression testing, store a small JSON/CSV under `docs/benchmarks/` and keep it tightly scoped.

## Generated Outputs (Do Not Commit)

These locations are for **generated** artifacts and are intentionally ignored by git:

- `runs/` — training checkpoints, tokenizers, intermediate training artifacts.
- `reports/` — benchmark outputs (tables, plots, JSON summaries).
- `outputs/` — ad-hoc scratch outputs (GPU env reports, quick experiments).
- `eval_data/`, `eval_out/` — local evaluation shards and outputs.
- `artifacts/` — exported model bundles for upload/sharing.
- `logs/` — captured stdout/stderr logs from runs.

If you want to share results, prefer:
- a short summary in `docs/` (with links to reproduction commands), and/or
- attaching the full artifacts to a GitHub release / external storage (S3, HF Hub, etc.).

## Naming Conventions

- **Python**: `snake_case.py` for modules and scripts.
- **YAML configs**: `snake_case.yaml`.
- **Output directories**: timestamped folder names (e.g., `reports/benchmarks/20260115_120000_<slug>/`).
- **Checkpoints**: `ckpt_step000123.pt` (zero-padded) where applicable.

## Script Conventions (New/Updated Scripts)

When adding or updating scripts under `scripts/`:

- Provide `--help` via `argparse` or `typer` and include at least one example invocation in the module docstring.
- Log a compact config header (timestamp, device/dtype, seed, key hyperparams, output dir).
- Prefer `--out` or `--output-dir` for output location (default under `reports/` or `outputs/`).
- Avoid writing into `src/`, `tests/`, or `docs/` as a side effect.
