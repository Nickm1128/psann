# Repo Structure & Conventions

This document defines **what belongs where** in the PSANN repository and where scripts should write outputs. The goal is to keep the repo easy to navigate and prevent large generated artifacts from being committed to git.

## Directory Map

| Path | What it contains | Notes |
| --- | --- | --- |
| `src/psann/` | Core library code shipped by `pip install psann` | Keep this lean and stable; optional integrations under `platform/` must stay lazy at base import time. |
| `psannlm/` | Separate LM tooling distribution | Contains LM training/CLI code and heavier dependencies. |
| `tests/` | Unit + integration tests | Keep “fast” tests default; GPU/slow tests opt-in. |
| `docs/` | Maintained documentation | Link new docs from `docs/README.md`; keep active plans in `docs/backlog/` and historical notes in `docs/archive/`. |
| `scripts/` | Operational CLIs and benchmark runners | Should have `--help`, log config + output directory, and write outputs outside tracked source directories. |
| `examples/` | Runnable examples and config snippets | Prefer small, focused YAML configs and short “how to run” notes. |
| `configs/` | Shared configuration files used by scripts | Prefer stable relative paths; keep configs small and documented. |
| `benchmarks/` | Small, versioned benchmark inputs + benchmark writeups | Keep datasets small; include provenance + regeneration instructions. |
| `datasets/` | Small, versioned fixtures | Mostly ignored by git; see `datasets/README.md`. |
| `notebooks/` | Exploratory notebooks | Keep outputs stripped; prefer `scripts/` + docs for reproducible runs. |
| `tools/` | One-off utilities (data prep, conversions, etc.) | Prefer deterministic, well-documented tools. |
| `deploy/` | Reference serving container definitions | Keep runtime dependencies locked under `constraints/`; mount model artifacts read-only. |

`src/psann/platform/explainability.py` is the optional SHAP orchestration boundary.
Serializable policies live in `explain_contracts.py`, feature games in
`explain_groups.py`, and the frozen differentiable raw-input adapter in
`explain_torch.py`. None of these modules may make SHAP a base-package import.

Phase 7 workplace boundaries live in `platform/accelerators.py` (device/dtype tiers),
`platform/streaming.py` (bounded restartable batches), `platform/operations.py`
(fingerprints, redaction, retention, and hooks), and `platform/performance.py`
(portable benchmark comparisons). `tools/workplace_benchmark.py` writes raw local
observations under ignored `reports/`; only reviewed aggregate baselines belong in
`docs/benchmarks/`.

Phase 8 certification lives in `platform/certification.py` so it can execute from an
installed wheel. `tools/workplace_certification.py` is the source-checkout facade,
`tools/check_public_api.py` enforces the exhaustive current
`docs/workplace_public_api.json` plus the public `docs/public_api_0_12_7.json`
compatibility inventory. The former release-certification, supply-chain security,
and HISSO benchmark workflows are retained in the
[workflow archive](archive/workflows/README.md) and do not run. There is currently no
active GitHub promotion gate; local certification reports are development evidence,
not release approval. Generated artifacts and privacy-safe reports belong under
`reports/certification/`.

## Benchmarks: What Goes Where

Within `src/psann/`, keep `sklearn.py` as the stable public estimator surface and place estimator implementation details under `src/psann/_sklearn/`.
Within `psannlm/` and `scripts/`, keep public CLI files as thin facades when a runner grows large and move the implementation details into nearby internal packages such as `psannlm/_train/` or `scripts/_<tool>/`.

`psannlm/_version.py` owns the LM distribution version and `psannlm/_compat.py`
enforces its declared core-package band. Coordinated releases synchronize the core
and LM version sources, but each installed package reports its own bundled version.

- **Benchmark scripts / runners**: `scripts/` (e.g., `scripts/benchmark_*.py`, `scripts/*_sweep.py`).
- **Benchmark configs**: `examples/` (or `configs/` if the config is used by multiple subsystems).
- **Benchmark inputs that must be versioned** (small): `benchmarks/` + a short README describing provenance.
- **Benchmark outputs**: `reports/` (ignored by git; see below).

If a benchmark needs to ship “golden” reference numbers for regression testing, store a small JSON/CSV under `docs/benchmarks/` and keep it tightly scoped. Follow `docs/benchmarks/promotion_guide.md` when promoting local run outputs.
Do not commit raw `reports/full_suite/` trees; summarize the reusable numbers in `docs/benchmarks/` instead.

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
