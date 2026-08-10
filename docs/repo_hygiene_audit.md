# Repo Hygiene Audit

Use this note when you want a quick read on whether the repository still matches its cleanup rules.

## Run the audit

From repo root:

```bash
python tools/repo_hygiene_audit.py
```

Add `--json` for machine-readable output. CI and `make hygiene` also pass
`--strict-long-files`, making the 800-line threshold blocking.

## What the audit checks

- tracked generated outputs under `reports/`, `runs/`, `outputs/`, or `logs/` (except the directory READMEs)
- root-level `test_outputs.txt`
- zipped benchmark bundles under `benchmarks/`
- committed notebook outputs, execution counts, widget state, and invalid notebook JSON
- tracked top-level files not present in the reviewed allowlist
- Python files at or above the default long-file threshold (`800` lines)
- missing tracked files in a dirty worktree are skipped when counting long Python files so cleanup branches can still run the audit mid-refactor

## Current expectations

- Raw experiment outputs stay local under `reports/`, `runs/`, `outputs/`, or `logs/`.
- Tracked notebooks contain source only; `nbstripout` runs before the local quality
  and hygiene pre-commit hooks.
- New top-level files require an explicit repository-level purpose. Prefer the owning
  package, `tools/`, `scripts/`, or `docs/`.
- If benchmark results need to be versioned, promote a compact JSON/CSV summary to `docs/benchmarks/`.
- Treat long Python files as a queue for modularization. For core library modules, extract helpers into nearby modules only when the split keeps the public surface clearer. For scripts, extract reusable loaders/report writers once a script starts mixing unrelated concerns.

## Current long-file state

Phase 1 split runtime helpers from `psannlm/lm/train/trainer.py` and CLI parsing from
`scripts/_bench_lm_bases/main.py`. The strict audit currently reports no Python file at
or above 800 lines.

The largest files remain visible in the JSON report as planning input, but they should
only be split when a clear responsibility boundary exists.
