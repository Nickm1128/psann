# Repo Hygiene Audit

Use this note when you want a quick read on whether the repository still matches its cleanup rules.

## Run the audit

From repo root:

```bash
python tools/repo_hygiene_audit.py
```

Add `--json` if you want machine-readable output.

## What the audit checks

- tracked generated outputs under `reports/`, `runs/`, `outputs/`, or `logs/` (except the directory READMEs)
- root-level `test_outputs.txt`
- zipped benchmark bundles under `benchmarks/`
- Python files above the default long-file threshold (`800` lines), reported as refactor targets rather than hard failures
- missing tracked files in a dirty worktree are skipped when counting long Python files so cleanup branches can still run the audit mid-refactor

## Current expectations

- Raw experiment outputs stay local under `reports/`, `runs/`, `outputs/`, or `logs/`.
- If benchmark results need to be versioned, promote a compact JSON/CSV summary to `docs/benchmarks/`.
- Treat long Python files as a queue for modularization. For core library modules, extract helpers into nearby modules only when the split keeps the public surface clearer. For scripts, extract reusable loaders/report writers once a script starts mixing unrelated concerns.

## Known hotspots

The audit intentionally reports long files without failing the run. Waves 2, 3, and 4 resolved the earlier `_fit_utils.py`, `hisso.py`, `lsm.py`, `test_hisso_primary.py`, `sklearn.py`, `psannlm/train.py`, and the large benchmark-script hotspots. The remaining major refactor target is:

- `psannlm/lm/train/trainer.py`

Treat that file as planning input for the next hygiene pass rather than a blind split target.
