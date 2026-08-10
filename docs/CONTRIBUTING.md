# Contributing Guide

Thanks for helping with the PSANN cleanup. This document captures the house rules while Task 7 (documentation refresh) and the estimator refactors are underway.

## Environment

1. Preferred bootstrap from repo root:
   ```bash
   make dev
   ```
   `make dev` now uses the virtualenv's Python on both Windows and Unix-like systems, installs `psann` with `[dev]`, installs the local `psannlm` package, and enables pre-commit hooks.
2. Manual bootstrap if `make` is unavailable:
   Windows PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -e .[dev]
   python -m pip install -e ./psannlm
   python -m pre_commit install
   ```
   macOS/Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e .[dev]
   python -m pip install -e ./psannlm
   python -m pre_commit install
   ```
3. Use Python 3.11, 3.12, or 3.13. The `[dev]` extra installs the complete quality,
   coverage, build, notebook, and pre-commit toolchain.
4. Optional: enable pre-commit hooks for formatting and linting if you skipped `make dev`:
   ```bash
   python -m pre_commit install
   ```

## Coding standards

- **Type hints & style** - Black is the only formatter. Ruff owns lint and import
  ordering; do not run Ruff's formatter. Use `python tools/quality.py format` and then
  `python tools/quality.py lint`.
- **Typed boundaries** - new workplace modules under `psann.platform` are checked
  strictly. Legacy internals remain under a documented gradual mypy scope.
- **Operational safety** - never place credentials or raw workplace data in fixtures,
  artifact metadata, model cards, logs, benchmark summaries, or committed reports.
  Add new device/dtype claims only with scheduled evidence and update
  `docs/workplace_operations.md`.
- **Shared helpers first** – when adding estimator behaviour, reach for `psann.estimators._fit_utils` (e.g., `normalise_fit_args`, `prepare_inputs_and_scaler`, `build_model_from_hooks`) instead of duplicating logic in `sklearn.py`.
- **ASCII-only** edits unless a file already uses Unicode symbols.

## Common commands

From repo root:

```bash
make dev        # bootstrap venv + install deps + pre-commit
make lint       # ruff + black + mypy
make test-fast  # pytest (exclude slow + GPU)
make coverage   # one test run, three scoped coverage reports
make hygiene    # generated output, notebook, root-file, and long-file checks
make build      # build both wheels
make package-smoke  # install built wheels into a temporary environment
```

## Testing

- Run `pytest` (or the targeted module tests) before and after changes touching training loops or helpers. Extras-focused suites remain skipped while that feature is reworked.
- For documentation-only changes, sanity-check code snippets with `python -m compileall path/to/file.py` when feasible to avoid syntax drift.
- Mark long-running or GPU/HISSO tests with `@pytest.mark.slow`, and keep quick iterations to CPU by running `python -m pytest -m "not slow"`.

## Documentation & task tracking

- Keep `README.md`, `docs/examples/README.md`, and `docs/migration.md` aligned with the code. Mention the reward registry and `transition_penalty` terminology when documenting HISSO flows.
- New docs live under `docs/`. Cross-link notable additions from the README and `pyproject` metadata where practical.
- Route roadmap notes through `docs/backlog/` or `docs/archive/` instead of dropping new TODO files at repo root.
- Use `docs/benchmarks/promotion_guide.md` when turning local run outputs into checked-in benchmark summaries, and keep alias terminology aligned with `docs/deprecation_policy.md`.
- After each work session, update the relevant section in `docs/project_cleanup_todo.md` with a short status note and any blockers.
- For new model bases, benchmarks, or datasets, follow `docs/how_to_add_model_benchmark_dataset.md`.

## Pull request checklist

- [ ] `make lint`, `make test-fast`, and `make hygiene` pass locally.
- [ ] Documentation reflects new behaviour (and points to `docs/migration.md` for edge cases).
- [ ] `docs/project_cleanup_todo.md` has been updated for the task you touched.
- [ ] Commits include concise summaries and link to the corresponding cleanup task where possible.

Questions? Open a draft PR or drop notes next to the task in `docs/project_cleanup_todo.md` so the next session can pick up smoothly.
