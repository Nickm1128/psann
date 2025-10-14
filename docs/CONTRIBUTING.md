# Contributing Guide

Thanks for helping with the PSANN cleanup. This document captures the house rules while Task 7 (documentation refresh) and the estimator refactors are underway.

## Environment

1. Create a virtual environment and install the project in editable mode:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   # source .venv/bin/activate    # macOS/Linux
   pip install --upgrade pip
   pip install -e .[dev]
   ```
2. The `[dev]` extra installs `pytest`, `ruff`, and `black`. Match the versions in `pyproject.toml`.

## Coding standards

- **Type hints & style** – follow the existing typing patterns. Run `ruff check src tests` and fix any lint failures before sending patches. Black is configured to the default line length.
- **Shared helpers first** – when adding estimator behaviour, reach for `psann.estimators._fit_utils` (e.g., `normalise_fit_args`, `prepare_inputs_and_scaler`, `build_model_from_hooks`) instead of duplicating logic in `sklearn.py`.
- **ASCII-only** edits unless a file already uses Unicode symbols.

## Testing

- Run `pytest` (or the targeted module tests) before and after changes touching training loops or helpers. Extras-focused suites remain skipped while that feature is reworked.
- For documentation-only changes, sanity-check code snippets with `python -m compileall path/to/file.py` when feasible to avoid syntax drift.

## Documentation & task tracking

- Keep `README.md`, `docs/examples/README.md`, and `docs/migration.md` aligned with the code. Mention the reward registry and `transition_penalty` terminology when documenting HISSO flows.
- New docs live under `docs/`. Cross-link notable additions from the README and `pyproject` metadata where practical.
- After each work session, update the relevant section in `CLEANUP_TODO.md` with a short status note and any blockers.

## Pull request checklist

- [ ] Lint (`ruff`) and tests (`pytest`) pass locally.
- [ ] Documentation reflects new behaviour (and points to `docs/migration.md` for edge cases).
- [ ] `CLEANUP_TODO.md` has been updated for the task you touched.
- [ ] Commits include concise summaries and link to the corresponding cleanup task where possible.

Questions? Open a draft PR or drop notes next to the task in `CLEANUP_TODO.md` so the next session can pick up smoothly.
