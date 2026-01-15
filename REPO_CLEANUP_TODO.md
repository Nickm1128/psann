# PSANN Repository Cleanup TODO

> Note for Codex / future edits: whenever we discover documentation changes (especially for the README), append concrete notes to the **Docs update backlog** section at the bottom so the README and docs stay comprehensive and in sync with the code.

High-level checklist for cleaning up the repo, tightening the core `psann` package, and separating LM/training code into its own installable distribution.

---

## A. Define the core `psann` surface

- [x] Decide what "core" means: confirm that the public API is exactly what `src/psann/__init__.py` exports (estimators, feature expanders, HISSO episodic tools, utilities) and that this matches README usage examples.
- [x] Audit `src/psann` for non-core code (e.g. LM-specific helpers, experimental utilities, heavy scripts) and either move them under a clear `extras/` or `experimental/` namespace or out of the installable package.
- [x] Verify that `psann.__all__` and the README/API docs are in sync; either add missing documented symbols to `__all__` or document that some modules are "internal".

## B. Slim the `psann` PyPI package

- [x] Change `[tool.hatch.build.targets.wheel].packages` from `["src/psann", "psannlm"]` to just `["src/psann"]` so the core wheel does not ship `psannlm/`.
- [x] Remove `psannlm/**` from `[tool.hatch.build.targets.sdist].include` so the `psann` source distribution only contains the core library.
- [x] Audit `project.dependencies` (currently `numpy`, `torch`) and move anything used only in rarely-used features into optional extras (e.g. `lm`, `viz`) or local imports, so a basic `pip install psann` is as light as feasible.
- [x] Ensure `src/psann` has no hard imports of optional tools (`sentencepiece`, `tokenizers`, `datasets`, Hugging Face, `lm_eval`, etc.); guard those behind extras and runtime checks.

## C. Split LM / training code into a separate installable

- [x] Decide the packaging strategy: e.g. separate project `psannlm` (distribution name) that depends on `psann>=X.Y`.
- [x] Add a new `pyproject.toml` under `psannlm/` defining an LM/training distribution (wheel packages just `["psannlm"]`, with its own dependencies for HF datasets/tokenizers).
- [x] Update imports in `psannlm/train.py` and related files to use `psannlm.lm` (LM APIs now live in `psannlm`).
- [x] Update scripts (`scripts/train_psann_lm.py`, `scripts/runpod_train_300m.sh`, etc.) and docs (`docs/lm.md`) to assume `pip install psann psannlm` for LM training in addition to the core `psann` install.
- [x] Keep LM-related heavy dependencies in the `psannlm` package so `pip install psann` stays light.

## D. Organize scripts, notebooks, and non-library code

- [x] Confirm that all heavy scripts (`scripts/`, `bench_psann_lm.py`, `psann_adapter.py`) remain excluded from wheels/sdists via `[tool.hatch.build.exclude]` (they mostly are already) and that this matches what you want end-users to download.
- [x] Categorize scripts by purpose (benchmarking, GPU validation, training orchestration, release tooling) and document them in `scripts/README.md` so it's obvious what's public vs. internal.
- [x] Move one-off experiment code and quick-and-dirty utilities out of `src/psann` into `examples/` or `notebooks/` where possible (audit complete; `src/psann` now only contains core library modules plus the supported HISSO logging CLI).
- [x] Standardize notebook naming and location (`notebooks/`) and ensure none of them are referenced as importable modules (audit complete; all notebooks live under `notebooks/` and are only referenced from docs/README, not imported as code).

## E. Code quality, consistency, and logical validity

- [x] Run `ruff` and `black` over `src/`, `psannlm/`, and `scripts/` and fix any style/lint issues that indicate real problems (unused imports, dead code paths, overly broad exceptions, etc.). (Ruff F/E9 issues are clean; remaining differences are formatting-only and governed by Black.)
- [x] Add type hints to public APIs in `src/psann` and `psannlm/lm` where feasible, and wire in `mypy` or `pyright` (optionally as a dev extra) to catch interface regressions. (Public surfaces are annotated and `mypy` is now part of the `dev` extra with a baseline `mypy.ini` configuration.)
- [x] Search for duplicated logic between training code (`psann.training`, `psannlm.lm.train`, `psannlm.train`) and refactor shared pieces into reusable utilities to avoid drift. (Audit complete; estimator training and LM training use distinct helpers, with no obvious duplication worth refactoring right now.)
- [x] Verify that versioning is consistent: the `project.version` in `pyproject.toml` and `__version__` in `src/psann/__init__.py` are kept in sync, or configure Hatch's version plugin to make the version single-sourced. (Checked: all are at `0.10.19`, including the new `psannlm/pyproject.toml`; single-sourcing via Hatch remains optional future work.)
- [x] Add or update tests in `tests/` to cover the main estimators, HISSO pipelines, and LM APIs (at least smoke tests that import and run minimal flows), and ensure CI runs them. (`tests/` already contains broad estimator/HISSO coverage plus `tests/lm/` smoke tests for `psannlm` public APIs, trainer, and tokenizer/dataset wiring.)

## F. Docs, README, and user-facing story

- [x] Update `README.md` so the "Getting started" section explicitly distinguishes:
  - Core library: `pip install psann`
  - LM/training add-on: `pip install psann psannlm` (or source install with `pip install -e ./psannlm`).
- [x] Make sure docs under `docs/` referencing LM training or evaluation use the new install story and correct module names (e.g., `docs/lm.md` now includes a PyPI installation section for `psann` + `psannlm`).
- [x] Document optional extras in the README, including what each extra (`dev`, `viz`, `sklearn`, `compat`) pulls in and that LM deps live in `psannlm`.
- [x] Add a short "Architecture / repo layout" section to explain `src/psann`, `src/psann/lm` (stub), `psannlm/`, `scripts/`, `benchmarks/`, `notebooks/`, and how they relate (see the "Package layout and tooling" section in the README).

## G. CI, release, and distribution

- [x] Audit existing GitHub Actions workflows and fix any jobs that currently fail on push (update Python versions, dependencies, or commands as needed). (CI now runs Ruff on functional issues only and Black on the reformatted code; existing jobs are aligned with the current codebase.)
- [x] Update GitHub Actions workflows under `.github/workflows/` to:
  - Build and test the core `psann` wheel.
  - Build and test the new LM wheel (if you split it). (The main CI workflow now builds both `psann` and `psannlm` wheels and runs a dedicated `package-smoke` job.)
- [x] Add release automation (via `scripts/release.py` or Hatch) that:
  - Tags releases consistently.
  - Builds and uploads both distributions (`psann` and LM) to PyPI. (`scripts/release.py` now bumps versions in both `pyproject.toml` files, builds both wheels/sdists, and uploads `dist/*` and `psannlm/dist/*` via Twine.)
- [x] Add a sanity check job that installs `psann` into a fresh env and verifies import of core APIs but not LM training modules, and a separate job for the LM package. (The new `package-smoke` job in `ci.yml` installs only `psann` to confirm core imports while `psannlm` is absent, then installs `psannlm` and checks its CLI integration.)

---

## Docs update backlog

Use this section to record specific documentation/README changes to make as we work through the cleanup:

- [x] Expand the README with a concise "Public API" section that lists the main top-level imports from `psann` (estimators, HISSO helpers, reward utilities, diagnostics, token utilities) so users can see the core surface at a glance.
- [x] Update the README "Language modeling (PSANN-LM)" section so that:
  - It recommends `pip install psann psannlm` for typical users (and uses `pip install -e ./psannlm` for the from-source/development path).
  - It makes clear that the high-level APIs live under `psannlm` (`psannLM`, `psannLMDataPrep`) while the training CLI is provided by the same package.
  - It links prominently to `docs/lm.md` for the full LM reference.
- [x] Add a short note in the README about the separation between the core `psann` package and the LM/training package (`psannlm`), so users understand why the repo hosts both and how the install story works.
- [x] Reconcile README + `docs/API.md` wording about core dependencies with `pyproject.toml` (clarify that the core wheel depends on NumPy and PyTorch, while heavy extras such as LM tokenizers and Hugging Face datasets live behind optional dependencies or the `psannlm` package).
- [x] Add an "Architecture / repo layout" subsection to the README that briefly explains `src/psann` (core), `src/psann/lm` (stub), `psannlm/` (standalone LM package), `scripts/`, `benchmarks/`, and `notebooks/`, and how they relate.
- [x] In `scripts/README.md`, add a short pointer from the main README (e.g. under a "Tooling" or "Scripts" section) so users can easily discover the categorized script documentation without browsing the tree manually.
- [x] In the notebooks README and/or main README, briefly summarize the purpose of the key notebooks (`HISSO_Logging_GPU_Run.ipynb`, `PSANN_Parity_and_Probes.ipynb`, etc.) and reiterate that they are example/analysis artifacts, not importable modules.
