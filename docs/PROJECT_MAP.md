# PSANN Project Map (Start Here)

PSANN is a research-driven PyTorch project that exposes sine-activated neural network variants behind a sklearn-style estimator API (plus a growing set of experimental architectures and benchmarking tools).

This document explains what PSANN *is*, what is considered *stable*, what is *experimental*, and how the repository is organized so new contributors can navigate it quickly.

---

## Who This Repo Is For

- **Practitioners** who want a drop-in sklearn-style regressor backed by PyTorch (CPU or GPU).
- **Researchers** exploring sine activations, sparse topologies (GeoSparse), and training dynamics.
- **Contributors** who want to extend model backbones and benchmark them reproducibly.

---

## What’s Supported vs Experimental

### Supported (Core)

These are intended to be stable and documented, and they are covered by the main test suite.

- **Sklearn-style estimators**
  - `PSANNRegressor`
  - `ResPSANNRegressor`
  - `ResConvPSANNRegressor`
  - `WaveResNetRegressor`
  - `SGRPSANNRegressor`
- **Training utilities** shared by those estimators
  - Data preparation (shape handling, scaling) in `psann.estimators._fit_utils`
  - Supervised training loop utilities and stateful streaming utilities
- **HISSO / episodic training**
  - HISSO training utilities and reward strategies used for episodic optimisation

### Experimental (Research)

These are under active iteration; APIs may change and performance characteristics are still being studied.

- **GeoSparse**
  - Torch backbone: `psann.nn_geo_sparse`
  - Estimator wrapper: `GeoSparseRegressor`
  - Benchmark scripts and sweep harnesses under `scripts/` and `reports/`
- **Language modeling**
  - Core LM library code lives in the separate `psannlm` distribution
  - Training/CLI utilities live in `psannlm.lm.train.cli` (see below)

If you depend on any experimental pieces, pin a version and expect breaking changes across minor releases.

---

## Installation Model (Current + Intended Direction)

### Current (today)

- `pip install psann` installs the core package and its runtime dependencies as defined in `pyproject.toml`.
- LM dependencies are **optional**:
  - `pip install psannlm` installs the LM tooling (and pulls in `datasets/tokenizers/sentencepiece`)

### Intended direction (cleanup goal)

We want the default install to be lighter and more newcomer-friendly.

- Keep `pip install psann` focused on the estimator/regression core.
- Keep large stacks (LM data tooling, transformers ecosystem, heavyweight benchmarks) out of the default install.
- **Decision:** keep LM library code in `psannlm` so installing `psann` stays lean.

---

## Repository Layout (Where Things Live)

- `src/psann/` — the `psann` Python package (library code)
  - `sklearn.py` — sklearn-style estimator implementations (core entry point for most users)
  - `estimators/_fit_utils.py` — shared fit/input-scaling/validation plumbing
  - `nn_geo_sparse.py` — GeoSparse backbone (experimental)
  - `lm/` — stub module that forwards users to `psannlm`
- `psannlm/` — separate Python package and distribution providing LM APIs + training/CLI utilities
- `tests/` — unit and integration tests for supported functionality
- `docs/` — documentation (see `docs/README.md` for the index)
- `scripts/` — operational scripts (training, evaluation, sweeps); not shipped in the wheel
- `examples/` — runnable examples and configuration snippets
- `reports/`, `runs/`, `eval_data/` — generated outputs (should not be committed; ignored by git)

---

## Support Policy and Versioning

- **Versioning:** semver-style (`MAJOR.MINOR.PATCH`)
  - Patch: bug fixes, doc fixes, internal refactors with no intentional API change
  - Minor: new features and/or new experimental components; may include small deprecations
  - Major: breaking API changes (rename/removal/semantic change of supported surfaces)
- **Deprecations:** supported APIs should be deprecated before removal when feasible.
- **Compatibility:** CPU-first correctness is required; GPU support is best-effort and depends on the user’s PyTorch/CUDA install.

---

## Where to Start

- “How do I use PSANN?” → `README.md` and `docs/API.md`
- “What docs are current?” → `docs/README.md`
- “Where do things live?” → `docs/REPO_STRUCTURE.md`
- “How do I contribute?” → `docs/CONTRIBUTING.md`
- “What are we doing next?” → `docs/project_cleanup_todo.md`
