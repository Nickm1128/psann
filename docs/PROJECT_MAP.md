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
  - `PSANNClassifier`
  - `ResPSANNRegressor`
  - `ResConvPSANNRegressor`
  - `WaveResNetRegressor`
  - `SGRPSANNRegressor`
- **Training utilities** shared by those estimators
  - Data preparation (shape handling, scaling) in `psann.estimators._fit_utils`
  - Fail-fast supervised training, structured events, custom metrics, and stateful
    streaming utilities
  - Restricted `.psann-train` checkpoints for deterministic interruption/resume
- **Workplace lifecycle**
  - Serializable task, model, data, training, and inference specifications
  - Registered model creation plus structured `TrainingRun` orchestration
  - Regression, binary, multiclass, and multilabel task adapters
  - Strict/reorder/positional named-feature schema policies
  - Checksummed `.psann` deployment artifacts, metadata-only inspection, restricted
    loading, version migrations, and explicit trusted legacy conversion
  - Bounded stateless inference, isolated streaming sessions, capability-gated
    Torch/ONNX exports, device pools, and the optional reference service
  - Optional SHAP explanations over the deployed raw-input contract, with explicit
    backgrounds, named task outputs, domain groups, and capability-gated frozen
    gradient adapters
  - Explicit accelerator/dtype tiers, bounded restartable data streams, privacy-safe
    fingerprints, retention/redaction contracts, optional operational hooks, and
    performance/security evidence
  - Six installed-wheel certification scenarios, exhaustive current/legacy API
    freezes, and task-oriented regression/classification/deployment/resume/SHAP quick
    starts
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
  - `psannlm.train` remains the public LM training entrypoint and now delegates to `psannlm/_train/`
- **Registered custom Torch modules**
  - Registered reconstructable factories can use the native artifact path
  - General derived exports and gradient explanations remain unsupported unless a
    future capability matrix certifies the exact plugin

If you depend on any experimental pieces, pin a version and expect breaking changes across minor releases.

---

## Installation Model (Current + Intended Direction)

### Current (today)

- `pip install psann` installs the core package and its runtime dependencies as defined in `pyproject.toml`.
- LM dependencies are **optional**:
  - `pip install psannlm` installs the LM tooling (and pulls in
    `datasets/tokenizers/sentencepiece`)
  - the 1.1 LM line requires `psann>=1.1.0rc1,<1.2` and rejects incompatible cores
    before importing LM implementation modules

### Intended direction (cleanup goal)

We want the default install to be lighter and more newcomer-friendly.

- Keep `pip install psann` focused on the estimator/regression core.
- Keep large stacks (LM data tooling, transformers ecosystem, heavyweight benchmarks) out of the default install.
- **Decision:** keep LM library code in `psannlm` so installing `psann` stays lean.

---

## Repository Layout (Where Things Live)

- `src/psann/` — the `psann` Python package (library code)
  - `sklearn.py` — thin sklearn-style estimator facade (core entry point for most users)
  - `_sklearn/` — internal estimator implementation modules split by concern
  - `estimators/_fit_utils.py` — shared fit/input-scaling/validation plumbing used by the estimator package
  - `platform/` — workplace lifecycle, safe artifact, deployment, export, optional
    explainability, bounded streaming, and operational contracts
  - `nn_geo_sparse.py` — GeoSparse backbone (experimental)
  - `lm/` — stub module that forwards users to `psannlm`
- `psannlm/` – separate Python package and distribution providing LM APIs + training/CLI utilities
  - `train.py` - thin compatibility facade for the LM training entrypoint
  - `_train/` - internal LM CLI helpers split into data, tokenizer, export, and CLI wiring modules
  - `eval_adapter.py` - active lm-eval adapter; the root `psann_adapter.py` file is now only a compatibility shim
- `tests/` – unit and integration tests for supported functionality
- `docs/` – documentation (see `docs/README.md` for the index)
  - `backlog/` - active roadmap notes and research TODOs
  - `archive/` - historical planning notes retained for traceability
- `scripts/` — operational scripts (training, evaluation, sweeps); not shipped in the wheel
  - large benchmark CLIs keep their historical filenames and delegate to nearby internal `scripts/_<tool>/` packages
- `examples/` — runnable examples and configuration snippets
- `reports/`, `runs/`, `eval_data/` — generated outputs (should not be committed; ignored by git)

---

## Support Policy and Versioning

- **Versioning:** semver-style (`MAJOR.MINOR.PATCH`)
  - Patch: bug fixes, doc fixes, internal refactors with no intentional API change
  - Minor: new features and/or new experimental components; may include small deprecations
  - Major: breaking API changes (rename/removal/semantic change of supported surfaces)
- **Deprecations:** supported APIs should be deprecated before removal when feasible.
- **Compatibility:** the workplace development line targets Python 3.11-3.13.
  CPU-first correctness is blocking; CUDA claims require recent scheduled evidence.
  See `docs/support_policy.md` and `docs/workplace_support_matrix.md`.

---

## Where to Start

- “How do I use PSANN?” → `README.md` and `docs/API.md`
- “What docs are current?” → `docs/README.md`
- “Where do things live?” → `docs/REPO_STRUCTURE.md`
- “How do I contribute?” → `docs/CONTRIBUTING.md`
- “What are we doing next?” → `docs/project_cleanup_todo.md`
