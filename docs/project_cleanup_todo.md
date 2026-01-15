# PSANN Project Cleanup & Packaging Roadmap (Comprehensive TODO)

This checklist is intended to turn the repo into a “newcomer-friendly” project with:
- a clear supported surface area (what PSANN *is* vs what’s experimental),
- consistent, readable, well-tested code,
- reproducible benchmarks,
- and a lean PyPI distribution (`pip install psann`) with heavy functionality behind optional extras or separate packages.

Conventions:
- Each line is actionable and checkable: `- [ ]`.
- Indentation indicates subtasks.
- “Definition of done” is embedded where helpful.

---

## 0) Decide Scope, Audience, and Support Policy

- [x] Write a 1-page “Project Map” doc (what’s included, who it’s for, what’s supported)
  - [x] Define the primary user personas (researcher, practitioner, contributor)
  - [x] Define the supported public surfaces (e.g., sklearn-style estimators, torch layers)
  - [x] Explicitly label experimental areas (GeoSparse, LM training, sweeps, scripts)
  - [x] Define support policy (best-effort vs stable APIs; stability guarantees)
  - [x] Define versioning policy (semver vs date-based; what triggers minor/major bumps)

- [x] Establish a “core vs addons” policy for the repo
  - [x] Identify “core library” functionality that should remain minimal and stable
  - [x] Identify “add-on” functionality that should be optional (extras) or split into separate distributions
  - [x] Decide whether the Language Modeling tooling belongs in `psann` or a separate distribution
    - [x] Decision: LM belongs in a separate distribution (keep `psann` core lean)

---

## 1) Repo Hygiene & Organization (Make It Navigable)

- [x] Produce a top-level repo structure standard
  - [x] Document directory purposes (e.g., `src/`, `tests/`, `docs/`, `scripts/`, `examples/`)
  - [x] Decide where benchmarks live (`benchmarks/` vs `scripts/` vs `examples/`)
  - [x] Decide where research runs/outputs live (outside git; clearly ignored)

- [x] Audit the repository for “should not be in git” artifacts
  - [x] Ensure `runs/`, `reports/`, `eval_data/`, caches, and model checkpoints are ignored
  - [x] Add a “do not commit” note in docs for outputs directories
  - [x] Add a small “outputs/README.md” describing expected generated directories

- [x] Clean up / standardize shell scripts and entrypoints
  - [x] Ensure every script has an explicit usage header and examples
  - [x] Standardize CLI parsing (argparse/typer) and consistent flags
  - [x] Standardize logging format (timestamp, config summary, output dirs)

- [x] Standardize naming conventions and module boundaries
  - [x] Decide on `snake_case` vs `kebab-case` for scripts/configs
  - [x] Ensure “lm” code is clearly separated from “core estimator” code
  - [x] Consolidate duplicated utilities in multiple places (scripts vs src)

---

## 2) Documentation Overhaul (For Newcomers)

- [x] Consolidate and simplify entry docs
  - [x] Make a single canonical “start here” doc (either `README.md` or `docs/README.md`)
  - [x] Ensure the README answers:
    - [x] What is PSANN?
    - [x] What’s the simplest 5-minute example?
    - [x] What problems does it solve?
    - [x] What are the stable APIs?
    - [x] How do I install CPU vs GPU?
  - [x] Add a “FAQ / Common issues” section (CUDA/PyTorch installs, dtype, AMP, memory)

- [x] Create minimal, runnable examples
  - [x] “Hello world” regression example (CPU-only)
  - [x] GeoSparse example (with clear caveats if experimental)
  - [x] Time-series/stateful example (if part of supported surface)
  - [x] Save/load example (serialization)

- [x] Improve API reference quality
  - [x] Ensure docstrings for public classes/functions explain shapes, dtype expectations, and defaults
  - [x] Add a short “Design & Architecture” doc (high-level diagram + module map)
  - [x] Add a “Performance tips” doc (batching, compile, tf32, dtype, GPU best practices)

- [x] Reduce doc fragmentation and stale docs
  - [x] Audit `docs/` for outdated plans (e.g., “inventory” docs that no longer match code)
  - [x] Archive or summarize very large reports into a short “results summary” + link to appendix
  - [x] Ensure the docs don’t contradict the current codebase

Definition of done:
- New contributor can run a minimal example in <10 minutes on CPU.
- Docs clearly distinguish stable APIs from experiments.

---

## 3) Code Readability, Maintainability, and Consistency

- [x] “Public API” audit
  - [x] List the supported API surface (what’s exported from `psann`)
  - [x] Identify internal-only modules and make that clear (underscore modules or docs)
  - [x] Reduce accidental surface area (avoid exporting experimental helpers by default)

- [x] Type hints and runtime guards
  - [x] Ensure imports of optional dependencies fail gracefully with clear messages
  - [x] Ensure shape validation errors are actionable (expected vs actual)
  - [x] Add lightweight runtime asserts for invariants (without spamming or slowing hot paths)

- [x] Remove dead code and “research leftovers”
  - [x] Identify unused modules/functions via grep + test coverage
  - [x] Delete or move to `docs/archive/` anything not meant to be supported
  - [x] Ensure no training artifacts/scripts are imported by the core library accidentally

- [x] Standardize logging and reproducibility
  - [x] Ensure training/bench scripts log config, seed, versions, and environment
  - [x] Ensure deterministic modes are documented and tested where feasible

---

## 4) Packaging & PyPI Distribution (Make `pip install psann` Lean)

**Goal:** `pip install psann` installs a minimal, broadly-compatible core. Heavy stacks are opt-in.

- [x] Audit what the wheel currently contains
  - [x] Confirm whether `psann/lm` ships inside the `psann` wheel (now a stub only)
  - [x] Decide whether LM code should be excluded from the `psann` wheel and shipped as `psannlm`

- [ ] Re-evaluate dependency strategy for PyTorch
  - [ ] Decide whether `torch` should remain a hard dependency
  - [ ] If removing hard dependency on `torch`:
    - [ ] Make imports lazy/optional so `import psann` works without torch
    - [ ] Provide clear error messages when torch-backed features are used without torch installed
    - [ ] Add docs for installing torch for CPU/GPU (link to PyTorch install selector)
  - [ ] If keeping torch as dependency:
    - [ ] Document how to avoid accidentally installing CPU torch on GPU machines (containers/constraints)

- [x] Optional extras cleanup and naming
  - [x] Ensure extras names are intuitive: `psann[dev]`, `psann[viz]`, `psann[sklearn]` (LM lives in `psannlm`)
  - [ ] Consider adding “meta extras”:
    - [ ] `psann[all]` for everything
    - [ ] `psann[bench]` for benchmark tooling
  - [x] Ensure extras are minimal and do not include unnecessary transitive deps

- [x] Distribution split plan (if needed)
  - [x] Create the `psannlm` package namespace and distribution
  - [x] Move LM-specific code and scripts under that distribution
  - [x] Add clear docs that `psannlm` depends on `psann`
  - [x] Add integration tests that `psannlm` works with released `psann`

- [x] Release hygiene
  - [x] Ensure `pyproject.toml` metadata is accurate (URLs, description, keywords)
  - [x] Add a “Minimal install” and “Extras install” section to README
  - [ ] Confirm sdist/wheel contents in CI (size checks, import checks)

Definition of done:
- Clean venv can `pip install psann` quickly and import the package.
- Heavy dependencies (datasets/tokenizers/transformers, LM scripts, benchmark stacks) are opt-in.

---

## 5) Testing Strategy, CI, and Quality Gates

- [x] Clarify test tiers and expected runtime
  - [x] Fast unit tests (default)
  - [x] Slow integration tests (opt-in marker)
  - [x] GPU tests (opt-in marker, skip in default CI unless GPU runner exists)

- [x] CI pipeline improvements
  - [x] Add/verify CI for: lint (`ruff`), format (`black --check`), typecheck (`mypy`)
  - [x] Run fast tests on multiple Python versions (>=3.9)
  - [x] Add packaging checks: build wheel/sdist, install into clean env, import smoke test

- [x] Test coverage targeting “newcomer paths”
  - [x] Add tests for the minimal example from README (or keep it as a doctest-style check)
  - [x] Add tests that optional deps fail gracefully with helpful messages

---

## 6) Performance & Optimization Workplan

- [x] Create a small, stable benchmarking harness
  - [x] Standard “microbench” script(s) for throughput/memory (CPU + GPU optional)
  - [x] Capture versions, shapes, dtype, compile flags, and write results to JSON
  - [x] Add comparisons against baselines (dense MLP / transformer)

- [x] Improve profiling ergonomics
  - [x] Add a single “profile run” wrapper (torch profiler + NVTX) with a consistent output format
  - [x] Add a “perf regression” microbench that can be run in CI (CPU-only)

- [x] GeoSparse performance profiling (if it’s a focus area)
  - [x] Profile forward/backward hotspots (kernel time breakdown)
  - [x] Evaluate `torch.compile` modes and stability (avoid OOM, avoid graph breaks)
  - [x] Explore vectorization / fused kernels / layout changes
  - [x] Document recommended settings per GPU generation (Ampere/Hopper/Blackwell)

- [x] Memory and allocation controls
  - [x] Document allocator env vars and recommended defaults
  - [x] Ensure scripts don’t over-reserve GPU memory unnecessarily

---

## 6.5) PSANN-LM / Language Modeling Subproject (If We Keep It Here)

- [x] Clarify where LM code lives and how it’s installed
  - [x] Split into the `psannlm` distribution
  - [x] Ensure `psann` core users aren’t exposed to LM concepts by default
  - [x] Define shared APIs between `psann` and `psannlm` (what’s imported from where)

- [x] Make LM training reproducible and newcomer-friendly
  - [x] One canonical entrypoint (single CLI) for: train, resume, eval, generate
  - [x] Single canonical config format (CLI flags; YAML helper retained for tiny CPU runs)
  - [x] Clear output directory conventions (runs, checkpoints, tokenizer, logs)
  - [x] Document minimum hardware + expected memory/throughput

- [x] Training data pipeline correctness and resume-safety
  - [x] Clearly document data sources and sampling strategy (shuffling, buffering, mixing)
  - [x] Ensure resume does not re-consume data in a biased/duplicated way (documented limitations)
  - [x] Provide “offline eval shard” generation and reuse (avoid redownloading; stable evaluation)
  - [x] Add “sidecar eval” script that can evaluate multiple checkpoints deterministically

- [x] Tokenizer workflow improvements
  - [x] Make tokenizer training reproducible (seed, shuffle, dataset snapshot)
  - [x] Ensure resuming training reuses the existing tokenizer by default
  - [x] Provide guardrails to prevent accidental retraining of tokenizer in a resumed run

- [x] LM packaging and dependency boundaries
  - [x] Ensure the base `psann` wheel does not require LM dependencies
  - [x] Ensure `psannlm` installs only what is needed (pin/compat notes where required)
  - [x] Add docs for running inside Docker/containers (GPU-enabled PyTorch constraints)

---

## 7) Developer Experience (Make Contributing Easy)

- [x] Add/verify “one command” local setup
  - [x] `make dev` / `just dev` style bootstrap (venv, install, pre-commit)
  - [x] Document common workflows: run tests, run formatter, build wheel

- [x] Improve contribution docs
  - [x] Update `docs/CONTRIBUTING.md` with current structure and expectations
  - [x] Add “How to add a new model base / benchmark / dataset” guide

---

## 8) Project Management and Backlog Hygiene

- [x] Consolidate scattered TODOs
  - [x] Inventory root TODO docs and `docs/backlog/` (`docs/backlog/todo_inventory.md`)
  - [x] Decide what becomes GitHub issues vs what remains in docs (see inventory guidance)
  - [x] Link this checklist to the canonical issue tracker (https://github.com/psann-project/psann/issues)

- [x] Define milestones for the cleanup
  - [x] Milestone A: newcomer onboarding + minimal install
    - Definition: README + docs + `make dev` path validated; `pip install psann` stays lean.
  - [x] Milestone B: packaging split (if chosen)
    - Definition: `psannlm` published separately; core wheel does not ship LM deps.
  - [x] Milestone C: performance + benchmark stability
    - Definition: microbench + profiler scripts stable; CI smoke checks in place.
  - [x] Milestone D: first “stable” release with docs + CI gates
    - Definition: CI lint/test/build gates + docs audit complete; release checklist ready.
