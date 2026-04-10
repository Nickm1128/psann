# Repo Hygiene Follow-Ups

This checklist captures the remaining cleanup work after the April 2026 hygiene pass.

Already completed in that pass:

- removed tracked generated outputs and benchmark bundles
- added `tools/repo_hygiene_audit.py` and a regression test
- blocked `scripts/run_full_suite.py --git-commit` from committing generated output trees
- fixed `psannlm` package-split import regressions
- fixed alias-default mismatches in low-level PSANN/conv builders

Use this file for the next cleanup waves.

Wave order lives in `docs/repo_hygiene_waves.md`.

## P0: CI and Guardrails

- [x] Add `python tools/repo_hygiene_audit.py --json` to CI and fail when `prohibited_tracked` is non-empty.
- [x] Add a packaging/import audit test that searches `psannlm/` for stale imports into nonexistent `psannlm.*` modules that should come from `psann.*`.
- [x] Add a smoke test for `scripts/run_full_suite.py` that asserts `--git-commit` is rejected when `--out-root` is under `reports/`, `runs/`, `outputs/`, or `logs/`.
- [x] Add Windows-oriented script smoke tests for path-sensitive scripts that are meant to run directly from repo checkout.

## P1: Core Library Modularization

- [x] Refactor `src/psann/sklearn.py` (completed as a thin public facade over `src/psann/_sklearn/`).
- [x] Before splitting `src/psann/sklearn.py`, add characterization tests around:
  - [x] constructor aliases and warnings
  - [x] serialization / load round-trips
  - [x] per-estimator validation and device behavior
- [x] Split `src/psann/sklearn.py` into smaller modules by concern while keeping top-level imports stable:
  - [x] shared/base estimator behavior
  - [x] dense / residual estimators
  - [x] conv / preserve-shape estimators
  - [x] WaveResNet / SGR / GeoSparse variants
  - [x] serialization helpers
- [x] Refactor `src/psann/estimators/_fit_utils.py` (1039 lines) into:
  - [x] fit-argument normalization
  - [x] input/target preparation
  - [x] validation preparation
  - [x] supervised training orchestration
  - [x] shared optimizer / noise helpers
- [x] Refactor `src/psann/hisso.py` (1145 lines) into:
  - [x] option/config normalization
  - [x] warm-start helpers
  - [x] context extraction helpers
  - [x] trainer entrypoints
- [x] Refactor `src/psann/lsm.py` (863 lines) into:
  - [x] dense LSM modules
  - [x] conv LSM modules
  - [x] expander/build helpers
- Status note (2026-04-10): Wave 2 split estimator fit helpers into `_fit_args.py`, `_fit_inputs.py`, `_fit_validation.py`, and `_fit_types.py`; moved HISSO and LSM internals behind compatibility facades; and added estimator characterization coverage before the `sklearn.py` split.
- Status note (2026-04-10): Wave 3 split estimator internals into `src/psann/_sklearn/{base,builders,scaling,inference,sequence,serialization,wave,sgr,residual,geosparse}.py` while keeping `src/psann/sklearn.py` as the stable import and checkpoint-compatibility facade.

## P1: LM and Benchmark Script Modularization

- [x] Refactor `psannlm/train.py` (1037 lines) into separate CLI, tokenizer prep, dataset wiring, and export/checkpoint helpers.
- [x] Refactor `scripts/bench_lm_bases.py` (1477 lines) into:
  - [x] config loading
  - [x] tokenizer bootstrap
  - [x] run execution
  - [x] report writing
- [x] Refactor `scripts/benchmark_geo_sparse_vs_dense.py` (1446 lines) into dataset/model/report modules.
- [x] Refactor `scripts/benchmark_regressor_ablations.py` (1145 lines) into reusable benchmark harness pieces.
- [x] Refactor `scripts/run_light_probes.py` (1127 lines) by separating dataset setup, experiment runners, and report aggregation.
- [x] Refactor `scripts/run_geosparse_vs_relu_benchmarks.py` (874 lines) into smaller reusable helpers.
- [x] Decide whether root-level `bench_psann_lm.py` (1303 lines) should move under `scripts/`, be split, or be archived.
- [x] Expand `scripts/_cli_utils.py` or introduce a small internal script-utils module for shared CLI/report helpers so benchmark scripts stop duplicating setup logic.
- Status note (2026-04-10): Wave 4 moved the large LM and benchmark entrypoints behind internal helper packages (`psannlm/_train/`, `scripts/_bench_lm_bases/`, `scripts/_benchmark_geo_sparse_vs_dense/`, `scripts/_benchmark_regressor_ablations/`, `scripts/_run_light_probes/`, `scripts/_run_geosparse_vs_relu_benchmarks/`, and `_bench_psann_lm/`) while keeping the existing command paths as compatibility facades. `bench_psann_lm.py` stays at repo root as a shim because existing RunPod and local workflows already reference that exact path.

## P1: Test Organization

- [x] Split `tests/test_hisso_primary.py` (923 lines) by behavior area:
  - [x] options/config normalization
  - [x] supervised warm-start behavior
  - [x] vectorized trainer behavior
  - [x] determinism / profiling
- [ ] Add focused tests for low-level alias defaults in `PSANNNet`, `SGRPSANNSequenceNet`, and `PSANNConv{1,2,3}dNet` so default alias regressions stay caught.
- [ ] Add focused tests for `psannlm` modules that intentionally depend on shared `psann` modules.

## P2: Root and Docs Cleanup

- [x] Move root TODO/backlog files into `docs/backlog/` or `docs/archive/`, then link them from `docs/README.md`:
  - [x] `PSANN_LM_Module_TODO.txt`
  - [x] `psann_lm_3b_todo.md`
  - [x] `psann_lm_todo.md`
  - [x] `psann_sparse3d_todo.md`
  - [x] `REPO_CLEANUP_TODO.md` if it still adds value beyond `docs/project_cleanup_todo.md`
- [x] Decide whether `psann_adapter.py` belongs in `src/`, `tools/`, `scripts/`, or archive it if it is historical.
- [x] Audit docs for any remaining references to versioned raw output trees or zip bundles that should stay local.
- [x] Add a short benchmark-promotion guide describing how to turn local `reports/` outputs into compact `docs/benchmarks/` artifacts.

## P2: Cross-Platform Developer Experience

- [x] Fix `Makefile` to stop assuming Unix-style `./.venv/bin/...` paths.
- [x] Either make `make dev` cross-platform or add a documented PowerShell equivalent in `scripts/` or `docs/CONTRIBUTING.md`.
- [x] Review script examples for path separators and shell assumptions on Windows.

## P3: Deprecation and Surface Cleanup

- [x] Inventory every public legacy alias (`hidden_width`, `hidden_channels`, related knobs) and document the deprecation/removal plan.
- [x] Keep alias behavior consistent across low-level modules and sklearn-style estimators.
- [x] Decide which warnings should remain user-facing versus internal-only once the estimator refactors are done.

- Status note (2026-04-10): Wave 5 moved stray root TODO files into `docs/backlog/` and `docs/archive/`, relocated the active lm-eval adapter into `psannlm/eval_adapter.py` with a root compatibility shim, added benchmark-promotion and deprecation-policy docs, and made the `make dev` bootstrap plus shell examples explicit for both Windows and Unix-like environments.

## Current Hotspot Queue

Use this list as the default refactor order unless a feature task forces a different order:

1. `psannlm/lm/train/trainer.py`
