# Repo Hygiene Waves

Use these waves in order. Each wave is meant to leave the repo in a stable state before the next refactor starts.

## Wave 1: Guardrails

Goal: lock in the April 2026 cleanup so the repo stops drifting backward.

Scope:

- add repo-hygiene audit enforcement to CI
- add an import-graph audit for `psannlm` package-split regressions
- add `run_full_suite.py` guardrail tests around generated-output paths
- add direct-entrypoint smoke tests for path-sensitive scripts run from repo checkout

Exit criteria:

- CI runs `tools/repo_hygiene_audit.py`
- guardrail tests cover `psannlm` relative imports and `run_full_suite.py`
- direct script smoke tests pass on local checkout

Status: Complete

## Wave 2: Foundations

Goal: split the oversized helper layers before touching the estimator entrypoint.

Scope:

- refactor `src/psann/estimators/_fit_utils.py`
- refactor `src/psann/hisso.py`
- refactor `src/psann/lsm.py`
- split `tests/test_hisso_primary.py`
- add characterization tests for estimator aliases, serialization, and validation behavior

Exit criteria:

- helper modules are separated by concern
- `test_hisso_primary.py` is split into smaller files
- estimator characterization tests exist before `sklearn.py` is split

Status: Complete

## Wave 3: Estimator Split

Goal: break up `src/psann/sklearn.py` without changing the public API.

Scope:

- extract shared/base estimator behavior
- extract dense/residual estimator implementations
- extract conv/preserve-shape estimator implementations
- extract WaveResNet / SGR / GeoSparse variants
- extract serialization helpers

Exit criteria:

- `src/psann/sklearn.py` becomes a thin public surface or compatibility layer
- public imports and serialization behavior stay stable

Status: Complete

## Wave 4: LM and Benchmark Scripts

Goal: reduce the maintenance cost of the large operational scripts.

Scope:

- refactor `psannlm/train.py`
- refactor `scripts/bench_lm_bases.py`
- refactor `scripts/benchmark_geo_sparse_vs_dense.py`
- refactor `scripts/benchmark_regressor_ablations.py`
- refactor `scripts/run_light_probes.py`
- refactor `scripts/run_geosparse_vs_relu_benchmarks.py`
- decide the fate of `bench_psann_lm.py`
- centralize shared script/report helpers

Exit criteria:

- large scripts delegate to reusable helper modules
- benchmark/report code paths stop duplicating setup logic

Status: Complete

## Wave 5: Docs, Platform, and Deprecations

Goal: finish the cleanup around repo layout, contributor ergonomics, and legacy surface area.

Scope:

- move root TODO/backlog files into `docs/`
- decide whether `psann_adapter.py` is active or historical
- add benchmark-promotion guidance for moving local results into docs
- fix cross-platform `Makefile` / setup workflow
- review shell/path assumptions in script examples
- inventory and document legacy alias removal policy

Exit criteria:

- root is less cluttered
- contributor setup is documented for Windows and Unix-like environments
- alias/deprecation policy is documented and consistent

Status: Complete
