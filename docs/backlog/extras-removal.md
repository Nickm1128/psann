# Extras Removal Backlog

This backlog distils the decisions from `extras_removal_inventory.md` into issue-ready chunks. Each section can become a standalone GitHub issue; copy the summary, tasks, and exit criteria when filing.

## 1. Core Estimator Surface
- **Summary:** Strip extras-related constructor parameters, caches, and lifecycle hooks from `PSANNRegressor` variants while keeping backwards-compatible warnings where feasible.
- **Tasks:**
  - Delete imports of `psann.extras` and warm-start helpers from `src/psann/sklearn.py`.
  - Remove `extras*` constructor kwargs and state (`_extras_cache_`, `_supervised_extras_meta_`, etc.).
  - Collapse extras-specific accessors (`get_extras_growth`, `set_extras_growth`, warm-start setters).
  - Refactor dense/residual fit paths to operate on primary outputs only.
  - Update serialization payloads to drop extras metadata while keeping HISSO caches intact.
- **Exit Criteria:** Estimator signatures no longer accept extras knobs, serialization omits extras fields, and unit tests cover pure-primary fits.

## 2. Shared Fit Utilities
- **Summary:** Simplify `_fit_utils` to remove extras metadata, warm-start orchestration, and supervised target handling.
- **Tasks:**
  - Delete extras-specific dataclasses and imports.
  - Rewrite `normalise_fit_args` and `prepare_inputs_and_scaler` to reject extras kwargs explicitly.
  - Remove extras warm-start helpers and HISSO scheduling branches.
- **Exit Criteria:** `_fit_utils` assumes primary-only data, and test coverage verifies error paths when extras kwargs are passed.

## 3. Extras-Specific Modules
- **Summary:** Remove standalone extras modules now that the estimator no longer depends on them.
- **Tasks:**
  - Delete `src/psann/extras.py`, `src/psann/extras_scheduling.py`, `src/psann/augmented.py`, and `src/psann/lm.py` extras helpers.
  - Migrate any reusable tensor utilities before deletion (currently none required).
- **Exit Criteria:** No remaining imports of the deleted modules; test suite passes without extras availability.

## 4. HISSO Pipeline
- **Summary:** Refactor HISSO helpers to operate without extras supervision or caches.
- **Tasks:**
  - Remove extras supervision branches from `src/psann/hisso.py`.
  - Delete predictive extras adapters and cache loaders.
  - Update any CLI or scripts that exposed `hisso_extras_*` flags.
- **Exit Criteria:** HISSO pipeline trains and resumes without extras metadata, and integration tests cover the primary-only flow.

## 5. Types and Package Surface
- **Summary:** Prune extras types and exports after the core refactor lands.
- **Tasks:**
  - Remove extras aliases from `src/psann/types.py`.
  - Update `src/psann/__init__.py` exports to drop extras helpers.
  - Refactor `utils/synthetic.py` to emit primary-only fixtures or delete unused paths.
- **Exit Criteria:** Package init exposes only the supported public API, and typing stubs compile without extras symbols.

## 6. Tests, Docs, and Serialization Artefacts
- **Summary:** Clean up the safety net after code changes land.
- **Tasks:**
  - Delete extras-specific tests or rewrite them for the new primary-only behaviour.
  - Refresh documentation, scripts, and notebooks to remove extras references.
  - Update migration notes so legacy checkpoints either error with guidance or migrate cleanly.
- **Exit Criteria:** Test suite reflects the simplified estimator, docs no longer mention extras, and migration guidance explains the change.

### Coordination Notes
- Tackle sections 1-5 sequentially; section 6 can happen incrementally once code deltas settle.
- Track progress by mirroring the section headers as GitHub labels or issue titles (for example, "Extras Removal: Core Estimator").
- Update `extras_removal_inventory.md` and this backlog together when scope changes.
