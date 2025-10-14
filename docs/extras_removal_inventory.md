# Task 8 Inventory — Extras Removal

Snapshot of every extras-dependent code path and the planned disposition before we start modifying implementations. Actions are either `delete` (feature goes away with extras) or `refactor` (we keep the surface but make it primary-output only).

## Core Estimator Surface
- `src/psann/sklearn.py:48-120` — imports of `psann.extras`, warm-start helpers, and extras-aware preprocessors. **Action:** delete once the pipeline no longer needs extras adapters.
- `src/psann/sklearn.py:175-292` — constructor arguments `extras`, `extras_growth`, warm-start flags, and cached extras state (`_extras_cache_`, `_hisso_extras_cache_`, `_supervised_extras_meta_`, `_extras_dim_model_`). **Action:** delete parameters/state; ensure backwards-compatible defaults documented separately.
- `src/psann/sklearn.py:329-388` — accessors/mutators for extras growth (`get_extras_growth`, `set_extras_growth`, `set_extras_warm_start_epochs`, extras-aware `set_params`). **Action:** delete APIs; confirm no downstream callers remain.
- `src/psann/sklearn.py:392-431` — `_set_extras_cache` and `_update_hisso_cache` that persist extras buffers. **Action:** delete and collapse cache handling to HISSO-only data.
- `src/psann/sklearn.py:907-1292` — dense fit path wiring extras options (`extras_kwargs`, `extras_cfg`, `maybe_run_extras_warm_start`, lifecycle plumbing). **Action:** refactor to pure primary-output flow; drop extras kwargs and lifecycle hooks.
- `src/psann/sklearn.py:1705-1830` — residual/convolutional fit mirrors with extras kwargs and lifecycle usage. **Action:** refactor alongside the dense path; remove extras branches.
- `src/psann/sklearn.py:1382-1435` — serialization payload keys (`extras_dim`, extras caches) and load-time resets. **Action:** refactor serialization to omit extras metadata while keeping HISSO caches intact.
- `src/psann/sklearn.py:1900-2000` and downstream variant hooks — hook factories still expose extras adapters/warm-start callbacks. **Action:** refactor hook signatures to drop extras-specific callbacks.

## Shared Fit Utilities
- `src/psann/estimators/_fit_utils.py:28-106` — imports of `ExtrasGrowthConfig`, `SupervisedExtrasConfig`, warm-start lifecycle types. **Action:** delete extras imports once dataclasses are removed.
- `src/psann/estimators/_fit_utils.py:109-210` — `NormalisedFitArgs`, `PreparedInputState`, and helper dataclasses storing `extras_options`, `extras_info`, `extras_meta`, extras dimensions. **Action:** refactor to remove extras metadata fields; keep primary-only structure.
- `src/psann/estimators/_fit_utils.py:290-352` — `normalise_fit_args` extras option scraping and validation. **Action:** refactor to ignore extras kwargs and error out early if callers still provide them.
- `src/psann/estimators/_fit_utils.py:361-516` — `prepare_inputs_and_scaler` extras supervision handling (`extras_cfg`, `extras_targets`, `_prepare_supervised_extras_targets`). **Action:** refactor to drop extras logic and return primary-only dimensions.
- `src/psann/estimators/_fit_utils.py:533-741` — extras metadata application (`_apply_extras_metadata`), supervised extras prep, and y reshaping with extras columns. **Action:** delete or fold into primary-output handling.
- `src/psann/estimators/_fit_utils.py:842-1134` — extras warm-start orchestration (`configure_extras_warm_start`, `maybe_run_extras_warm_start`) and HISSO extras scheduling (`_plan_hisso_training`, `_configure_hisso_extras`). **Action:** delete warm-start helpers; refactor HISSO planning to ignore extras weights/modes.
- `src/psann/estimators/_fit_utils.py:1161-1376` — training loop glue that branches on `extras_info` for loss construction and validation prep. **Action:** refactor to assume primary-only outputs while keeping HISSO validation shape checks.

## Extras-Specific Modules
- `src/psann/extras.py` — definitions for `SupervisedExtrasConfig`, `ExtrasGrowthConfig`, head expansion, cache utilities, supervised extras rollout. **Action:** delete module; migrate any generally useful tensor helpers before removal (none currently needed post-extras).
- `src/psann/extras_scheduling.py` — warm-start lifecycle dataclass and callbacks. **Action:** delete; HISSO training will no longer gate on extras heads.
- `src/psann/augmented.py` — predictive extras trainer with cache management. **Action:** delete; no longer part of supported estimator surface.
- `src/psann/lm.py` — language-model extras trainer and cache helpers. **Action:** delete or archive; confirm no remaining importers once extras APIs are removed.

## HISSO Pipeline
- `src/psann/hisso.py:12-239` — imports `SupervisedExtrasConfig`, extras targets/loss weight options, `_prepare_supervised_extras_targets`, and metadata caching. **Action:** refactor to drop extras supervision branches while retaining HISSO primary training.
- `src/psann/hisso.py:339-538` — predictive extras adapters (`make_predictive_extras_trainer_from_estimator`, cache load/save) that depend on extras trainers. **Action:** delete along with predictive extras modules; ensure HISSO checkpoints remain functional without extras caches.
- `scripts/profile_hisso.py` & HISSO-related tests reference `hisso_extras_*` parameters. **Action:** refactor scripts/tests to primary-only HISSO settings after code changes.

## Types & Package Surface
- `src/psann/types.py:56-96` — `ExtrasGrowthConfig` type alias and `ExtrasFitParams`/`ExtrasNormalisedOptions`. **Action:** delete extras aliases; trim HISSO typed dicts to remove `hisso_extras_*` entries.
- `src/psann/__init__.py:31-95` — exports for extras helpers (`make_predictive_extras_trainer_from_estimator`, `ensure_supervised_extras_config`, etc.). **Action:** delete exports alongside module removal.
- `src/psann/utils/synthetic.py:71-83` — synthetic data generator that emits extras regimes. **Action:** refactor to produce primary-only signals or remove unused outputs.

## Tests, Docs, and Serialization Artefacts
- `tests/test_*` suites covering extras scheduling, supervised extras, and HISSO extras weighting depend on soon-to-be-removed APIs. **Action:** delete or rewrite to validate the simplified estimator behaviour.
- Documentation (`README.md`, `docs/`, `scripts/README.md`, examples, notebooks) contains extras usage examples. **Action:** update during Task 9 after code removal; note references for follow-up.
- Saved-model payloads currently include `extras_dim` and extras caches. **Action:** adjust migration notes and loaders so old checkpoints either error clearly or migrate by dropping extras fields.

## Follow-Up Refactor Outline
1. Strip extras constructor parameters/state from `PSANNRegressor` and variants; prune serialization payloads the same time so checkpoints become extras-free.
2. Collapse `_fit_utils` data carriers to primary-only fields, then rewrite `prepare_inputs_and_scaler` and training glue without `extras_*` kwargs.
3. Delete orphaned extras modules (`extras.py`, `extras_scheduling.py`, predictive trainers) alongside their exports and typed dicts.
4. Refactor HISSO helpers to operate without extras supervision/caches and remove public `hisso_extras_*` arguments.
5. Sweep tests, scripts, and docs to drop extras references, then add coverage confirming the streamlined estimator state.

### Progress Log
- 2025-10-14: `_fit_utils` and `types.py` now return primary-only results while feeding legacy extras fields with neutral defaults so existing estimator hooks keep importing; next step is to update the estimator implementations before removing the compatibility shims.
