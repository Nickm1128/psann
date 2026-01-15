# LSM Robustness TODO

> **Note:** The extras framework has been retired. This backlog now tracks robustness work for the primary-output pipeline and stateful controllers; the former extras checklist is summarised below for historical reference.

## Active Focus (2025-10-28)

- [ ] Formalise the current state update (rho/beta/max_abs/init/detach) into a documented dynamical system so we can reason about stability and convergence.
- [ ] Unify the estimator, sequence predictors, and online trainers around a `StateConfig` object instead of ad-hoc dicts, ensuring construction, cloning, and checkpoint reloads stay consistent.
- [ ] Rework `predict_sequence` / `predict_sequence_online` to share a single backend that handles detach semantics, teacher forcing, and gradient flow guarantees.
- [ ] Introduce bounded-state safeguards (normalisation/clipping schedules, optional learnable gain) and prove they keep the update contract satisfied under typical streaming lr values.
- [ ] Add diagnostics to surface exploding/vanishing state norms during fit and streaming evaluation, tying alerts back to config suggestions.
- [ ] Build regression suites with synthetic drift, shock, and regime-switch datasets to validate free vs. teacher-forced rollouts across resets.
- [ ] Document the recommended staged training workflow for stateful models, calling out when to enable streaming optimisers and how to checkpoint/reset safely.
- [ ] Run a compatibility pass over existing HISSO/stateful notebooks to flag behaviours that break once the new formalism lands.

## Historical Extras Work (archived 2025-10-28)

All extras-expansion tasks landed between 2025-10-25 and 2025-10-28, culminating in staged warm-start policies, optimiser resets, regression coverage in `tests/test_supervised_extras.py`, and doc updates. Refer to the Git history from 2025-10-25..28 for implementation details if deeper context is needed.

## Completed

- [x] Generalise `expand_extras_head` for conv/preserve_shape/per_element regressors and add regression coverage so extras expansion stays stable across layouts (src/psann/extras.py:352, tests/test_supervised_extras.py:183, tests/test_supervised_extras.py:209).
- [x] Finalise a unified `extras_growth` API (constructor arg plus setter) and map existing extras flags onto it so estimators, helpers, and configs present a single surface area (src/psann/extras.py:35, src/psann/sklearn.py:148, tests/test_supervised_extras.py:125).
- [x] Ship an `expand_extras_head` helper that clones a fitted PSANNRegressor into a wider extras head while preserving trunk weights and optimiser schedules (src/psann/extras.py:239, src/psann/__init__.py:12, tests/test_supervised_extras.py:164).
- [x] Detect extras width mismatches when loading checkpoints and auto-trigger head expansion rather than erroring out (src/psann/sklearn.py:1876, tests/test_supervised_extras.py:179).
- [x] Align alias handling with estimator conventions to match warning-based shims instead of hard errors (src/psann/lsm.py:68, :147, :432, :499; reference behaviour in src/psann/sklearn.py:154, :156).
- [x] Wire the declared \batch_size parameter into the optimiser loop or remove it for parity with the revised regressors (src/psann/lsm.py:130).
- [x] Extend the expanders to accept both NumPy arrays and Torch tensors (and round-trip tensors when provided) to stay compatible with the new regressor fit/validate pathways (src/psann/lsm.py:225, :352, :365, :559, :627).
- [x] Add a thin \forward (plus \to/\train/eval) wrapper that delegates to the fitted LSM so expanders can plug directly into lsm= (src/psann/lsm.py:106, _resolve_lsm_module in src/psann/sklearn.py:415).
- [x] Provide a score_reconstruction helper for the conv expander to mirror the dense path and support diagnostics promised in the docs (src/psann/lsm.py:469).
- [x] Add regression coverage that exercises PSANNRegressor with both LSMExpander and LSMConv2dExpander under lsm_train=True/False and preserve_shape modes to keep _fit integration stable (\tests/).
- [x] Refresh API docs and examples once behaviour changes land so the LSM sections reflect canonical parameter naming, tensor support, and conv diagnostics (docs/API.md, docs/examples/README.md).
# Archived Note

This document is historical and superseded by the cleanup roadmap in
`docs/project_cleanup_todo.md`. Keep for reference only.
