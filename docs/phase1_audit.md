# Phase 1 Audit Summary

## 1.1 Parameter Naming Inventory
- **hidden_width**: core dense width parameter used in `psann.nn.PSANNNet`, `psann.sklearn` regressors, `lsm.LSM`, and PSANN LM wrapper.
- **hidden_channels**: convolutional width parameter used in `psann.conv` conv nets, convolutional branches in `psann.sklearn`, and LSM conv expanders.
- **output_shape** (sklearn API) vs **out_dim** (network classes) vs inferred `n_targets`: governs target dimensionality.
- **extras / extras_dim / primary_dim**: extras count handled in regressors, HISSO trainer, and LM extras cache.
- **in_channels / input_dim**: convolution vs dense naming in underlying modules.
- **kernel_size / conv_kernel_size**: used for conv nets vs estimator argument.
- **batch_episodes / episode_length / batch_size**: multiple batching concepts in HISSO and LM.

## 1.2 Canonical Naming Proposal
- **hidden_units** for dense layers (alias existing `hidden_width` until deprecated).
- **conv_channels** for convolutional hidden width (alias `hidden_channels`).
- Standardise on **output_dim** for final feature count; treat `output_shape` as convenience helper and compute from `output_dim` when needed.
- Use **extras_dim** consistently for extras outputs (keep `extras` as user-facing alias).
- Adopt **input_dim** (dense) and **in_channels** (conv) but expose cohesive docstring guidance.
- Document mapping table and plan backwards-compat keyword translation in `_normalize_params` helper during refactor phase.

## 1.3 Supervised Extras Behaviour (Time-Series)
- Allow users to append extras columns to `y` or supply `extras_targets`; detect columns automatically when `extras_dim > 0`.
- During supervised fit, propagate extras sequentially like HISSO: maintain rolling state buffers and ensure extras predictions feed into next-step inputs for evaluation/inference.
- Provide helper to roll supervised predictions over time-series (mirroring `hisso_infer_series`) to keep extras semantics aligned across training modes.

## 1.4 Module Responsibility Map
- **sklearn.py**: retain estimator wrappers but move heavy logic into:
  - `training.py`: shared `_fit_core`, optimizer/validation utilities, extras scheduling.
  - `hisso.py`: HISSO config, trainer wiring, warm-start orchestration.
  - `preproc.py`: preprocessor/LSM adapters plus registration surface.
- **lm.py**: reuse shared training helper; keep LM-specific embedding prep + curriculum.
- **lsm.py**: expose concrete preprocessors implementing new interface.
- Ultimately keep estimator classes thin wrappers assembling config + calling shared utilities.

## 1.5 TODO / Coverage Review
- No inline `TODO` markers detected under `src/`.
- `# pragma: no cover` only present on `BaseTokenizer` abstract methods (`src/psann/tokenizer.py` lines 16-26); acceptable as interface stubs but flag for potential abstract-base test doubles.
# Archived Note

This document captures an earlier Phase 1 snapshot and is retained for traceability.
Refer to `docs/project_cleanup_todo.md` and `docs/REPO_STRUCTURE.md` for current guidance.
