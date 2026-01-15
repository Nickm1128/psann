# Architecture Overview

This document sketches the PSANN module layout and how data flows through the core estimator stack. It is intentionally high level; see `docs/API.md` for the public surface and `docs/REPO_STRUCTURE.md` for repo layout.

## High-level stack

```
psann/
  sklearn.py                -> sklearn-style estimators (public surface)
  estimators/_fit_utils.py  -> shared fit prep: scaling, shaping, hooks
  activations.py            -> PSANN/ResPSANN/SGR activations + configs
  layers/                   -> building blocks (sine residual, geo_sparse, etc.)
  nn_geo_sparse.py          -> GeoSparseNet backbone (experimental)
  hisso/                    -> episodic training + reward utilities
  utils/                    -> diagnostics + small helpers
  lm/                       -> LM library code (experimental; may move to psannlm)
```

```
psannlm/                     -> separate distribution (LM training/CLI utilities)
```

## Core estimator flow (supervised)

1. **Input normalisation** via `normalise_fit_args` (dtype, validation splits, shape hints).
2. **Scaling + shape prep** via `prepare_inputs_and_scaler`:
   - decides flatten vs preserve-shape paths
   - applies optional scalers
   - prepares metadata for prediction and streaming paths
3. **Model build** via `build_model_from_hooks`:
   - selects base (PSANN, ResPSANN, WaveResNet, SGR, GeoSparse)
   - attaches optional LSM expanders or attention
4. **Training** via `run_supervised_training`:
   - shared optimizer/scheduler logic
   - early stopping + validation hooks
5. **Prediction** reuses prepared metadata for consistent output shapes.

## HISSO flow (episodic)

- `HISSOOptions` resolves reward, transforms, and context configuration.
- `EpisodeTrainer` runs episodes on the estimator’s device and logs rewards.
- `hisso_infer_series` and `hisso_evaluate_reward` reuse the stored episode config.

## LM flow (experimental)

- `psannlm.psannLMDataPrep` handles tokenisation + dataset packing.
- `psannlm.psannLM` exposes a compact fit/generate interface.
- The CLI / long-run training utilities live in `psannlm.lm.train.cli`.

## Design goals

- **Stable core surface**: sklearn-style estimators are the primary supported API.
- **Shared fit helpers**: keep preprocessing and training logic in `_fit_utils`.
- **Experimental isolation**: GeoSparse and LM code are clearly labeled experimental.
