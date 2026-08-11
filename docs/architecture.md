# Architecture Overview

This document sketches the PSANN module layout and how data flows through the core estimator stack. It is intentionally high level; see `docs/API.md` for the public surface and `docs/REPO_STRUCTURE.md` for repo layout.

## High-level stack

```
psann/
  sklearn.py                -> sklearn-style estimator facade (public surface)
  _sklearn/                 -> base/scaling/builders/variant implementation modules
  estimators/_fit_utils.py  -> shared fit prep: scaling, shaping, hooks
  estimators/_fit_contracts.py -> fail-fast config, device, shape, warm-start checks
  training.py              -> optimizer loop, events, metrics, fallback policies
  training_events.py       -> stable structured event contract and logger adapter
  training_checkpoint.py   -> restricted, checksummed resume checkpoint container
  platform/specs.py        -> serializable task/model/data/train/inference config
  platform/registry.py     -> backbone and training-component registries
  platform/tasks.py        -> target validation, task losses, probabilities, metrics
  platform/lifecycle.py    -> create_model/train/TrainingRun orchestration
  platform/artifacts.py    -> export/inspect/load/migrate public artifact boundary
  platform/artifact_io.py  -> atomic ZIP and checksum validation
  platform/artifact_models.py -> safe model state and registered reconstruction
  platform/inference.py    -> batched stateless runtime, sessions, device pools
  platform/exports.py      -> parity-gated torch.export/ONNX derivation
  platform/integrations.py -> explicit artifact resolver adapters
  platform/module_adapter.py -> limited arbitrary torch-module integration
  serving.py               -> optional reference HTTP worker
  activations.py            -> PSANN/ResPSANN/SGR activations + configs
  layers/                   -> building blocks (sine residual, geo_sparse, etc.)
  nn_geo_sparse.py          -> GeoSparseNet backbone (experimental)
  hisso/                    -> episodic training + reward utilities
  utils/                    -> diagnostics + small helpers
  lm/                       -> compatibility stub directing users to psannlm
  platform/                 -> unified workplace lifecycle boundary
```

```
psannlm/                     -> separate distribution (LM training/CLI utilities)
```

## Core estimator flow (supervised)

1. **Input normalisation** via `normalise_fit_args` (dtype, finite-value boundary,
   validation splits, and shape hints).
2. **Preflight contracts** via `_fit_contracts` (configuration, device/fallback policy,
   deterministic mode, and mutually incompatible options).
3. **Scaling + shape prep** via `prepare_inputs_and_scaler`:
   - decides flatten vs preserve-shape paths
   - applies optional scalers
   - prepares metadata for prediction and streaming paths
4. **Model build** via `build_model_from_hooks`:
   - selects base (PSANN, ResPSANN, WaveResNet, SGR, GeoSparse)
   - attaches optional LSM expanders or attention
5. **Pre-optimizer forward check** requires exact prediction/target shape parity.
6. **Training** via `run_supervised_training`:
   - shared optimizer/scheduler, non-finite, metric, and fallback logic
   - structured events and standard logging
   - atomic resumable checkpoints with optimizer, RNG, and early-stopping state
7. **Prediction** reuses prepared metadata for consistent output shapes.

`psann.sklearn` stays as the stable import and checkpoint path, while the implementation lives in `psann._sklearn.*`.

## HISSO flow (episodic)

- `HISSOOptions` resolves reward, transforms, and context configuration.
- `EpisodeTrainer` runs episodes on the estimator’s device and logs rewards.
- `hisso_infer_series` and `hisso_evaluate_reward` reuse the stored episode config.

## LM flow (experimental)

- `psannlm.psannLMDataPrep` handles tokenisation + dataset packing.
- `psannlm.psannLM` exposes a compact fit/generate interface.
- The CLI / long-run training utilities live in `psannlm.lm.train.cli`.
- Trainer checkpoint/optimizer/cache/validation responsibilities live in
  `psannlm.lm.train.runtime`; `trainer.py` owns the training loop.

## Workplace platform boundary

- `TaskSpec`, `ModelSpec`, `TrainingConfig`, `DataSchema`, and `InferenceConfig`
  define JSON-safe lifecycle configuration.
- `create_model` resolves a declared backbone and composes an existing regressor or
  the task-aware `PSANNClassifier`.
- `train` calls the estimator fit path and returns a structured `TrainingRun`.
- Task adapters own target validation, task loss, probability conversion, thresholds,
  and default metrics.
- The schema boundary captures feature/output names and preprocessing contracts and
  rejects incompatible named-feature requests.
- Explicit registries resolve stable identifiers without importing optional workplace
  integrations during base-package import.
- `TorchModuleAdapter` uses the shared observable loop but deliberately promises only
  in-process training/inference.
- `ArtifactManifest` defines required native-artifact metadata fields.
- `TrainingRun.export` writes the versioned `.psann` bundle; `inspect_artifact`
  validates it without tensor loading, and `load_model` reconstructs a registered
  model through restricted `state_dict` loading.
- `InferenceResult` carries task, output, artifact, model, and run metadata.
- `InferenceRuntime` applies schema, preprocessing, task conversion, and inverse
  transforms in bounded raw-input chunks under a concurrency lock and inference mode.
- Stateful rollouts use isolated `InferenceSession` estimator copies; device pools use
  independently loaded runtimes.
- Derived exports are advertised only after alternate-batch parity checks and always
  ship a preprocessing/postprocessing contract.
- `PSANNExplainer` composes the same `InferenceRuntime` raw-input path with optional
  SHAP permutation/partition algorithms. A frozen differentiable clone is used only
  when preprocessing, layout, context, state, backbone, and activation pass the
  gradient capability gate.
- `StreamingSupervisedData` restarts a caller-owned bounded batch source for each
  logical epoch; `numpy_shard_stream` memory maps uncompressed shards.
- Accelerator capabilities are policy objects rather than implicit device probes.
  Operational events carry redacted fingerprints/configuration into optional
  caller-supplied tracking, registry, and monitoring sinks.

Phase 4 owns the safe `.psann` deployment artifact. Phase 5 owns the higher-level
batched/concurrent inference runtime, derived exports, and reference service. Phase 6
owns explicit background governance, explanation output/group semantics, and
capability-gated SHAP integration. Phase 7 owns explicit accelerator/dtype tiers,
bounded streaming, privacy-safe operational metadata, performance observations, and
supply-chain evidence.

## Design goals

- **Stable core surface**: sklearn-style estimators are the primary supported API.
- **Shared fit helpers**: keep preprocessing and training logic in `_fit_utils`.
- **Experimental isolation**: GeoSparse and LM code are clearly labeled experimental.
