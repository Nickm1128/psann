# Changelog

All notable user-facing changes are recorded here. PSANN follows semantic versioning;
release and support details live in `docs/releasing.md` and
`docs/support_policy.md`.

## Unreleased

No user-facing changes have been recorded after the `1.1.0rc1` candidate snapshot.

## 1.1.0rc1 - 2026-08-10

### Added

- Added a built-wheel workplace certification runner covering tabular regression,
  binary classification, multiclass classification, convolutional, context/state,
  and registered custom-backbone scenarios with privacy-safe evidence.
- Added multiclass `InferenceConfig(top_k=...)` results with ranked labels,
  probabilities, and class indices while retaining the complete probability matrix.
- Added native restricted-artifact reconstruction for registered custom
  `torch.nn.Module` factories, explicitly retaining experimental support and reduced
  export/explanation guarantees.
- Added a machine-readable `workplace-v1` API freeze, five task-oriented quick starts,
  a retained public `0.12.7` legacy checkpoint migration fixture, explicit native
  schema-migration tests, a zero-warning certification policy, and a clean-checkout
  CPU/CUDA release-candidate workflow.
- Added an exhaustive two-module 1.1 API inventory and a provenance-linked public
  `0.12.7` export/principal-estimator signature inventory, both enforced against
  source and installed wheels.
- Added blocking installed-wheel Linux/Windows jobs across Python 3.11-3.13,
  Python 3.13 SHAP coverage, and explicit core/SHAP floor, snapshot, and current
  dependency profiles.
- Accepted workplace lifecycle, compatibility, artifact, deployment, and deprecation
  decisions.
- Added the workplace support matrix, phased implementation roadmap, and dated
  repository baseline.
- Added typed platform contracts for model specifications, task adapters, artifact
  manifests, and inference results.
- Added `SigmoidParam` and the canonical `sigmoid` activation identifier.
- Added separate core, PSANN-LM, and scripts coverage reports.
- Added built-wheel smoke tests and scheduled CUDA validation workflow.
- Added fail-fast supervised training validation, named schedulers, detached custom
  metrics, structured training events, and a standard logging adapter.
- Added atomic, checksummed `.psann-train` checkpoints with restricted loading,
  optimizer/scheduler/AMP/RNG state, bounded retention, and exact CPU resume tests.
- Added JSON-serializable `TaskSpec`, `ModelSpec`, `TrainingConfig`, `DataSchema`, and
  `InferenceConfig` objects plus registered `create_model` and structured `train`
  orchestration.
- Added `PSANNClassifier` with binary, multiclass, and multilabel logit losses,
  probabilities, thresholds, metrics, and sklearn pipeline/grid-search compatibility.
- Added registered dense/residual/1D/2D/3D convolutional factories, GELU/SiLU
  activations, named-feature schema enforcement, and a limited arbitrary-module
  adapter.
- Added atomic, checksummed `.psann` deployment artifacts with JSON schemas,
  restricted `state_dict` loading, generic `load_model`, metadata-only inspection,
  runtime/plugin compatibility checks, and in-memory format migrations.
- Added an explicitly trusted legacy-checkpoint loader and migration tool; legacy
  class-specific save/load now emits a security and deprecation warning.
- Added `InferenceRuntime` and `InferenceResult` deployment inference with bounded
  raw-input batching, eval/inference-only execution, stable metadata, concurrent
  request isolation, and explicit stateful sessions.
- Added capability-gated dynamic-batch `torch.export` and ONNX exports with generated
  preprocessing/postprocessing contracts and parity evidence.
- Added optional explicit artifact-registry resolvers, independent device pools, and
  a FastAPI reference service with liveness, readiness, metadata, prediction, and
  input-safe structured metrics.
- Added a non-root CPU serving container, locked deployment snapshot, smoke workflow,
  and tag-gated GHCR publishing workflow.
- Added optional `psann[explain]` SHAP integration over the deployed raw-input
  contract, with explicit background governance, named task outputs, spatial/sequence
  groups, bounded permutation/partition explainers, and standard `shap.Explanation`
  results carrying artifact/run metadata.
- Added a frozen differentiable inference adapter with built-in scaler, inverse-target,
  layout, cosine-context, and classification output parity; unsupported gradient/deep
  combinations fall back with an explicit reason or fail by policy.
- Added registered intermediate-layer explanation aliases plus aggregate explanation
  drift summaries and raw-row-free offline reports.
- Added an explicit CPU/CUDA/MPS accelerator and fp32/fp16/bf16 policy with scheduled
  CUDA lifecycle, AMP, compile, export, explanation, and memory evidence.
- Added bounded restartable supervised streams and memory-mapped NumPy shard loading
  for large regression datasets.
- Added model/data fingerprints, credential rejection, redacted operational events,
  retention contracts, and optional experiment-tracker/registry/monitor hooks.
- Added portable workplace performance baselines where correctness blocks and noisy
  regressions alert by default.
- Added dependency, repository-secret, and container vulnerability workflows;
  per-distribution and image SBOM generation; and Dependabot coverage for Python,
  Actions, and Docker dependencies. Third-party security/container actions use
  immutable commit pins.

### Changed

- Corrected compatibility claims so internal `0.13` through `0.16` workplace phase
  labels are not presented as published artifact producers; rewritten manifest
  metadata tests are explicitly synthetic schema validation.
- Linked every support-matrix row to an executable test, workflow, retained fixture,
  or accepted policy, and narrowed Windows CUDA and macOS arm64 from unproven support
  tiers to unsupported/experimental status.
- Made PSANN-LM's 35% Alpha coverage floor and the release helper's 60% focused floor
  blocking while retaining aggregate scripts coverage as an explicitly observational
  report.
- Selected `1.1.0rc1` / `v1.1.0rc1` as the unique release-candidate identity and
  reserved `1.1.0` / `v1.1.0` for general availability. The historical `v1.0.0` tag
  remains immutable and is not reused by this release line.
- Made the release helper fail closed on dirty trees, version drift, missing
  changelog entries, reused Git/PyPI identities, incompatible PSANN-LM metadata,
  malformed artifact sets, failed Twine/package smoke checks, and unconfirmed
  uploads.
- Made `psannlm.__version__` report its own distribution version and constrained the
  1.1 LM line to `psann>=1.1.0rc1,<1.2` with clear import-time mismatch errors.
- Raised the development line to Python 3.11+ and the workplace dependency floors
  documented in the support matrix.
- Made Black the sole formatter and expanded Ruff to blocking import-order and
  high-signal `E4`/`E7` rules.
- Added coordinated core/LM version sources that the release helper synchronizes
  while each installed distribution reports its own bundled version.
- Split LM trainer runtime helpers and LM benchmark CLI parsing out of their previous
  800-line modules.
- Callback and gradient-hook failures now raise by default; non-finite and
  accelerator/AMP/compile fallback behavior is controlled by explicit policies.
- Training and validation inputs now reject NaN/infinity, and prediction/target shape
  parity is checked before optimizer creation.
- Test discovery is constrained to `tests/`, preventing generated reports, SBOM
  staging trees, and build outputs from shadowing real test modules.

### Repository

- Notebook outputs and execution counts are now prohibited by the hygiene audit.
- Undocumented top-level files and Python files at or above 800 lines are blocking.
- Removed the empty `notebooks/Untitled.ipynb` scratch notebook.

## 0.12.4

- Consolidated the estimator implementation under `src/psann/_sklearn/`.
- Kept `psann.sklearn` as the stable estimator facade and checkpoint import path.
- Split language-model functionality into the separate `psannlm` distribution.

Earlier migration notes remain in `docs/migration.md`.
