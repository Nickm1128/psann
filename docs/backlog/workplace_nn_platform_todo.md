# Workplace Neural-Network Platform Roadmap

Status: Active roadmap; Phases 0-7 complete, Phase 8 promotion blocked, Phase 9 in progress; promotion workflows archived

Last reviewed: 2026-08-11

Scope: Core `psann` estimators and supporting workplace model lifecycle. The separate
`psannlm` distribution remains out of scope except where shared repository, packaging,
or security gates apply.

## Target Outcome

PSANN should provide a stable, documented workflow for creating, training, evaluating,
explaining, exporting, loading, and deploying neural-network models in a work
environment.

The target user experience is:

```python
spec = psann.ModelSpec(
    task="regression",
    backbone="respsann",
    input_schema=...,
    output_schema=...,
)

model = psann.create_model(spec)
run = psann.train(model, train_data, validation_data=validation_data)
artifact = run.export("artifacts/demand_forecast.psann")

deployed = psann.load_runtime(
    artifact,
    config=psann.InferenceConfig(batch_size=256, device="cpu"),
)
predictions = deployed.predict(batch).values
explanation = deployed.explain(batch, background=reference_data)
```

The existing sklearn-style estimators must remain usable throughout the transition.
The high-level workplace API should wrap or compose them rather than duplicating the
training implementation.

## Definition of Workplace-Ready

The milestone is complete when all of the following are true:

- A user can select a supported task and backbone through one stable model
  specification.
- Regression, binary classification, multiclass classification, and multilabel
  classification have explicit estimator contracts.
- Training rejects invalid configurations early and handles non-finite values
  according to a documented policy.
- A training run can be resumed exactly enough for operational use.
- A deployment artifact is versioned, inspectable, checksummed, migration-aware, and
  loadable without arbitrary Python unpickling.
- Inference is batched, schema-validated, stateless by default, and safe for concurrent
  request handling.
- SHAP explanations are available through an optional dependency and preserve feature,
  shape, context, and output metadata.
- CPU correctness is required, supported accelerators have an explicit test matrix,
  and performance regressions are measured.
- CI, documentation, security guidance, package smoke tests, and release checks pass
  from a clean environment.

## Priority Labels

- **P0**: Blocks a trustworthy baseline or makes a documented API incorrect.
- **P1**: Required for the workplace-ready milestone.
- **P2**: Important hardening or usability work that may follow the first end-to-end
  vertical slice.

## Phase Summary

| Phase | Outcome | Depends on |
| --- | --- | --- |
| 0 | Contracts, support policy, and baseline decisions | None |
| 1 | Clean repository and enforceable quality gates | Phase 0 |
| 2 | Correct, observable, resumable training core | Phase 1 |
| 3 | Unified task, model, and data API | Phases 0-2 |
| 4 | Safe versioned model and training artifacts | Phases 2-3 |
| 5 | Deployment-grade inference and export | Phase 4 |
| 6 | First-class SHAP explainability | Phases 3-5 |
| 7 | Accelerator, scale, security, and operations hardening | Phases 4-6 |
| 8 | Workplace certification and stable release | All prior phases |
| 9 | Resolve release-review findings and promote an exact candidate | Phase 8 |

## Roadmap Status

| Phase | Status | Tracking issue |
| --- | --- | --- |
| 0 | Complete (2026-07-27) | [#2](https://github.com/Nickm1128/psann/issues/2) |
| 1 | Complete (2026-07-27) | [#3](https://github.com/Nickm1128/psann/issues/3) |
| 2 | Complete (2026-07-27) | [#4](https://github.com/Nickm1128/psann/issues/4) |
| 3 | Complete (2026-07-27) | [#5](https://github.com/Nickm1128/psann/issues/5) |
| 4 | Complete (2026-07-27) | [#6](https://github.com/Nickm1128/psann/issues/6) |
| 5 | Complete (2026-07-27) | [#7](https://github.com/Nickm1128/psann/issues/7) |
| 6 | Complete (2026-07-27) | [#8](https://github.com/Nickm1128/psann/issues/8) |
| 7 | Complete (2026-07-27) | [#9](https://github.com/Nickm1128/psann/issues/9) |
| 8 | Implementation complete; promotion blocked | [#10](https://github.com/Nickm1128/psann/issues/10) |
| 9 | Planned | Not yet created |

## Phase 0 - Freeze Contracts and Decisions

Goal: agree on the platform boundaries before implementing new public surfaces.

### Product and support decisions

- [x] **P0** Write an ADR for the high-level lifecycle API: `ModelSpec`,
  `create_model`, `train`, `TrainingRun`, `export`, `load_model`, and `explain`.
- [x] **P0** Define which existing estimator methods remain stable and which new API
  owns orchestration.
- [x] **P0** Define the supported task set for the first workplace milestone:
  regression, binary classification, multiclass classification, and multilabel
  classification.
- [x] **P0** Define the stable backbone set. Keep GeoSparse experimental until it passes
  the same artifact, inference, and task-contract tests as core backbones.
- [x] **P0** Decide whether Python 3.9 and 3.10 remain supported. Current SHAP releases
  require Python 3.11+, so either use environment-marked SHAP versions or raise the
  platform minimum.
- [x] **P1** Publish a compatibility matrix covering Python, NumPy, PyTorch,
  scikit-learn, SHAP, operating system, and accelerator support.
- [x] **P1** Define semantic-versioning rules for model artifacts separately from the
  Python package version.
- [x] **P1** Define what "deployment support" means for native Python, `torch.export`,
  ONNX, and a reference HTTP service. Only promise formats that pass parity tests.

### Baseline and issue slicing

- [x] **P0** Record the current validation baseline:
  - fast CPU suite result;
  - core package coverage;
  - Ruff, formatter, and mypy status;
  - current long-file and notebook-output inventory;
  - save/load round-trip coverage;
  - supported estimator and activation matrix.
- [x] **P0** Convert each P0 and P1 checklist group in this roadmap into a GitHub issue
  with an owner, acceptance criteria, and dependency links.
- [x] **P1** Add a roadmap status table linking phases to their tracking issues.
- [x] **P1** Define a deprecation window for legacy whole-object checkpoints and any
  estimator parameters replaced by structured configuration.

### Exit criteria

- [x] The lifecycle, artifact, task, compatibility, and deprecation ADRs are approved.
- [x] Every P0/P1 group has a trackable issue and explicit acceptance criteria.
- [x] The initial support matrix and baseline report are checked into `docs/`.

### Phase 0 evidence

- Accepted decisions: [`docs/adr/`](../adr/README.md)
- Compatibility and capability contract:
  [`docs/workplace_support_matrix.md`](../workplace_support_matrix.md)
- Measured starting point:
  [`docs/workplace_platform_baseline_2026-07-27.md`](../workplace_platform_baseline_2026-07-27.md)
- Deprecation policy: [`docs/deprecation_policy.md`](../deprecation_policy.md)
- Phase trackers with owners, acceptance criteria, and dependency links:
  [#2](https://github.com/Nickm1128/psann/issues/2),
  [#3](https://github.com/Nickm1128/psann/issues/3),
  [#4](https://github.com/Nickm1128/psann/issues/4),
  [#5](https://github.com/Nickm1128/psann/issues/5),
  [#6](https://github.com/Nickm1128/psann/issues/6),
  [#7](https://github.com/Nickm1128/psann/issues/7),
  [#8](https://github.com/Nickm1128/psann/issues/8),
  [#9](https://github.com/Nickm1128/psann/issues/9), and
  [#10](https://github.com/Nickm1128/psann/issues/10).

## Phase 1 - Establish a Clean, Enforceable Baseline

Goal: make the repository's automated quality claims match its actual state.

### Formatting, linting, and typing

- [x] **P0** Fix the current Ruff failures.
- [x] **P0** Format the current Black-drifted files.
- [x] **P0** Fix duplicate-module discovery in the configured mypy command.
- [x] **P0** Make `make lint`, pre-commit, and CI run the same checks over the same
  directories.
- [x] **P1** Choose one canonical formatter or document why both Ruff formatting and
  Black are needed; prevent formatter ping-pong.
- [x] **P1** Increase Ruff beyond `F,E9` in staged steps, with documented and narrowly
  scoped ignores.
- [x] **P1** Establish a typed public boundary for model specs, artifacts, task
  adapters, and inference results even if internal modules remain gradually typed.

### Tests and coverage

- [x] **P0** Scope coverage reporting so core library coverage is not diluted by
  operational scripts.
- [x] **P0** Replace CI's `--fail-under=0` with a ratcheted threshold. Start no lower
  than the measured core baseline and raise it as new platform modules land.
- [x] **P1** Publish separate coverage reports for `src/psann`, `psannlm`, and scripts
  instead of one misleading aggregate.
- [x] **P1** Add a supported-Python matrix based on the Phase 0 compatibility decision.
- [x] **P1** Run package smoke tests against built wheels, not only editable installs.
- [x] **P1** Add a scheduled accelerator job while retaining CPU-only pull-request
  checks.

### Repository and documentation hygiene

- [x] **P0** Strip committed outputs and execution counts from tracked notebooks.
- [x] **P0** remove, rename, or document `notebooks/Untitled.ipynb`.
- [x] **P0** Extend the hygiene audit to detect notebook outputs and undocumented
  top-level scratch files.
- [x] **P0** Reconcile the long-file audit output with
  `docs/repo_hygiene_audit.md`.
- [x] **P1** Refactor `psannlm/lm/train/trainer.py` and
  `scripts/_bench_lm_bases/main.py` only along clear responsibility boundaries.
- [x] **P1** Update active API documentation for target scaling, warm starts, AMP,
  compilation, context builders, and current LM package boundaries.
- [x] **P1** Single-source the package version used by `pyproject.toml`,
  `psann/__init__.py`, and `psannlm`.
- [x] **P1** Add or refresh `CHANGELOG.md`, `SECURITY.md`, release instructions, and
  supported-version policy.
- [x] **P2** Add dependency constraints or a tested constraints file for reproducible
  workplace installations while keeping the base wheel lean.

### Exit criteria

- [x] `make lint`, `make test-fast`, the hygiene audit, and package smoke tests pass
  from a clean checkout.
- [x] CI rejects formatting, typing, notebook-output, and coverage regressions.
- [x] Active documentation matches the implemented public API.
- [x] No unexplained tracked scratch files or generated outputs remain.

### Phase 1 evidence

- Canonical local and CI quality entry point: [`tools/quality.py`](../../tools/quality.py),
  shared by `Makefile`, pre-commit, and the CI quality job.
- Fast CPU validation: 230 passed, 1 skipped, and 37 deselected on 2026-07-27.
- Scoped coverage: core 70%, PSANN-LM 39%, and scripts 1%, with a blocking 70% core
  threshold and separate XML reports.
- Package validation: isolated source-distribution and wheel builds for `psann` and
  `psannlm`, followed by fresh-environment import, version, and package-boundary smoke
  tests.
- Compatibility automation: CPU CI for Python 3.11, 3.12, and 3.13 plus a scheduled
  accelerator workflow.
- Hygiene enforcement: notebook-output, undocumented top-level-file, and 800-line
  checks pass; the two named long files were split along runtime and CLI boundaries.
- Public contract and operations documentation:
  [`src/psann/platform/`](../../src/psann/platform/),
  [`docs/API.md`](../API.md), [`docs/releasing.md`](../releasing.md),
  [`docs/support_policy.md`](../support_policy.md), [`SECURITY.md`](../../SECURITY.md),
  and [`constraints/`](../../constraints/).

## Phase 2 - Correct and Harden the Training Core

Goal: make training behavior explicit, validated, observable, and resumable.

### Correctness and configuration validation

- [x] **P0** Resolve `loss_reduction="none"`:
  - either reject it for optimizer-driven training with a precise error; or
  - define a scalar backward reduction while retaining unreduced values for metrics.
- [x] **P0** Add regression tests for every documented built-in loss and reduction.
- [x] **P0** Validate optimizer names instead of silently falling back to Adam.
- [x] **P0** Validate activation names, loss names, scheduler names, dimensions,
  learning rates, batch sizes, patience, device requests, and mutually incompatible
  options before model construction.
- [x] **P0** Introduce a non-finite policy with `error`, `skip_step`, and explicitly
  opt-in `continue` behaviors. Default workplace behavior should fail clearly.
- [x] **P1** Define missing-value and infinite-input policies at the data boundary.
- [x] **P1** Ensure target and prediction shape validation happens before the first
  optimizer step.
- [x] **P1** Test warm-start behavior after compatible and incompatible parameter
  changes.

### Observability and callbacks

- [x] **P0** Stop swallowing callback and gradient-hook exceptions by default.
- [x] **P1** Replace print-only training output with structured events and an optional
  standard logger adapter.
- [x] **P1** Record effective device, dtype, seed, optimizer, learning rates, parameter
  counts, data shapes, compile/AMP fallback decisions, and epoch metrics.
- [x] **P1** Give compile and accelerator fallbacks explicit warning/error policies.
- [x] **P1** Define stable callback events for train start/end, epoch start/end,
  validation end, checkpoint, early stop, and failure.
- [x] **P2** Add pluggable metric collections and user-defined metrics without coupling
  them to loss computation.

### Resume and reproducibility

- [x] **P1** Add a training checkpoint distinct from a deployment artifact.
- [x] **P1** Persist model state, optimizer state, scheduler state, AMP scaler state,
  epoch/step counters, best metric, early-stopping state, and training history.
- [x] **P1** Persist Python, NumPy, Torch, CUDA, and sampler RNG state needed for a
  practical resume contract.
- [x] **P1** Add `resume_from=` and test loss/parameter continuity after interruption.
- [x] **P1** Add a deterministic-mode option and document the performance and
  cross-platform limits.
- [x] **P2** Add atomic checkpoint writes and retention policies for latest, best, and
  periodic checkpoints.

### Exit criteria

- [x] All documented configuration values either work or fail before training with a
  specific error.
- [x] Non-finite behavior, callback failures, and compile fallbacks are observable.
- [x] Interrupted training resumes within documented tolerances.
- [x] Training checkpoint and deployment artifact responsibilities are separate.

### Phase 2 evidence

- Fail-fast configuration, input-finiteness, prediction-shape, warm-start, device,
  compile, AMP, optimizer, scheduler, and loss contracts:
  [`src/psann/estimators/_fit_contracts.py`](../../src/psann/estimators/_fit_contracts.py)
  and [`tests/test_training_contracts.py`](../../tests/test_training_contracts.py).
- Structured events, logger integration, custom metrics, explicit callback failure
  handling, and observable non-finite policies:
  [`src/psann/training.py`](../../src/psann/training.py) and
  [`src/psann/training_events.py`](../../src/psann/training_events.py).
- Restricted-load, checksummed, atomic `.psann-train` checkpoints with latest, best,
  and periodic retention; model, optimizer, scheduler, AMP, early-stop, history, and
  RNG state are covered by
  [`tests/test_training_resume.py`](../../tests/test_training_resume.py).
- Exact interrupted-versus-uninterrupted CPU loss and parameter continuity is covered
  by the deterministic resume regression test. Cross-device and cross-platform limits
  are documented in [`docs/training_core.md`](../training_core.md).
- Phase-focused validation: 51 tests passed. Full fast CPU validation: 281 passed,
  1 skipped, and 37 deselected on 2026-07-27. Core branch coverage is 71%, above the
  blocking 70% threshold.
- Release-facing validation: canonical Ruff/Black/mypy checks, pre-commit, strict
  repository hygiene, source/wheel builds for both distributions, and built-wheel
  package-boundary smoke tests pass.

## Phase 3 - Introduce a Unified Task, Model, and Data API

Goal: support conventional workplace neural-network tasks without making users assemble
raw Torch loops.

### Structured configuration and factories

- [x] **P1** Add serializable `TaskSpec`, `ModelSpec`, `TrainingConfig`,
  `DataSchema`, and `InferenceConfig` types.
- [x] **P1** Add a registry-backed `create_model(spec)` factory with stable backbone
  identifiers.
- [x] **P1** Add `train(model, ...) -> TrainingRun` while preserving estimator
  `fit(...) -> self` behavior for sklearn compatibility.
- [x] **P1** Define a public backbone protocol for PSANN backbones and registered custom
  `torch.nn.Module` implementations.
- [x] **P1** Define activation, normalization, dropout, optimizer, scheduler, loss, and
  metric registries with validation and serialization rules.
- [x] **P2** Support carefully scoped plugin registration without importing workplace
  integrations in the base package.

### Task support

- [x] **P1** Add `PSANNClassifier` with sklearn-compatible `classes_`,
  `predict`, `predict_proba`, and `score`.
- [x] **P1** Implement binary classification with logits plus BCE-with-logits.
- [x] **P1** Implement multiclass classification with logits plus cross-entropy.
- [x] **P1** Implement multilabel classification with explicit threshold policy.
- [x] **P1** Add regression, classification, and multilabel metric sets with correct
  target validation.
- [x] **P1** Make output activation and probability conversion task-owned, not
  backbone-owned.
- [x] **P1** Add estimator checks for cloning, parameters, fitted-state errors, feature
  counts, and pipeline/grid-search compatibility.

### Standard neural-network breadth

- [x] **P1** Promote stable dense MLP, residual MLP, and 1D/2D/3D convolutional
  backbones through the factory.
- [x] **P1** Support standard ReLU, tanh, sigmoid, GELU, and SiLU activations alongside
  PSANN activations.
- [x] **P1** Add explicit dropout and normalization configuration where each backbone
  supports it.
- [x] **P1** Document which attention, spectral, stateful, and convolutional options
  are compatible with each task and shape.
- [x] **P2** Add an arbitrary-module adapter for advanced users, with a deliberately
  smaller artifact/export guarantee than registered backbones.

### Data and schema contract

- [x] **P1** Capture `n_features_in_`, `feature_names_in_`, output names, input shape,
  data format, dtype, and task metadata at fit time.
- [x] **P1** Preserve pandas column names when pandas inputs are provided.
- [x] **P1** Reject missing, reordered, duplicated, or unexpected named features
  according to a configurable schema policy.
- [x] **P1** Persist preprocessing and target-scaling contracts as structured data.
- [x] **P2** Define extension points for categorical encoding and missing-value
  imputation without making pandas or scikit-learn mandatory base dependencies.

### Exit criteria

- [x] One high-level API trains and evaluates regression and classification models.
- [x] Existing regressors remain backward compatible and share the same internal
  training path.
- [x] Registered standard and PSANN backbones pass a common task/shape test matrix.
- [x] Feature and output schemas survive save/load and reject incompatible requests.

### Phase 3 evidence

- Serializable lifecycle configuration and JSON round trips:
  [`src/psann/platform/specs.py`](../../src/psann/platform/specs.py).
- Registered backbone and training-component identifiers, lazy plugin registration,
  and schema-transform extension points:
  [`src/psann/platform/registry.py`](../../src/psann/platform/registry.py).
- Task-owned target validation, logit losses, probabilities, thresholds, and
  regression/classification/multilabel metrics:
  [`src/psann/platform/tasks.py`](../../src/psann/platform/tasks.py) and
  [`src/psann/_sklearn/classifier.py`](../../src/psann/_sklearn/classifier.py).
- High-level `create_model` and `train(...)->TrainingRun` compose the existing
  estimator fit path; arbitrary modules use the same observable loop with explicitly
  limited artifact guarantees:
  [`src/psann/platform/lifecycle.py`](../../src/psann/platform/lifecycle.py) and
  [`src/psann/platform/module_adapter.py`](../../src/psann/platform/module_adapter.py).
- The common regression, binary, multiclass, and multilabel task/shape matrix covers
  all eight registered backbones. Sklearn clone, pipeline, grid-search, fitted-state,
  pandas schema, preprocessing contract, trusted save/load, and extension behavior
  are covered by
  [`tests/test_workplace_platform.py`](../../tests/test_workplace_platform.py).
- Phase-focused validation: 57 tests passed. Full fast CPU validation: 336 passed,
  1 skipped, and 37 deselected on 2026-07-27. Core branch coverage is 72%, above the
  blocking 70% threshold.
- Release-facing validation: canonical Ruff/Black/mypy checks, strict repository
  hygiene, source/wheel builds for both distributions, and installed-wheel lifecycle
  import/factory smoke tests pass.
- Public behavior, compatibility matrices, schema policies, and the Phase 4 artifact
  boundary are documented in [`docs/workplace_api.md`](../workplace_api.md).

## Phase 4 - Build Safe, Versioned Artifacts

Goal: replace whole-object pickle checkpoints with portable, inspectable artifacts.

### Artifact format

- [x] **P0** Define a versioned artifact manifest and JSON schema.
- [x] **P0** Store registered backbone/task identifiers, JSON-safe configuration, and
  tensor state separately.
- [x] **P0** Use `state_dict`-oriented loading with restricted deserialization. Do not
  require `weights_only=False` for new artifacts.
- [x] **P1** Include feature/output schemas, preprocessing state, task metadata,
  package/runtime versions, training-run identifier, and artifact creation time.
- [x] **P1** Include per-file cryptographic hashes and validate them before loading.
- [x] **P1** Make artifact writes atomic and detect truncated or corrupted bundles.
- [x] **P1** Define required versus optional manifest fields and forward-compatible
  handling of unknown optional metadata.
- [x] **P2** Support an optional human-readable model card in the bundle.

### Loading, migration, and trust

- [x] **P0** Add generic `psann.load_model(path, device=...)` dispatch using the
  artifact registry.
- [x] **P0** Add actionable errors for unsupported artifact versions, missing plugins,
  incompatible runtime versions, and checksum failures.
- [x] **P1** Add artifact migrations with fixtures for every supported historical
  schema version.
- [x] **P1** Keep legacy checkpoint loading behind an explicit
  `trusted_legacy_checkpoint=True` path with a security warning and migration tool.
- [x] **P1** Document the trust model: locally created artifacts, internally signed
  artifacts, and untrusted external artifacts.
- [x] **P1** Test cross-device, cross-process, wheel-installed, and supported-version
  load parity.
- [x] **P2** Define extension metadata for external model registries without coupling
  the artifact format to one vendor.

### Exit criteria

- [x] New deployment artifacts load without arbitrary Python object unpickling.
- [x] A generic loader reconstructs every registered core model.
- [x] Corruption, unsupported versions, and missing extensions fail safely.
- [x] Legacy artifacts have a documented, explicitly trusted migration path.

### Phase 4 evidence

- Native format, schema, atomic ZIP/checksum validation, restricted tensor state,
  inspection, compatibility checks, and migrations:
  [`src/psann/platform/artifact_schema.py`](../../src/psann/platform/artifact_schema.py),
  [`src/psann/platform/artifact_io.py`](../../src/psann/platform/artifact_io.py),
  and [`src/psann/platform/artifacts.py`](../../src/psann/platform/artifacts.py).
- Registered estimator reconstruction preserves fitted schema, task, preprocessing,
  device, and non-tensor WaveResNet schedule metadata through
  [`src/psann/platform/artifact_models.py`](../../src/psann/platform/artifact_models.py).
- Corruption-before-deserialization, atomic replacement, preview-format migration,
  runtime/plugin errors, all eight core backbones, all four tasks, cross-process
  parity, explicit legacy trust, and migration coverage:
  [`tests/test_model_artifacts.py`](../../tests/test_model_artifacts.py). Scheduled
  CPU-to-CUDA mapping is covered in
  [`tests/gpu/test_torch_basic.py`](../../tests/gpu/test_torch_basic.py).
- The built-wheel smoke test performs train, export, metadata inspection, generic load,
  and numerical parity from the installed `0.13.0` wheel.
- Phase-focused validation: 26 tests passed. Full fast CPU validation: 362 passed,
  1 skipped, and 38 deselected on 2026-07-27. Core branch coverage is 73%, above the
  blocking 70% threshold.
- Release-facing validation: canonical Ruff/Black/mypy, all pre-commit hooks, strict
  repository hygiene, Markdown-link and workflow-YAML checks, source/wheel builds for
  both distributions, and installed-wheel artifact smoke tests pass.
- Format, trust, migration, custom-component, and legacy guidance:
  [`docs/artifacts.md`](../artifacts.md), [`SECURITY.md`](../../SECURITY.md), and
  [`docs/migration.md`](../migration.md).

## Phase 5 - Deliver Deployment-Grade Inference

Goal: make loaded models safe and predictable inside batch jobs and online services.

### Inference runtime

- [x] **P1** Add chunked/batched prediction with configurable batch size and device
  transfer policy.
- [x] **P1** Use an inference-only execution path and guarantee eval mode.
- [x] **P1** Make ordinary deployed inference stateless and concurrency-safe.
- [x] **P1** Put stateful/streaming behavior behind an explicit session object whose
  lifecycle cannot leak between requests.
- [x] **P1** Apply schema validation, scaling, layout transforms, context construction,
  model execution, and inverse target transforms through one public inference adapter.
- [x] **P1** Define prediction result metadata, including output names, task, artifact
  version, and optional model/run identifiers.
- [x] **P1** Add deterministic request tests, concurrent request tests, and large-batch
  memory tests.
- [x] **P2** Add configurable device pools only after the single-device runtime is
  stable.

### Export and serving

- [x] **P1** Add one mandatory native PSANN deployment format.
- [x] **P1** Evaluate `torch.export` and ONNX for each stable backbone; expose only the
  formats that pass numerical parity and shape-dynamic tests.
- [x] **P1** Include preprocessing or provide a generated preprocessing contract for
  non-native exports.
- [x] **P1** Add a reference service with `/health`, `/ready`, artifact metadata, and
  batched prediction endpoints.
- [x] **P1** Add structured request metrics for latency, batch size, errors, device,
  and artifact identity without logging raw sensitive inputs.
- [x] **P1** Publish a minimal container image, locked dependency set, and artifact
  mounting/loading instructions.
- [x] **P2** Add adapters for common internal model registries or serving stacks as
  optional integrations.

### Exit criteria

- [x] Native artifact predictions match the fitted estimator within documented
  tolerances.
- [x] Supported exports pass parity tests for every declared task/backbone combination.
- [x] Concurrent stateless requests cannot mutate shared model state.
- [x] A clean container can load an artifact and serve a health check plus predictions.

### Phase 5 evidence

- The public raw-input runtime, stable result metadata, bounded chunking, eval and
  inference-only execution, shared-model concurrency lock, isolated session lifecycle,
  and independent device replicas live in
  [`src/psann/platform/inference.py`](../../src/psann/platform/inference.py).
- Explicit URI-to-local-artifact adapters preserve normal native-bundle validation
  without vendor coupling or implicit discovery in
  [`src/psann/platform/integrations.py`](../../src/psann/platform/integrations.py).
- `torch.export` and ONNX capability evaluation requires numerical parity plus an
  alternate dynamic batch before advertising or writing a derived export. Each output
  receives a generated preprocessing/postprocessing contract through
  [`src/psann/platform/exports.py`](../../src/psann/platform/exports.py).
- The regression, binary, multiclass, and multilabel matrix certifies both derived
  formats across all eight registered stable backbones with `atol=1e-5` and
  `rtol=1e-4`. Base installs skip ONNX certification until `psann[export]` is present.
- The optional reference worker provides liveness, readiness, reduced artifact
  metadata, aggregate metrics, and batched prediction without logging raw inputs:
  [`src/psann/serving.py`](../../src/psann/serving.py).
- The non-root CPU image and locked Python 3.11 dependency snapshot are defined by
  [`deploy/Dockerfile`](../../deploy/Dockerfile) and
  [`constraints/deployment-py311.txt`](../../constraints/deployment-py311.txt).
  Local clean-container validation built `psann-serving:0.14.0`, mounted a native
  artifact read-only, and passed health, readiness, metadata, and prediction requests.
  The container workflow repeats that smoke and publishes GHCR images on version tags.
- Runtime, concurrency, state/session, pooling, resolver, derived-export, service,
  metrics, and container-contract coverage lives in
  [`tests/test_deployment_inference.py`](../../tests/test_deployment_inference.py),
  [`tests/test_deployment_exports.py`](../../tests/test_deployment_exports.py), and
  [`tests/test_reference_service.py`](../../tests/test_reference_service.py). The
  focused Phase 5 matrix passes 92 tests; the full non-slow, non-GPU suite passes 454
  tests with one environment-dependent skip and 38 scheduled/slow deselections. Core
  branch coverage is 73% against the blocking 70% threshold.
- Canonical Ruff, Black, mypy, pre-commit, repository hygiene, Markdown-link, and
  workflow-YAML gates pass. Fresh `0.14.0` source/wheel builds and isolated installed-
  wheel smoke tests pass for both `psann` and `psannlm`.
- The scheduled CUDA suite now covers native CPU-to-CUDA artifact mapping plus bounded
  `InferenceRuntime` prediction on CUDA. No local CUDA claim is made by this CPU
  validation run.
- Public behavior, tolerances, output semantics, optional dependencies, registry
  boundary, service endpoints, metrics, and mounting instructions are documented in
  [`docs/deployment.md`](../deployment.md).

## Phase 6 - Add First-Class SHAP Explainability

Goal: provide reliable Shapley analyses over the same raw-input contract used for
deployment inference.

### Dependency and API design

- [x] **P1** Add a `psann[explain]` optional extra with Python-version-compatible SHAP
  constraints based on the Phase 0 support decision.
- [x] **P1** Add `model.make_explainer(...)` and `model.explain(...)` or equivalent
  high-level functions without importing SHAP at base-package import time.
- [x] **P1** Return standard `shap.Explanation` objects plus PSANN artifact/run
  metadata.
- [x] **P1** Define background-data policies: explicit data, sampled reference data,
  persisted summary, and prohibited implicit use of training data.
- [x] **P1** Define output selection for regression, binary probability/logit,
  multiclass, multilabel, and multi-output regression.
- [x] **P1** Persist explainer configuration separately from the deployment artifact
  unless an explicitly approved background summary is included.

### Model-agnostic explanations

- [x] **P1** Implement the first vertical slice using the public raw-input prediction
  adapter and SHAP's model-agnostic interface.
- [x] **P1** Preserve feature names, output names, original spatial/sequence shapes,
  and data-format metadata in explanations.
- [x] **P1** Add independent, partitioned, and domain-specific masker configuration
  where supported.
- [x] **P1** Define grouped features for time steps, channels, and spatial regions so
  flattening does not silently create misleading independent-feature games.
- [x] **P1** Add batching and evaluation limits to prevent explanation requests from
  exhausting memory or service capacity.

### Gradient-based explanations

- [x] **P1** Add a public differentiable Torch inference adapter that can include
  supported preprocessing and layout transforms.
- [x] **P1** Ensure the adapter freezes parameter updates and state updates and restores
  model mode safely.
- [x] **P1** Support SHAP DeepExplainer/GradientExplainer only for tested backbone,
  activation, preprocessing, and context combinations.
- [x] **P1** Detect non-differentiable custom scalers/context builders and fall back to
  model-agnostic explanations with an explicit reason.
- [x] **P2** Support intermediate-layer explanations through registered layer names
  rather than private module traversal.

### Validation and governance

- [x] **P1** Test attribution and base-value shapes across single- and multi-output
  tasks.
- [x] **P1** Test additivity within algorithm-specific tolerances.
- [x] **P1** Test explanation consistency before and after artifact save/load.
- [x] **P1** Test deterministic behavior for fixed background data and seed.
- [x] **P1** Test context-required, stateful, convolutional, and sequence failure or
  support paths explicitly.
- [x] **P1** Document correlation, causal-interpretation, background-selection, and
  sensitive-feature limitations.
- [x] **P2** Add explanation drift summaries and offline reporting helpers after core
  attribution correctness is certified.

### Exit criteria

- [x] A deployed tabular regression or classification artifact can produce named SHAP
  explanations from raw inputs.
- [x] Supported spatial and sequence explanations preserve meaningful feature groups.
- [x] Gradient explainers never bypass required preprocessing silently.
- [x] Explanation tests cover shapes, additivity, determinism, persistence parity, and
  state isolation.

### Phase 6 evidence

- `psann[explain]` installs a Python-minor-aware SHAP range while the default package
  remains importable without SHAP.
- `InferenceRuntime.make_explainer` and `InferenceRuntime.explain` use the deployed
  raw-input adapter. Results contain a standard `shap.Explanation`, task/output names,
  feature groups, data format, artifact/run identity, limits, additivity error, and
  fallback metadata.
- Explicit, sampled, and persistence-approved background policies are serializable
  separately from the native deployment artifact. Training data is never selected
  implicitly.
- Model-agnostic permutation/partition paths cover tabular, sequence, and spatial
  input games. The frozen differentiable adapter preserves built-in preprocessing,
  inverse target scaling, layout, cosine context, and classification output
  conversion; unsupported combinations fall back with a reason or fail by policy.
- Registered intermediate-layer aliases, aggregate explanation-drift summaries, and
  raw-row-free JSON reports complete the Phase 6 P2 scope.
- `tests/test_explainability.py` passes 29 tests covering all declared tasks,
  probability/logit selection, shape/name/additivity/determinism, artifact parity,
  state isolation, preprocessing/layout parity, capability gates, groups, limits, and
  reporting. The complete fast CPU suite passes 483 tests with one optional skip, and
  scoped core coverage remains above its blocking threshold at 74%. CI runs the
  explanation suite with the Python 3.11 and 3.12 SHAP dependency bands.
- Strict Ruff, Black, mypy, repository-hygiene, workflow-YAML, local-link, fresh
  `0.15.0` source/wheel build, and isolated installed-wheel smoke gates pass.
- The contract, examples, privacy boundary, causal/correlation limitations, and
  supported gradient combinations are documented in
  [`docs/explainability.md`](../explainability.md).

## Phase 7 - Harden Accelerators, Scale, Security, and Operations

Goal: certify that the platform behaves reliably beyond local CPU experimentation.

### Accelerator and performance matrix

- [x] **P1** Add scheduled CUDA tests for forward, backward, save/load, resume,
  inference, AMP, and supported export paths.
- [x] **P1** Define and test supported fp32, bf16, and fp16 combinations.
- [x] **P1** Make unsupported AMP, compile, device, and dtype combinations fail or
  degrade according to documented policy.
- [x] **P1** Add performance baselines for training throughput, inference latency,
  memory, artifact load time, and explanation cost.
- [x] **P1** Add regression tolerances that distinguish correctness gates from noisy
  performance alerts.
- [x] **P2** Evaluate MPS support and document it as supported, experimental, or
  unsupported.
- [x] **P2** Evaluate distributed training after stabilizing resume, artifacts,
  logging, and single-GPU behavior. Decision: defer the public DDP contract until
  rank ownership, cursor-aware resume, event ordering, and single-writer promotion
  are specified and tested.

### Data and operational robustness

- [x] **P1** Add large-dataset loading and streaming interfaces that do not require all
  training data in memory.
- [x] **P1** Test empty inputs, single rows, very large batches, missing columns,
  non-contiguous arrays, mixed dtypes, and malformed contexts.
- [x] **P1** Add model and data fingerprinting that avoids storing sensitive raw data.
- [x] **P1** Define retention and redaction policies for histories, checkpoints,
  explanations, and service logs.
- [x] **P1** Add dependency and container vulnerability scanning.
- [x] **P1** Generate an SBOM for release artifacts and images.
- [x] **P1** Document secrets handling and prohibit credentials in model manifests,
  logs, and promoted benchmark summaries.
- [x] **P2** Add optional hooks for experiment tracking, model registries, and
  monitoring without making them core runtime dependencies.

### Exit criteria

- [x] Every supported accelerator/dtype combination has an automated evidence path;
  release promotion requires a recent green artifact from the scheduled hardware job.
- [x] Performance regressions are visible and correctness remains the blocking gate.
- [x] Release artifacts and containers have vulnerability and SBOM evidence.
- [x] Operational metadata is useful without exposing raw sensitive inputs or secrets.

### Phase 7 evidence

- `accelerator_support_matrix()` and the scheduled accelerator workflow define the
  stable CPU/CUDA fp32 paths, CUDA AMP fp16/bf16 paths, separate CUDA compile path,
  experimental MPS fp32 path, and explicit unsupported combinations. The CUDA job
  covers forward/backward, save/load, resume, inference, export, explanation, and
  memory evidence; the MPS observation is non-blocking.
- This Windows validation host has no NVIDIA or Apple accelerator. The seven
  hardware-only tests therefore skip locally by design; a recent green scheduled
  artifact remains a release-promotion requirement and local CPU results are not
  represented as CUDA/MPS evidence.
- `StreamingSupervisedData`, `numpy_shard_stream`, and `train_streaming` provide
  bounded, restartable, memory-mapped regression training with explicit limitations
  for checkpoints, schedulers, classifiers, custom modules, and distributed state.
- Deterministic model/data fingerprints, credential rejection, redacted operational
  events, retention metadata, and vendor-neutral tracker/registry/monitor hooks are
  covered by the Phase 7 unit suite and documented in
  [`docs/workplace_operations.md`](../workplace_operations.md).
- The final CPU observation passed correctness and all performance tolerances:
  105.83 training samples/second, 0.526 ms inference p50, 0.816 ms p95,
  74,953,431 bytes peak traced Python memory, 4.64 ms artifact-load p50, and
  7,083.86 ms explanation time.
- The locked 27-package deployment surface reports zero known vulnerabilities. A
  clean repository secret scan and the rebuilt non-root `0.16.0` image's fixed
  high/critical Trivy gate pass. SPDX JSON SBOMs were generated for both core and
  language-model wheels/source distributions and for the final image. Third-party
  scanner, SBOM, and container actions are immutable-commit pinned.
- Fresh `0.16.0` wheels and source distributions pass Twine validation and isolated
  installed-wheel smoke tests. The rebuilt service image passes version, readiness,
  health, metadata, and prediction checks.
- The complete fast CPU suite passes 509 tests with one optional skip and 45
  slow/GPU deselections. Phase 7's focused suite passes 26 tests with seven
  hardware-only skips. Core branch coverage is 74%, above the blocking 70% floor.
  Ruff, Black, mypy, pre-commit, strict repository hygiene, workflow YAML parsing,
  and checked-in performance-baseline comparison all pass.

## Phase 8 - Certify the Workplace Platform

Goal: prove the complete workflow and publish a stable support commitment.

Implementation status (2026-07-28): complete for the source, built-wheel CPU,
artifact, container, documentation, compatibility, and security gates. Release
promotion remains intentionally open until the self-hosted CUDA soak passes for the
exact candidate commit and a maintainer reviews the linked evidence before tagging.
The subsequent pre-PyPI review found release-identity, clean-environment
certification, compatibility-evidence, support-matrix, and API-freeze gaps. Those
findings reopen the affected readiness claims below and are tracked systematically in
Phase 9.

### End-to-end certification scenarios

- [x] **P1** Tabular regression: named pandas features, target scaling, early stopping,
  resume, safe artifact, batch inference, and SHAP explanation.
- [x] **P1** Binary classification: probabilities, threshold policy, classification
  metrics, safe artifact, service inference, and SHAP explanation.
- [x] **P1** Multiclass classification: class labels, probability matrix, top-k output,
  artifact parity, and per-class explanations.
- [x] **P1** Convolutional workload: preserved shape, deployment parity, bounded-memory
  inference, and grouped spatial explanations.
- [x] **P1** Sequence/context workload: explicit context contract, state isolation,
  deployment parity, and supported explanation behavior.
- [x] **P1** Custom registered backbone: training and native artifact round trip with a
  documented reduced export/support guarantee.

### Release readiness

- [x] **P0** Run the full quality, package, artifact, deployment, security, and
  documentation suite from a clean checkout.
- [x] **P0** Test upgrade and artifact migration from every supported prior release
  using authentic producer artifacts or narrow the support claim to retained evidence.
- [x] **P0** Align the final support and compatibility matrix with blocking CI evidence.
- [x] **P0** Publish migration guidance for legacy checkpoints and estimator-only
  workflows.
- [x] **P0** Publish task-oriented quick starts for regression, classification,
  deployment, resume, and SHAP.
- [ ] **P1** Freeze the complete documented stable API for the release candidate.
- [ ] **P1** Run a release-candidate soak using representative CPU and CUDA workloads.
- [x] **P1** Triage every warning emitted by the certification suite as fixed,
  documented, or explicitly accepted.
- [ ] **P1** Tag the workplace-ready release only after all blocking exit criteria are
  linked to evidence.

### Final exit criteria

- [x] All six certification scenarios pass from built wheels and versioned artifacts.
- [ ] The release has no failing format, lint, typing, test, coverage, hygiene,
  security, or artifact-compatibility gate.
- [x] Users can follow one documented API from model specification through training,
  deployment, and explanation.
- [x] Stable versus experimental capabilities are explicit and enforced by tests.

### Phase 8 evidence

- The shipped `psann.platform.certification` runner passes all six scenarios from the
  installed pre-identity placeholder `1.0.0rc1` wheel for five CPU soak iterations
  with warnings treated as errors. Its privacy-safe JSON report is under
  `reports/certification/built-wheel-cpu-final/`.
- The Phase 8 snapshot passed Ruff, Black, mypy, its then-current 24-name public API
  freeze, strict repository hygiene, and the complete non-slow/non-GPU suite:
  522 passed, 1 skipped, and 45 deselected. Phase 9.4 supersedes that partial freeze
  with exhaustive current and public `0.12.7` inventories.
- Core coverage is 75%, above the blocking 70% floor. The pinned deployment dependency
  audit, secret scan, fixed HIGH/CRITICAL image scan, non-root container smoke, Twine
  checks, package smoke, and SPDX SBOM generation pass.
- The former manual `Workplace release-candidate certification` workflow repeated
  the clean source gates, certified the built wheel, ran the six-scenario CUDA soak
  on a self-hosted GPU runner, reused the supply-chain and container gates, and
  exposed a final promotion gate. It was [archived on
  2026-08-11](../archive/workflows/README.md) and is historical design context, not
  current evidence.
- The current workstation has a CUDA-enabled Torch build but no available CUDA device.
  Therefore the CUDA checkbox, aggregate no-failing-gates checkbox, and release tag
  remain open; they require workflow evidence from the exact pushed candidate commit.

## Phase 9 - Resolve Release-Review Findings

Goal: close every actionable finding from the 2026-07-28 pre-PyPI review, align
documentation with executable evidence, and promote one uniquely identifiable clean
candidate through the complete release workflow.

Status (2026-08-11): in progress; 36 of 44 tracked items are complete. Publishing,
tagging, and stable support promotion remain blocked until every P0 and P1 item below
is complete and linked to evidence from the same pushed commit. The former release
certification, supply-chain security, and HISSO benchmark workflows are archived, so
an active promotion path must also be restored or replaced.

### Phase 9.1 - Release identity (complete)

- [x] **P0** Choose a package version and Git/GitHub tag strategy that does not reuse
  the existing historical `v1.0.0` tag. Verify the selected versions remain available
  on PyPI and make the container, SBOM, release-note, and artifact naming conventions
  use the same identity.
- [x] **P0** Update `src/psann/_version.py`, the generated `psannlm/_version.py`
  mirror, `CHANGELOG.md`, release documentation, migration guidance, and package
  metadata for the selected release identity.
- [x] **P0** Replace the stale `0.16.x` security-support statement with the actual
  public support window and confirm that PyPI, GitHub, documentation, and
  `SECURITY.md` describe the same supported releases.

Selected identity: package version `1.1.0rc1`, candidate tag `v1.1.0rc1`, GA package
version `1.1.0`, and GA tag `v1.1.0`. The historical `v1.0.0` tag is permanently
reserved. PyPI and remote-tag availability were verified on 2026-08-10. Naming and
promotion rules are recorded in [`release_identity.md`](../release_identity.md).

### Phase 9.2 - Release and packaging automation (complete)

- [x] **P1** Make the installed-wheel CPU certification environment install an
  explicit FastAPI/Starlette test-client dependency that works under the supported
  dependency band.
- [x] **P1** Make the CUDA certification environment use the same compatible service
  test-client contract and pass with `PYTHONWARNINGS=error`; do not rely on a
  deprecated `httpx` fallback.
- [x] **P1** Add a clean-environment regression test that installs exactly the
  release-workflow dependency command and imports/constructs `TestClient` before
  running all six certification scenarios.
- [x] **P1** Teach `scripts/release.py` to handle PEP 440 prerelease versions for
  documented bump and dry-run flows, including `1.1.0rc1`.
- [x] **P1** Remove command-line PyPI token examples and prefer environment, keyring,
  or trusted-publishing credentials so tokens do not enter command history or process
  listings.
- [x] **P1** Add release-helper preflights for a clean tree, synchronized package
  versions, changelog entry, unique tag/version, Twine validation, package smoke, and
  explicit upload confirmation.
- [x] **P1** Give `psannlm` a compatibility constraint that prevents a new LM wheel
  from silently using an unsupported old core package.
- [x] **P1** Make `psannlm.__version__` report its own installed distribution version
  and add mismatched-core/LM installation tests, while retaining a clear compatibility
  error when the core version is outside the supported band.

Phase 9.2 establishes `psann>=1.1.0rc1,<1.2` as the 1.1 PSANN-LM compatibility
band. The LM package reports its own bundled version and validates the installed core
before importing LM implementation modules. Release preparation now fails closed on
dirty state, version/changelog/identity/compatibility drift, malformed artifacts,
Twine or installed-wheel smoke failures, and missing exact upload confirmation.

### Training, artifact, and configuration correctness

- [x] **P1** Make binary training and validation accuracy use the configured
  `TaskSpec.threshold`; prove that history, `TrainingRun.evaluate`, classifier score,
  artifact inference, and service inference agree for non-default thresholds.
- [x] **P1** Make multilabel training and validation accuracy use scalar or per-label
  configured thresholds and cover heterogeneous threshold tuples.
- [x] **P2** Enforce finite standard JSON values in model, data, training, inference,
  and explainer specifications. Reject `NaN`, positive infinity, and negative infinity
  before a specification can be saved or exported.
- [x] **P2** Replace the three-integer artifact/plugin version parser with strict PEP
  440 comparison. Cover prerelease, postrelease, development, local, malformed, and
  vendor-suffixed Torch version cases.
- [x] **P2** Apply bounded member, metadata, total-size, duplicate-name, and fixed
  layout validation to `.psann-train` files before reading checkpoint members into
  memory.
- [x] **P2** Add compressed checkpoint-bomb, oversized metadata/state, duplicate
  member, unexpected member, malformed checksum, and valid large-checkpoint tests.
- [x] **P2** Either support registered custom `torch.nn.Module` factories for every
  task they may declare or reject unsupported classification declarations during
  registration/spec validation before training. Never allow the current late
  estimator-attribute failure.

### Phase 9.3 - Authentic compatibility evidence (complete)

- [x] **P1** Retain authentic native artifacts produced by every supported preview
  line where immutable producers exist, or narrow the support claim. The repository
  and public-package audit found no released `0.13`-`0.16` producers, so native
  producer support now begins with 1.1 and those labels are documented as unpublished
  development phases.
- [x] **P1** Retain an actual public `0.12.7` legacy checkpoint fixture and automate
  trusted load/migration parity so the public upgrade path is continuously exercised.
- [x] **P1** Stop treating a current artifact with rewritten version strings as
  sufficient historical compatibility evidence. Keep such tests only for manifest
  validation and label them accordingly.

The retained checkpoint was generated through the exact hash-pinned public `0.12.7`
wheel, includes producer inputs/predictions and a fixture digest, and passes current
trusted load plus native-migration parity. Rewritten package and format fields remain
covered, but test names and documentation identify them as synthetic manifest/schema
validation. The full boundary and reproduction procedure are recorded in
[`compatibility_evidence.md`](../compatibility_evidence.md).

### Phase 9.4 - API, CI, and support evidence

- [x] **P1** Define the complete stable top-level and `psann.platform` API inventory.
  Make the freeze gate fail when a documented stable name is missing from the
  manifest, removed from exports, or changes its constructor/function parameters or
  required methods outside policy.
- [x] **P1** Add the prior public `0.12.7` API inventory and principal estimator
  signatures to compatibility checks so the new workplace freeze does not overlook
  the existing stable estimator surface.
- [x] **P1** Run blocking installed-wheel tests for Python 3.11, 3.12, and 3.13 with
  the optional workplace dependencies required by the support claim.
- [x] **P1** Add the documented SHAP band on Python 3.13 to blocking explainability
  CI, or narrow the support matrix until that evidence exists.
- [x] **P1** Add blocking Windows x86_64 CPU correctness, package, artifact, inference,
  and service jobs, or remove Windows Tier 1 status.
- [x] **P1** Add explicit floor/current jobs for NumPy, PyTorch, scikit-learn, and
  SHAP. Ensure the PyTorch floor job actually installs 2.4 and that maintained
  constraint files are consumed by CI.
- [x] **P1** Make every row in `docs/workplace_support_matrix.md` link to the workflow
  or retained artifact that proves it; unproven combinations remain experimental or
  unsupported.
- [x] **P2** Decide and document whether registered arbitrary Torch modules are a
  regression-only experimental feature or a broader task contract, then make the
  registry, runtime errors, tests, and documentation agree.
- [x] **P2** Establish an explicit PSANN-LM quality target. Raise its current
  non-blocking coverage from the observed 40% around user-facing CLI, SFT, data, and
  trainer paths, or keep the distribution clearly Alpha/experimental with a recorded
  acceptance rationale.
- [x] **P2** Decide whether scripts coverage remains observational. If so, document
  why the 1% report is non-blocking and add focused tests for release-critical scripts
  such as `scripts/release.py`.

Phase 9.4 freezes 147 stable top-level exports, all 115 `psann.platform` exports,
their inspectable signatures, and required lifecycle methods; it separately protects
53 authentic public `0.12.7` exports and six principal estimator signatures. Blocking
CI now exercises installed wheels on Linux and Windows across Python 3.11-3.13,
includes SHAP on 3.13, consumes maintained core/SHAP floor and current profiles, and
verifies the PyTorch 2.4 floor explicitly. Every support row has a linked authority;
unproven Windows CUDA and macOS package claims were narrowed. PSANN-LM remains Alpha
with a blocking 35% floor and a 50% Beta target, aggregate scripts remain
observational, and `scripts/release.py` has a dedicated blocking 60% floor.

### Repository hygiene and review gates

- [x] **P2** Add `git diff --check` or an equivalent whitespace-error check to local
  quality and CI gates so repository hygiene covers patch-level whitespace errors.
- [x] **P3** Remove the extra blank line at EOF from the five workplace ADRs and
  `docs/adr/README.md`.
- [x] **P2** Repeat local-link validation and workflow-YAML parsing after the release
  workflow and documentation changes, retaining the commands or tests as normal
  repository gates.

### Promotion provenance (deferred until implementation closes)

- [ ] **P0** Commit all intended phase work on a release branch, verify there are no
  unintended tracked or untracked files, push the exact candidate, and record its
  full commit SHA. Temporary local snapshot commits are not promotion evidence.
- [ ] **P0** Create `v1.1.0rc1` only after every blocking workflow is green for that
  exact pushed commit. Archived workflows do not qualify. Never move or replace the
  historical `v1.0.0` tag.

### Phase 9 exit criteria

- [ ] A unique package version, Git tag, GitHub release, container tag, changelog
  section, and security-support statement identify the same release.
- [ ] The exact pushed candidate commit has a clean tree and all P0/P1 changes,
  regression tests, and documentation committed.
- [ ] Source and installed-wheel quality, API, training, artifact, inference, SHAP,
  service, coverage, hygiene, build, Twine, package-smoke, and dependency checks pass
  from clean environments.
- [x] Authentic compatibility fixtures prove every retained native-artifact and
  legacy-checkpoint support claim.
- [x] The support matrix contains no platform, Python, dependency, accelerator, task,
  export, or explanation claim without a matching automated evidence path.
- [ ] The exact candidate's CPU, Windows, CUDA, security, container, SBOM, and
  promotion jobs are green and their retained evidence has been reviewed.
- [ ] No release credential appears in source, command history guidance, workflow
  logs, artifacts, model metadata, or promoted reports.
- [ ] A maintainer records the evidence links and explicitly approves tagging and PyPI
  publishing; automation does not infer promotion from a partial gate set.

### Phase 9 evidence record

Complete this section as work closes; do not replace evidence with unchecked prose.

- Exact candidate commit: pending
- Workflow archive disposition: the release-certification, supply-chain security,
  and HISSO benchmark definitions were archived on 2026-08-11 after the failing runs
  linked in the [archive record](../archive/workflows/README.md). They are not active
  promotion evidence, so replacement exact-commit evidence remains pending.
- Selected package and tag identity: `1.1.0rc1` / `v1.1.0rc1`; GA is reserved as
  `1.1.0` / `v1.1.0`. Public package and remote-tag availability verified on
  2026-08-10; release-preparation branch is `codex/release-1.1.0rc1`.
- Phase 9 progress: 36 of 44 items complete; 8 remain.
- Clean source/coverage/hygiene results: canonical Ruff, Black, mypy, staged/unstaged
  whitespace, local-link, workflow-YAML, public-API-freeze, and strict hygiene gates
  passed locally on 2026-08-10.
- Release-identity regression results: 30 focused version, release-helper,
  repository-contract, and reference-service tests passed locally on 2026-08-10.
- Package identity evidence: isolated core and PSANN-LM source/wheel builds produced
  only `1.1.0rc1` artifacts, and all four distributions passed Twine validation on
  2026-08-10.
- Release-automation evidence: 32 focused release/version/LM-compatibility tests
  passed; isolated built-wheel smoke verified distribution-owned versions, the LM
  wheel's declared core range, compatible import, and mismatched-core rejection on
  2026-08-10.
- Regression results: the complete non-slow/non-GPU suite passed locally on
  2026-08-10 with 595 passed, one optional skip, and 45 deselections.
- Scoped coverage results: core 75% (70% floor), PSANN-LM 40% (35% Alpha floor),
  release helper 62% (60% floor), and aggregate scripts 3% observational all passed
  on 2026-08-10. Four scoped XML reports were generated by the canonical gate.
- Test-client compatibility: an isolated FastAPI 0.140.13 / Starlette 1.3.1 /
  httpx2 2.9.1 environment constructed `TestClient` with warnings promoted to errors.
- Built-wheel and six-scenario CPU certification: pending
- Authentic compatibility fixture matrix: the public `0.12.7` wheel
  (`43e6bc16a06a27b72e9073d1f80dbac70e07634df4dd01459ab949032997699b`)
  produced the retained legacy checkpoint
  (`99d49317fce455b3b1c419b3ea02e8846c3a5a8c345c62daefd244c2832e897b`).
  Six focused authentic-migration and synthetic-schema checks passed both from the
  source tree and from an isolated installed `1.1.0rc1` wheel on 2026-08-10. The
  `0.13`-`0.16` labels have no immutable published producers, so the unsupported
  historical claim was withdrawn rather than simulated.
- API compatibility evidence: the current freeze covers 262 scoped stable exports
  across `psann` and `psann.platform`; the authentic public `0.12.7` inventory covers
  53 exports and six principal estimator signatures. Source and isolated installed-
  wheel checks passed locally on 2026-08-10.
- Python/dependency/Windows CI matrix: blocking workflow definitions now cover
  installed wheels on Linux and Windows for Python 3.11-3.13, SHAP on all three
  minors, the PyTorch/NumPy/sklearn floor, a maintained snapshot, admitted current
  dependencies, and SHAP floor/current. Exact pushed-commit workflow runs remain
  pending promotion evidence.
- CUDA certification artifact: pending
- Dependency, secret, container, and SBOM evidence: pending
- Maintainer promotion approval: pending
- Post-publication clean-install verification for `psann` and `psannlm`: pending

## Cross-Phase Test Matrix

Every stable task/backbone combination should be evaluated against the applicable rows:

| Contract | Required evidence |
| --- | --- |
| Construction | Valid configuration builds; invalid configuration fails early |
| Training | Loss decreases on a deterministic fixture; history is structured |
| Validation | Metrics, early stopping, and best-state restoration are correct |
| Resume | Interrupted and resumed run remains within documented tolerance |
| Prediction | Single-row and batched shapes, names, and dtypes are correct |
| sklearn | Clone, parameters, fitted-state, pipeline, and score behavior |
| Artifact | Safe save/load, checksums, migrations, device mapping, prediction parity |
| Deployment | Concurrent stateless inference and bounded-memory batching |
| Export | Numerical and dynamic-shape parity for every promised format |
| Explainability | Attribution shape, names, additivity, determinism, and state isolation |
| Accelerator | Supported device/dtype forward, backward, artifact, and inference paths |
| Security | Untrusted artifact rejection, redaction, dependency scan, and SBOM |

## Critical Path

The shortest path to a useful workplace vertical slice is:

1. Complete Phases 0 and 1.
2. Fix Phase 2 P0 correctness items.
3. Implement regression plus binary classification through the Phase 3 high-level API.
4. Implement the Phase 4 safe native artifact and generic loader.
5. Implement Phase 5 stateless batched inference.
6. Implement Phase 6 model-agnostic tabular SHAP.
7. Certify the tabular regression and binary-classification scenarios.
8. Expand the same contracts to remaining backbones, tasks, accelerators, and export
   formats.

This ordering produces an end-to-end platform slice early while keeping broader model
and infrastructure work behind contracts that have already been proven.

## Explicit Non-Goals for the First Workplace Milestone

- Moving language-model training back into the core `psann` distribution.
- Promoting GeoSparse to stable before it passes the common task, artifact, inference,
  and explanation contracts.
- Building a hosted model-registry control plane.
- Supporting every Torch module through every export format.
- Treating SHAP values as causal effects or silently selecting background data.
- Adding distributed training before single-device resume and deployment are reliable.
