# Migration Notes

Guidance for upgrading projects to the refactored training surface introduced after 0.9.19. Follow this checklist when updating downstream code so documentation and implementation stay aligned.

The `0.13` through `0.16` numbers previously attached to workplace phases below were
internal development labels, not published distributions or native-artifact producer
lines. They remain parenthetical historical labels only so older planning notes can
be interpreted correctly.

## What changed

- **Workplace 1.1 release candidate (`1.1.0rc1`)**
  - PSANN-LM now declares `psann>=1.1.0rc1,<1.2`, rejects mismatched cores during
    import, and reports its own LM distribution version instead of aliasing the core
    version. Upgrade the two packages together when moving onto the 1.1 line.
  - New projects should use the frozen `ModelSpec` → `create_model` → `train` →
    `TrainingRun.export` → `load_runtime` → `explain` path. The machine-readable
    name/signature contract is [`workplace_public_api.json`](workplace_public_api.json).
  - `InferenceConfig(top_k=k)` adds ranked labels, probabilities, and indices to
    `InferenceResult.top_k` for multiclass probability output. The complete
    probability matrix remains in `InferenceResult.values`.
  - A registered custom factory returning `torch.nn.Module` can now use the safe
    native artifact path. It remains experimental, must be registered before load,
    and has no general derived-export or gradient-explanation guarantee.
  - Six executable certification scenarios ship in the wheel and are invoked with
    `python -m psann.platform.certification`. See the final support matrix for the
    distinction between stable, capability-gated, experimental, and out-of-scope
    behavior.

- **Workplace Phase 7 operations hardening (internal label 0.16; unpublished)**
  - High-level training and inference now enforce the documented CPU/CUDA/MPS,
    fp32/fp16/bf16, AMP, compile, and fallback policy. Use
    `fallback_policy="error"` for certification and deployment.
  - `train_streaming` and `numpy_shard_stream` add bounded, restartable regression
    training without loading complete NumPy shards into memory. Optimizer state is
    batch-local, and streaming classification/resume/schedulers remain explicit
    unsupported cases.
  - Training/export now attach SHA-256 model/data fingerprints without raw rows.
    Artifact metadata, registry extensions, and model cards reject credential-like
    content.
  - `OperationalHooks` is the dependency-free boundary for workplace tracking,
    registry publication, and monitoring. `RetentionPolicy` is metadata; the host
    system must enforce deletion.
  - Performance, scan, and SBOM evidence are documented in
    [`workplace_operations.md`](workplace_operations.md).

- **Workplace Phase 6 SHAP explainability (internal label 0.15; unpublished)**
  - Install `psann[explain]`; SHAP remains absent from the base dependency and is
    imported only when an explainer is created.
  - Explain a fitted model or loaded `InferenceRuntime` with an explicit `background`,
    `reference_data`, or `BackgroundSummary`. Training data is never inferred.
  - Model-agnostic algorithms call the deployed raw-input path. Requested gradient or
    deep algorithms are capability-gated and record an explicit fallback reason unless
    `fallback="error"` is selected.
  - `ExplanationResult.explanation` is a standard `shap.Explanation`; the wrapper adds
    artifact/run identity, feature groups, limits, output semantics, and state policy.
  - SHAP 0.50+ uses NumPy 2. Keep explanation environments separate from the NumPy
    1.26 `compat` snapshot. See [`explainability.md`](explainability.md).

- **Workplace Phase 5 deployment runtime (internal label 0.14; unpublished)**
  - Prefer `load_runtime(path, config=InferenceConfig(...))` for deployed batch jobs
    and services. `load_model` remains the lower-level fitted-estimator loader.
  - Classification runtime values are probabilities by default. Select
    `classification_output="label"` or `return_logits=True` explicitly when those
    contracts are required.
  - `device_transfer="per_batch"` is bounded-memory; `full_batch` is an explicit
    opt-in. Stable deployment dtype is currently float32.
  - Stateful inference now has an isolated `InferenceSession` lifecycle. Ordinary
    runtime predictions do not advance shared state.
  - Derived `.pt2` and `.onnx` files are capability-gated and require their generated
    `.preprocessing.json` sidecar. Continue retaining the native `.psann` artifact as
    the source of truth.
  - The reference service is optional via `psann[serve]`; ONNX tooling is optional via
    `psann[export]`. See [`deployment.md`](deployment.md).

- **Workplace Phase 4 safe artifacts (internal label 0.13; unpublished)**
  - `TrainingRun.export("model.psann")` now writes an atomic, checksummed native
    bundle with JSON configuration/schema/preprocessing state and restricted-load
    tensor weights.
  - Use `inspect_artifact` for metadata-only validation and `load_model` for generic
    registered-backbone reconstruction. `.psann-train` resume checkpoints are never
    accepted by the deployment loader.
  - Artifact format versions migrate independently of package versions. Use
    `migrate_artifact(source, destination)` to rewrite a supported historical bundle.
  - Class-specific estimator `.save()` / `.load()` now emit
    `LegacyCheckpointWarning` because they use unrestricted pickle.
  - Generic loading refuses legacy `.pt` files by default. Conversion requires
    `migrate_legacy_checkpoint(..., trusted_legacy_checkpoint=True)` after provenance
    has been verified. The flag acknowledges risk; it does not sanitize the source
    before deserialization.
  - Direct arbitrary modules, callable context builders, opaque LSM objects, and
    custom scaler objects remain in-process only. A module created by a registered
    reconstructable backbone factory can use the native artifact path in 1.1, but
    remains experimental and plugin-dependent. See [`artifacts.md`](artifacts.md).

- **Workplace Phase 3 unified API (unreleased)**
  - New projects can use `TaskSpec`, `ModelSpec`, `DataSchema`, `create_model`, and
    `train`; existing estimator construction and `fit(...)->self` remain supported.
  - Use `PSANNClassifier` instead of treating classification labels as regression
    targets. Binary, multiclass, and multilabel losses/probabilities are task-owned.
  - Standard GELU and SiLU activations are now available in the dense, residual, and
    convolutional PSANN cores.
  - Pandas-like feature names are captured at fit. Prediction is strict by default;
    select `DataSchema(feature_policy="reorder")` or call
    `set_feature_schema_policy("reorder")` for safe name-based reordering.
  - `ModelSpec.parameters` accepts canonical names only. Move activation,
    normalization, dropout, and data format into their dedicated specification
    fields.
  - Phase 4 now supplies `TrainingRun.export` and generic artifact loading; estimator
    snapshots remain trusted legacy persistence.

- **Workplace Phase 2 training core (unreleased)**
  - Unknown optimizer, loss, activation, scheduler, dimension, LR, batch, patience,
    device, and incompatible option values now fail before model construction.
  - `loss_reduction="none"` is no longer accepted for optimizer-driven training.
    Choose `mean` or `sum`.
  - NaN and infinity in training/validation arrays are rejected at the data boundary.
  - Callback and gradient-hook exceptions now raise by default. Use
    `callback_error_policy="warn"` only when intentionally preserving legacy behavior.
  - Compile, AMP, and unavailable-accelerator downgrades follow
    `fallback_policy="warn" | "error"` and emit structured events.
  - Named `step` and `cosine` schedulers, detached user metrics, standard logging, and
    stable training events are available through `fit`.
  - Use `checkpoint_dir` and `resume_from` for `.psann-train` resume checkpoints.
    Estimator `save`/`load` remains a separate trusted legacy inference snapshot.
  - `deterministic=True` enables deterministic algorithms and persists sampler/RNG
    state; see `training_core.md` for platform limitations.

- **Workplace Phase 1 baseline (unreleased)**
  - Python 3.11 is now the package minimum; Python 3.11, 3.12, and 3.13 are the tested
    development matrix. Pin the last compatible release for Python 3.9/3.10.
  - NumPy 1.26 and PyTorch 2.4 are the new runtime floors. The scikit-learn extra now
    targets scikit-learn 1.4 through the latest pre-2.0 release.
  - `src/psann/_version.py` is the only package-version source shared by `psann` and
    `psannlm`.
  - Black is the canonical formatter; Ruff owns lint and import ordering.
  - The empty `notebooks/Untitled.ipynb` file was removed, and committed notebook
    outputs/execution counts are now rejected.

- **0.10.12 diagnostics tweaks (2025-10-31)**  
  - `psann.utils.encode_and_probe` now returns explicit `probe_accuracy`, `baseline_accuracy`, and a `baseline_metrics` summary while keeping `accuracy` as the higher of the two for backwards compatibility.  
  - Inspect `accuracy_source` to determine whether the linear probe or the raw baseline features produced the winning score when tracking regressions across releases.

- **0.10.5 housekeeping (2025-10-19)**  
  - Development tooling extras now install coverage and build so CI can publish coverage reports and validate wheels.  
  - HISSO integration tests are marked `slow` to unblock quick `pytest -m "not slow"` iterations while the refactor settles.  
  - GitHub Actions runs Ruff/Black across the full tree, captures coverage on Python 3.11, and uploads built artifacts.  
  - Benchmark data provenance moved into `benchmarks/README.md` with a helper downloader; legacy Colab instructions live under `docs/archive/`.

- **Primary-output pipeline** - predictive extras and growth schedules were removed. Constructors ignore legacy `extras_*` arguments and emit warnings so downstream projects can detect stale configuration.
- **Shared fit helpers** - all estimators route through `normalise_fit_args`, `prepare_inputs_and_scaler`, `build_model_from_hooks`, and `run_supervised_training`. Custom loops should import these helpers instead of copying logic from `PSANNRegressor.fit`.
- **HISSO options** - episodic runs resolve reward, transform, noise, and warm-start settings via `HISSOOptions`. The public API still uses familiar keyword arguments (`hisso_window`, `hisso_reward_fn`, etc.), but the resolved options are now stored for evaluation helpers.
- **Neutral terminology** - episodic configs standardise on `transition_penalty`. The legacy aliases (`transition_cost`, `trans_cost`) remain temporarily and trigger deprecation warnings.

## Helper replacement table

| Previous touchpoint                             | Updated helper / destination                         | Notes |
|-------------------------------------------------|------------------------------------------------------|-------|
| Manual dtype/validation handling inside `fit`   | `normalise_fit_args`                                 | Converts validation triplets to float32. |
| Ad-hoc scaler prep + flattening                 | `prepare_inputs_and_scaler`                          | Returns `PreparedInputState` with train tensors + metadata. |
| Variant-specific model construction             | `build_model_from_hooks` + `FitVariantHooks`         | Supply hooks instead of overriding the full `fit`. |
| Episodic adapters per estimator                 | `build_hisso_training_plan` via hooks                | Ensures conv/dense variants share the same HISSO flow. |
| Direct reward-function wiring                   | `register_reward_strategy` / `get_reward_strategy`   | Bundles reward functions and secondary metrics. |

## Primary-output workflow

- Remove extra heads: drop `extras`, `extras_growth`, and `extras_*` kwargs from estimator construction and calls to `fit`. Any remaining references will raise warnings so you can locate stale code.
- HISSO warm starts use `hisso_supervised={"y": targets}`. Provide `hisso_window`, `hisso_reward_fn`, and optional `hisso_context_extractor` as before; the helpers consolidate them into `HISSOOptions` and persist the configuration for evaluation utilities.
- If you previously inspected `_extras_cache_` or other extras-specific attributes, switch to the HISSO helpers (`hisso_infer_series`, `hisso_evaluate_reward`) or the shared prepared-input metadata.

## Upgrade checklist

1. Replace manual preprocessing with `normalise_fit_args` and `prepare_inputs_and_scaler` so shapes and scalers match estimator behaviour.
2. Remove any `extras_*` constructor arguments or `extras_targets` usage. Confirm your tests no longer expect extras outputs.
3. Swap bespoke reward wiring for registry lookups (`register_reward_strategy`, `get_reward_strategy`) and keep configs on the neutral naming (`transition_penalty`).
4. Update notebooks and scripts to mention the curated example set (`examples/21`, `26`, `27`, etc.) instead of the retired predictive extras demos.
5. Log progress in `CLEANUP_TODO.md` whenever you touch docs or code that affects the migration effort.

## Move an estimator-only workflow to the workplace API

Existing estimator construction remains supported, but the workplace path makes the
task, schema, training, artifact, and inference contracts reviewable:

```python
# Previous estimator-only flow
model = psann.PSANNRegressor(hidden_layers=2, hidden_units=32, epochs=25)
model.fit(train_x, train_y)
predictions = model.predict(batch)

# Workplace flow
spec = psann.ModelSpec(
    input_schema=psann.DataSchema(input_shape=(train_x.shape[1],)),
    parameters={"hidden_layers": 2, "hidden_units": 32},
)
run = psann.train(
    psann.create_model(spec),
    (train_x, train_y),
    config=psann.TrainingConfig(epochs=25),
)
artifact = run.export("artifacts/model.psann")
predictions = psann.load_runtime(artifact).predict(batch).values
```

For classification, select `TaskSpec(kind="binary" | "multiclass" | "multilabel")`
instead of fitting numeric labels through a regressor.

## Move a legacy checkpoint

Whole-object estimator `.save()` / `.load()` files use Python pickle and may execute
code. Do not pass an untrusted file to the migration tool.

```python
migrated = psann.migrate_legacy_checkpoint(
    "verified-estimator.pt",
    "artifacts/model.psann",
    trusted_legacy_checkpoint=True,
)
runtime = psann.load_runtime(migrated)
```

The trust flag records an operator decision; it does not sanitize the legacy payload
before deserialization. Keep the original file quarantined and record provenance.
The repository retains a public `0.12.7` checkpoint with a pinned producer-wheel
hash and expected predictions. The 1.1 candidate continuously verifies explicit
trusted loading, conversion to native format `1.0`, and numerical parity for that
fixture. Native format `1.0` starts its supported producer history with the 1.1 line.
Manifest format `0.9` is migrated in memory and can be rewritten with
`migrate_artifact`, but that coverage is deliberately labeled synthetic schema
validation rather than historical producer evidence. See
[`compatibility_evidence.md`](compatibility_evidence.md).

## Outstanding TODOs

- Stage GPU benchmark baselines once shared hardware becomes available so CI can compare CPU and CUDA runs side by side.
- Expand regression coverage around HISSO evaluation helpers to exercise both supervised warm starts and reward-only episodes.
- Publish CI guidance tied to the contributor workflow (ruff and pytest gates) after the documentation refresh completes.

