# Public API Surface

This page is the human-readable guide to the **supported** public API surface. The
exhaustive, machine-readable inventories are
[`workplace_public_api.json`](workplace_public_api.json) for the 1.1 candidate and
[`public_api_0_12_7.json`](public_api_0_12_7.json) for the retained public legacy
surface. Anything absent from those inventories should be treated as internal.

## Top-level imports (stable)

These are the main categories exported from `psann.__init__` and safe to import
directly. The freeze classifies all 148 top-level exports: 147 stable names and the
experimental `GeoSparseRegressor`.

### Estimators
- `PSANNRegressor`
- `PSANNClassifier`
- `ResPSANNRegressor`
- `ResConvPSANNRegressor`
- `SGRPSANNRegressor`
- `WaveResNetRegressor`

### Episodic training (HISSO)
- `HISSOOptions`
- `EpisodeTrainer`, `EpisodeConfig`
- `hisso_infer_series`, `hisso_evaluate_reward`
- `get_reward_strategy`, `register_reward_strategy`
- `RewardStrategyBundle`, `FINANCE_PORTFOLIO_STRATEGY`
- `multiplicative_return_reward`, `portfolio_log_return_reward`
- `make_episode_trainer_from_estimator`

### Expanders and activation config
- `LSM`, `LSMExpander`
- `LSMConv2d`, `LSMConv2dExpander`
- `SineParam`, `SigmoidParam`, `ActivationConfig`
- `StateConfig`, `StateController`, `ensure_state_config`

### Token utilities
- `SimpleWordTokenizer`
- `SineTokenEmbedder`

### Core wave backbones
- `WaveResNet`, `WaveEncoder`, `WaveRNNCell`
- `build_wave_resnet`
- `scan_regimes`

### Diagnostics and synthetic data
- `jacobian_spectrum`, `ntk_eigens`, `participation_ratio`, `mutual_info_proxy`
- `encode_and_probe`, `fit_linear_probe`
- `make_context_rotating_moons`, `make_drift_series`, `make_shock_series`, `make_regime_switch_ts`

### Parameter counting helpers
- `count_params`, `dense_mlp_params`, `geo_sparse_net_params`, `match_dense_width`

### Supervised training contracts

- `TrainingEvent`
- `TrainingCheckpointError`
- `ArtifactInfo`
- `ArtifactError`, `ArtifactChecksumError`, `ArtifactFormatError`,
  `ArtifactVersionError`, `ArtifactExtensionError`
- `LegacyCheckpointTrustError`, `LegacyCheckpointWarning`

The stable estimator `fit` options for validation, events, metrics, deterministic mode,
and `.psann-train` resume checkpoints are documented in
[`training_core.md`](training_core.md).

## Workplace lifecycle API

The following names are stable top-level imports and are also public from
`psann.platform`:

- `TaskSpec`, `ModelSpec`, `TrainingConfig`, `DataSchema`, `InferenceConfig`
- `InferenceResult`, `TopKResult`, `InferenceRuntime`, `InferenceSession`,
  `InferencePool`
- `SupervisedData`, `TrainingRun`
- `create_model`, `train`, `load_model`, `load_runtime`
- `create_inference_runtime`, `load_runtime_pool`
- `inspect_artifact`, `is_model_artifact`
- `migrate_artifact`, `migrate_legacy_checkpoint`
- `evaluate_export_capabilities`, `export_derived`, `preprocessing_contract`
- `DerivedExport`, `ExportCapability`, `ExportReport`
- `DerivedExportError`, `ExportNotSupportedError`
- `TorchModuleAdapter`, `adapt_module`
- `register_backbone`, `register_metric`, `register_schema_transform`
- `register_artifact_resolver`, `resolve_artifact`, `load_registry_runtime`
- `create_app` (requires the optional `serve` dependencies when called)
- `ExplainerConfig`, `BackgroundSummary`, `FeatureGroup`, `ExplanationResult`
- `ExplanationDriftSummary`, `PSANNExplainer`
- `make_explainer`, `explain`, `summarize_background`
- `save_explainer_config`, `load_explainer_config`
- `summarize_explanation_drift`, `write_explanation_report`
- `DifferentiableInferenceAdapter`
- `list_explainable_layers`, `register_explainable_layer`
- `ExplainabilityError`, `ExplainabilityUnavailableError`, `BackgroundPolicyError`,
  `ExplanationCapabilityError`
- `AcceleratorCapability`, `accelerator_capability`, `accelerator_support_matrix`,
  `canonical_dtype`, `runtime_accelerator_evidence`
- `StreamingSupervisedData`, `NumpyShard`, `numpy_shard_stream`, `train_streaming`
- `PerformanceBaseline`, `PerformanceRegression`, `PerformanceReport`,
  `compare_performance`
- `OperationalEvent`, `OperationalHooks`, `OperationalMetadataError`,
  `RetentionPolicy`
- `fingerprint_data`, `fingerprint_model`, `redact_sensitive`, `sensitive_paths`,
  `validate_no_secrets`

The lower-level typed contracts remain public from `psann.platform`:
`ModelSpecContract`, `ArtifactManifest`, `BackboneProtocol`, `TaskAdapter`, and
`TaskKind`. Registry objects for backbones, activations, normalization,
dropout, optimizers, schedulers, losses, and metrics are available from
`psann.platform`.

The 1.1 candidate freezes every exported name in both `psann` and `psann.platform`,
all inspectable stable constructor/function parameter names, explicit signature
exemptions for non-inspectable public types, and required estimator/runtime methods.
`python tools/check_public_api.py` also checks that every public `0.12.7` export
remains available and that the six principal legacy estimator signatures retain
their parameters and methods. Additive legacy-compatible parameters may be added;
the current 1.1 signature snapshot changes only through the deprecation policy.

After an intentionally reviewed API change, a maintainer can prepare a candidate
snapshot with:

```bash
python tools/snapshot_public_api.py --acknowledge-api-change workplace-v1
```

The resulting diff is the review surface; running the command does not itself approve
the change.

`TrainingRun.export` plus the generic loader form the stable native persistence
boundary. The lower-level `export_model` helper, artifact format/schema constants, and
registry objects remain public from `psann.platform`. See
[`artifacts.md`](artifacts.md) and [`workplace_api.md`](workplace_api.md).
The deployment runtime, session, export, resolver, service, and container contracts
are documented in [`deployment.md`](deployment.md).
Optional SHAP installation, explicit background policy, output/group semantics,
gradient capability gates, and interpretation limitations are documented in
[`explainability.md`](explainability.md).
Accelerator/dtype enforcement, bounded streaming, fingerprints, hooks, retention,
performance, and supply-chain evidence are documented in
[`workplace_operations.md`](workplace_operations.md).

## Experimental APIs

These are available but may change without notice:

- `GeoSparseRegressor` (experimental GeoSparse backbone).
- `psannlm` (LM utilities; packaged separately from the core `psann` distribution).

## Internal-only modules (not stable)

The following modules are **internal** implementation details:

- `psann.estimators._fit_utils`
- `psann.layers.*`
- `psann.nn`, `psann.nn_geo_sparse`
- `psann.utils.hf_cache`

If you must rely on internal modules, pin a version and expect breaking changes.
