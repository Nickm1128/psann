# Unified Workplace Model API

Status: Active

The Phase 3 API provides one serializable boundary for choosing a task, backbone,
input schema, training policy, and metrics. It composes the existing sklearn-style
estimators; it does not introduce a second registered-backbone training loop.

## End-to-end workflow

```python
import psann

spec = psann.ModelSpec(
    task=psann.TaskSpec(kind="binary", positive_label="yes"),
    backbone="respsann_mlp",
    input_schema=psann.DataSchema(
        feature_names=("amount", "velocity", "account_age"),
        input_shape=(3,),
        feature_policy="reorder",
    ),
    activation="gelu",
    normalization="layer",
    parameters={
        "hidden_layers": 3,
        "hidden_units": 64,
        "random_state": 7,
    },
)

model = psann.create_model(spec)
run = psann.train(
    model,
    (train_features, train_labels),
    validation_data=(validation_features, validation_labels),
    config=psann.TrainingConfig(
        epochs=50,
        batch_size=128,
        learning_rate=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        metrics=("accuracy",),
        deterministic=True,
        checkpoint_dir="runs/fraud/checkpoints",
    ),
)

probabilities = model.predict_proba(validation_features)
metrics = run.evaluate((validation_features, validation_labels))
artifact = run.export("artifacts/fraud.psann")
deployed = psann.load_runtime(
    artifact,
    config=psann.InferenceConfig(batch_size=256, device="cpu"),
)
result = deployed.predict(validation_features)
```

`create_model` returns an existing regressor for regression or a
`PSANNClassifier` that delegates model construction and fitting to an existing
regressor for classification. Direct estimator use remains supported, and estimator
`fit` still returns `self`. High-level `train` returns a `TrainingRun`.

## Serializable specifications

The following frozen dataclasses expose `to_dict()` and `from_dict()`:

- `TaskSpec`: task kind, class names, positive label, and binary/multilabel threshold;
- `DataSchema`: input/output names, input shape, data format, dtype, named-feature
  policy, and structured preprocessing declarations;
- `ModelSpec`: task, registered backbone, schema, activation, normalization, dropout,
  and canonical architecture parameters;
- `TrainingConfig`: optimizer-driven fit, scheduler, metric, fallback, determinism,
  and resume/checkpoint settings;
- `InferenceConfig`: batch, device, dtype, feature-policy, classifier-output, and
  device-transfer preferences for deployment.

Specifications accept JSON-safe values only. They store registry identifiers rather
than Python callables. `psann.platform.save_spec` and `load_model_spec` provide a
human-readable JSON round trip.

`ModelSpec.parameters` uses canonical estimator names. Deprecated aliases such as
`hidden_width` and fields already owned by the specification are rejected.

## Task contracts

Task adapters own target validation, loss, probabilities, prediction conversion,
default metrics, class metadata, and thresholds.

| Task | Training output/loss | Prediction contract | Evaluation metrics |
| --- | --- | --- | --- |
| Regression | Numeric outputs; configurable regression loss | Numeric vector/matrix | MAE, MSE, R2 |
| Binary | One logit; BCE-with-logits | Two probability columns and original class labels | Accuracy |
| Multiclass | One logit per fitted class; cross-entropy | One probability column per `classes_` entry | Accuracy |
| Multilabel | One logit per label; BCE-with-logits | Per-label probabilities and boolean threshold decisions | Subset accuracy, Hamming loss |

Binary targets must contain exactly two classes. Multiclass targets are one-dimensional
labels. Multilabel targets are a two-dimensional 0/1 indicator matrix. Validation
targets cannot introduce unknown classes. Target scaling is rejected for
classification because it would change task semantics.

`PSANNClassifier` exposes sklearn-compatible `classes_`, `predict`,
`predict_proba`, `decision_function`, and accuracy-oriented `score`. It supports
cloning, pipelines, parameter search, fitted-state errors, `n_features_in_`, and
`feature_names_in_`.

## Registered backbones and shape contracts

| Identifier | Non-batch input shape | Normalization | Dropout | Notable constraints |
| --- | --- | --- | --- | --- |
| `psann_mlp` | `(features,)` | `none` | No | Dense; optional attention/state configuration through canonical estimator options |
| `respsann_mlp` | `(features,)` | `none`, `layer`, `rms` | No | Residual normalization and stochastic-depth options remain distinct |
| `psann_conv1d` | `(channels, length)` | `none` | No | `channels_last` is available through `DataSchema.data_format` |
| `psann_conv2d` | `(channels, height, width)` | `none` | No | Attention requires known spatial dimensions |
| `psann_conv3d` | `(channels, depth, height, width)` | `none` | No | Attention requires known spatial dimensions |
| `respsann_conv2d` | `(channels, height, width)` | `none`, `layer`, `rms` | No | Residual 2D convolution only |
| `wave_resnet` | Rank 1-4 | `none`, `rms`, `weight` | Yes | PSANN activation only; optional FiLM, spectral gate, or convolutional stem |
| `sgr_psann` | Rank 1-3 | `none` | No | PSANN activation only; sequence spectral gating; no convolutional preserve-shape mode |

Registered dense, residual, and convolutional backbones accept regression, binary,
multiclass, and multilabel task adapters. Their common regression and binary shape
matrix is covered by automated tests.

Stable activation identifiers are `relu`, `tanh`, `sigmoid`, `gelu`, `silu`, `psann`,
and `relu_sigmoid_psann`. The factory rejects an activation, normalization, dropout,
task, or rank that its selected backbone does not declare.

Attention, stateful behavior, spectral gating, LSM, segmentation/per-element output,
and context builders retain the compatibility rules documented in
[`API.md`](API.md) and [`training_core.md`](training_core.md). In particular:

- attention and LSM cannot be combined in the current training core;
- stateful updates are supported by the base dense path, not WaveResNet or SGR;
- spectral gating is a WaveResNet/SGR capability and requires a meaningful sequence
  axis;
- per-element output requires a preserve-shape convolutional path and is not a
  classifier contract in Phase 3;
- context is a separate input and must match the fitted context dimension.

## Feature and output schema behavior

For pandas-like inputs, fit captures column names without importing pandas in the base
package. Duplicate columns are rejected. Prediction applies one of three policies:

| Policy | Behavior |
| --- | --- |
| `strict` | Require the same named features in the same order. |
| `reorder` | Require the same set, then safely reorder by name. |
| `positional` | Validate shape and use the supplied order. |

Missing and unexpected named features are rejected under `strict` and `reorder`.
Configured `input_shape` is checked before model construction. Fit records
`n_features_in_`, `feature_names_in_` when available, input shape, dtype, data format,
output names, and task metadata.

Built-in input and target scaler kinds, parameters, and fitted states are summarized
in `preprocessing_contract_` as structured data. `DataSchema` also reserves
identifier-based `categorical_encoder` and `missing_value_imputer` extension points.
Extensions are registered explicitly with `register_schema_transform`; importing the
base package never imports a workplace integration.

Feature, output, task, and preprocessing metadata survive the existing trusted
estimator save/load path. That path can execute Python pickle payloads and is not the
portable deployment contract.

## Registries and custom modules

The public registries cover backbones, activations, normalizations, dropout strategies,
optimizers, schedulers, losses, and metrics. Optional packages can register a
backbone or metric explicitly:

```python
psann.register_backbone(
    "acme.forecaster",
    factory,
    supported_tasks=("regression",),
    input_ranks=(1,),
    activations=("relu",),
    factory_kind="torch_module",
    plugin="acme-psann",
)
```

The specification stores only `"acme.forecaster"`. It never serializes the factory.
Registration is scoped to the current process and does not trigger plugin discovery
or integration imports.

`adapt_module(torch_module, ...)` and direct `TorchModuleAdapter` construction support
advanced in-process training and inference using the shared observable training loop;
they have no portable artifact guarantee. When `create_model` resolves a registered
factory declared with `factory_kind="torch_module"`, the regression adapter can
round-trip through the native artifact using the identifier, compatible plugin
registration, JSON-safe parameters, and restricted `state_dict`. Registered module
factories reject classification capability declarations; use `adapt_module` for
experimental in-process module classification. This path is experimental and does
not promise general Torch/ONNX export or gradient explanations.

## Scale and operations boundary

`TrainingConfig.amp_dtype` accepts canonical fp16/bf16 aliases for CUDA AMP.
`InferenceConfig.fallback_policy` makes unavailable-device behavior explicit.
`accelerator_support_matrix()` is the authoritative CPU/CUDA/MPS tier rather than
hardware discovery alone.

`train_streaming(model, stream, ...)` trains registered regression estimators over a
restartable `StreamingSupervisedData` source. `numpy_shard_stream` memory maps
uncompressed `.npy` shards. Its batch-local optimizer and unsupported
classification/resume/scheduler boundaries are intentional and fail clearly.

Training runs attach model/data fingerprints. `OperationalHooks` optionally forwards
redacted lifecycle events to caller-owned tracking, registry, and monitoring sinks;
no vendor package is imported by core. `RetentionPolicy` describes workplace storage
limits but does not delete external records. Performance comparison, secrets
handling, accelerator evidence, and supply-chain automation are documented in
[`workplace_operations.md`](workplace_operations.md).

## Persistence boundary

`TrainingRun.export(path)` writes the safe native `.psann` deployment bundle, and
`psann.load_model(path, device=...)` reconstructs every registered core backbone
through restricted `state_dict` loading. `psann.inspect_artifact(path)` validates
structure, JSON metadata, and per-file checksums without deserializing tensors.

`psann.load_runtime(path, config=...)` adds the Phase 5 raw-input deployment adapter:
bounded batching, eval/inference mode, concurrency isolation, task/output metadata,
and explicit stateful sessions. `torch.export` and ONNX are derived only after
alternate-batch parity certification and receive a generated preprocessing contract.
The optional reference service and container use this same runtime. See
[`deployment.md`](deployment.md).

`InferenceRuntime.make_explainer(...)` and `InferenceRuntime.explain(...)` add the
Phase 6 optional SHAP boundary. They require an explicit background, reference sample,
or persistence-approved summary and return a standard `shap.Explanation` alongside
artifact/run metadata. Model-agnostic explanations always use the runtime raw-input
path; gradient explanations use a frozen clone only when its preprocessing, layout,
context, state, activation, and output contract is certified. See
[`explainability.md`](explainability.md).

Feature/output schemas, fitted task metadata, built-in scaler state, runtime/package
requirements, training-run identity, plugin requirements, and optional model-card or
registry metadata travel with the artifact. Direct arbitrary modules, callables,
custom scaler objects, credentials, and raw training data are prohibited.

Resumable `.psann-train` checkpoints remain separate and are rejected by
`load_model`. Class-specific estimator snapshots remain a deprecated, trusted legacy
path because they can execute Python pickle payloads. Use
`migrate_legacy_checkpoint(..., trusted_legacy_checkpoint=True)` only after verifying
the source. See [`artifacts.md`](artifacts.md) for the format, trust model, migrations,
and failure behavior.
