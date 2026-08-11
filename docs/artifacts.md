# Native Model Artifacts

Status: Active

The workplace 1.1 line introduces `.psann` as the stable model-deployment boundary. A native
artifact stores validated JSON configuration and fitted metadata separately from a
restricted-load tensor `state_dict`; it never embeds an estimator, arbitrary module,
callable, credential, or raw training row.

## Export, inspect, and load

```python
from pathlib import Path

import psann

model = psann.create_model(spec)
run = psann.train(model, (train_inputs, train_targets), config=training_config)

artifact = run.export(
    Path("artifacts") / "demand_forecast.psann",
    model_card="# Demand forecast\nOwned by the Forecasting team.\n",
    metadata={"owner": "forecasting"},
    registry={"uri": "registry://forecasting/demand"},
)

info = psann.inspect_artifact(artifact)
print(info.artifact_id, info.backbone, info.task, info.run_id)

deployed = psann.load_model(artifact, device="cpu")
predictions = deployed.predict(batch)
```

`inspect_artifact` validates the complete bundle and its checksums but does not call
`torch.load` or construct a model. `load_model` repeats validation, verifies runtime
and extension requirements, rebuilds the registered model, restricted-loads the
tensor state with `weights_only=True`, and applies it strictly. The returned estimator
is in evaluation mode; Phase 5 adds the higher-level batched and concurrency-safe
deployment adapter.

`TrainingRun.export` requires a `.psann` filename and writes through an atomic
same-directory temporary file. If writing or replacement fails, an existing target
is left unchanged and the temporary file is removed.

## Bundle layout

Artifact format `1.0` uses this fixed layout:

```text
model.psann
  manifest.json
  config/model.json
  config/fitted.json
  schema/input.json
  schema/output.json
  weights/model.pt
  preprocessing/state.json
  checksums.sha256
  model-card.md              # optional
```

The checksum index covers every other member, including the manifest and optional
model card. Loading rejects duplicate, missing, unexpected, unsafe-path, oversized,
truncated, or checksum-mismatched members before tensor deserialization.

The manifest records:

- artifact, manifest-schema, package, Python, NumPy, and PyTorch versions;
- minimum compatible Python, PSANN, and PyTorch versions;
- artifact identity and UTC creation time;
- task, registered backbone, plugin, capabilities, and experimental status;
- training-run identity;
- required registered schema transforms;
- optional JSON-safe organizational metadata and external-registry references.

Unknown optional manifest metadata is ignored within the current artifact major.
Unknown required members or extensions fail closed. `ARTIFACT_MANIFEST_JSON_SCHEMA`
is public from `psann.platform` for registry tooling, and runtime validation has no
extra JSON-schema dependency.

## Integrity and trust

Checksums detect accidental corruption and incomplete transfers. They do not prove who
created a model. Use these trust tiers:

1. **Locally created artifact**: validate with `inspect_artifact`; protect the source
   workspace and dependency environment.
2. **Internally distributed artifact**: validate the artifact and verify an
   organization-controlled signature, digest, or registry attestation outside the
   bundle before loading.
3. **Untrusted external artifact**: do not treat checksums as authentication. Verify
   provenance and an approved signature in a sandboxed intake process first.

The loader never imports a plugin named by the artifact. Optional packages must be
installed and explicitly register the required backbone or schema transform before
loading. Plugin identifiers and versions are checked against the current process.

Do not place credentials, access tokens, private paths, customer identifiers, raw
training examples, or sensitive statistics in model cards, `metadata`, or `registry`
fields.

## Supported and unsupported state

All eight core registered backbones support native artifacts for regression, binary
classification, multiclass classification, and multilabel classification. Artifacts
preserve:

- fitted input/output shape, feature names, output names, dtype, and layout;
- task classes, labels, probability semantics, and thresholds;
- built-in `standard` and `minmax` input/target scaler states;
- context dimensions and registered string context-builder configuration;
- model tensor state and required WaveResNet schedule state.

A direct arbitrary `TorchModuleAdapter`, callable context builder, opaque LSM module,
or custom scaler object is rejected. A registered backbone factory that returns a
`torch.nn.Module` can use the native artifact path: the artifact stores its registry
identifier, plugin/version requirement, JSON-safe `ModelSpec`, and restricted
`state_dict`. It never serializes the factory. Loading therefore requires the same
compatible registration in the process and remains experimental, with no general
derived-export or gradient-explanation guarantee.

`.psann-train` files are resumable training checkpoints, not deployment artifacts.
`load_model` rejects them and directs callers to `resume_from`.

## Artifact versions and migration

`artifact_format_version` is independent of the Python package version. Major changes
may alter interpretation or layout; minor changes are additive. The loader migrates
supported historical manifests in memory before current-schema validation.

Native format `1.0` begins its supported producer history with the 1.1 line. The
repository's `0.13` through `0.16` labels were internal workplace phase labels, not
published producer releases, and no immutable native artifacts exist for them. They
are therefore not historical compatibility claims.

The loader also migrates manifest format `0.9` to `1.0` in memory. Its test fixture is
a deliberately modified current-format artifact and proves schema migration only;
it is not represented as an authentic historical producer artifact. Newer or unknown
formats fail closed.

Use `migrate_artifact(source, destination)` to rewrite a supported historical bundle
with the current manifest. It always writes to an explicit destination; it does not
silently overwrite the source.

## Legacy whole-object checkpoints

Class-specific estimator `.save()` and `.load()` remain temporarily available but
emit `LegacyCheckpointWarning`. Those files use unrestricted Python pickle and may
execute code during loading.

Generic loading refuses a legacy checkpoint unless trust is explicit:

```python
trusted_model = psann.load_model(
    "old_model.pt",
    trusted_legacy_checkpoint=True,
)
```

Prefer converting a verified legacy file immediately:

```python
safe_path = psann.migrate_legacy_checkpoint(
    "old_model.pt",
    "artifacts/model.psann",
    trusted_legacy_checkpoint=True,
)
```

The trust flag does not make the source safe. It confirms that the caller has already
established provenance and accepts unrestricted deserialization. Migration then
writes a new artifact that passes the native structure, checksum, JSON, and
restricted-state contract.

The repository retains a provenance- and hash-checked checkpoint created by the
public `0.12.7` wheel and continuously verifies trusted load, migration, and numerical
parity. See [`compatibility_evidence.md`](compatibility_evidence.md) for the exact
support boundary and reproduction command.

See [`ADR 0004`](adr/0004-artifact-and-deployment-contract.md) for the normative
format decision and [`ADR 0005`](adr/0005-legacy-deprecation-policy.md) for the legacy
removal timeline.
