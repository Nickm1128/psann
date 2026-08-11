# ADR 0004: Artifact and Deployment Contract

- Status: Accepted
- Date: 2026-07-27
- Decision owner: Nickm1128
- Tracking issue: https://github.com/Nickm1128/psann/issues/2

## Context

Current estimator checkpoints serialize whole Python objects and load them with
unrestricted unpickling. That is convenient for local round trips but is not an
inspectable, portable, or safe workplace deployment boundary.

PyTorch documents that `torch.load` uses an unpickler and that untrusted inputs must not
be loaded. It recommends state-dict-oriented serialization and restricted
`weights_only` loading where applicable:
https://docs.pytorch.org/docs/stable/notes/serialization.html

## Decision

### Native artifact

The only mandatory stable deployment format is a native PSANN bundle with the
`.psann` extension and an independent artifact format version.

The initial logical layout is:

```text
model.psann
  manifest.json
  config/model.json
  schema/input.json
  schema/output.json
  weights/model.pt
  preprocessing/
  checksums.sha256
  model-card.md        # optional
```

Requirements:

- `manifest.json` declares artifact format, model, task, package, runtime, creation,
  capability, and experimental-status metadata.
- Configuration and schemas are JSON-safe and validated before construction.
- `weights/model.pt` contains a tensor/primitive state dictionary loadable through a
  restricted `weights_only=True` path.
- Built-in preprocessing state uses JSON and NumPy-safe arrays. Custom preprocessing
  requires an approved registry identifier.
- Every file has a cryptographic checksum.
- Writes are atomic and incomplete bundles are rejected.
- Arbitrary callables, estimator objects, modules, credentials, raw training data, and
  opaque pickle payloads are prohibited.

### Artifact versioning

`artifact_format_version` uses `MAJOR.MINOR` independently of the package version:

- major: incompatible layout or interpretation change;
- minor: backward-compatible additive fields or capabilities.

A package must read every artifact version it writes. A stable package major must keep
read or migration support for the immediately previous artifact major for its
documented support window.

### Training checkpoints

Resumable training uses a separate `.psann-train` checkpoint contract. It may contain
model, optimizer, scheduler, AMP, RNG, counters, history, and early-stopping state, but
it follows the same restricted-deserialization and checksum principles.

Training checkpoints are not accepted by deployment loaders.

### Loading

`psann.load_model(path, device=...)` is the generic deployment entry point. It:

1. validates bundle structure and checksums;
2. validates artifact version and required extensions;
3. validates JSON schemas;
4. constructs registered task, backbone, and preprocessing components;
5. loads tensor state through restricted deserialization;
6. verifies declared capabilities and fitted metadata;
7. returns a stateless-by-default inference object.

### Deployment and export support

- Native Python inference from `.psann` is the required stable format.
- `torch.export` and ONNX are derived exports, never the source of truth.
- An export is supported only for task/backbone/preprocessing combinations passing
  numerical, dynamic-shape, and environment parity tests.
- The reference HTTP service demonstrates health, readiness, metadata, and batched
  prediction. It is not a hosted control plane or a mandatory base dependency.
- Stateful and streaming inference uses explicit request/session state rather than
  mutable shared deployment state.

## Consequences

- New artifacts can be inspected and validated without importing arbitrary model
  classes from a pickle stream.
- Training resume and deployment have distinct contracts.
- Export support can grow without weakening the native artifact.
- Custom modules need registration to receive artifact guarantees.

## Rejected alternatives

- Continuing whole-object pickle as the workplace format is unsafe and version-fragile.
- Making ONNX the only artifact would exclude valid PSANN/stateful capabilities and
  couple persistence to exporter coverage.
- Embedding preprocessing only in service code would make offline and online
  predictions diverge.
