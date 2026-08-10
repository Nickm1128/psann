# Security Policy

## Supported Versions

The public stable package is currently `0.12.7`. The selected workplace candidate is
`1.1.0rc1`; release candidates receive fixes during certification but are not a
production support claim.

| Version line | Security status |
| --- | --- |
| `1.1.0rc1` | Candidate: fixes are applied before promotion; not yet published as stable |
| `0.12.7` | Supported legacy stable line until 90 days after `1.1.0` GA |
| `<0.12.7` | Unsupported; upgrade or migrate before reporting a version-specific issue |

After `1.1.0` is published, the latest `1.1.x` patch becomes the primary supported
line. The `0.12.7` transition window and artifact/checkpoint migration commitments are
defined in [`docs/support_policy.md`](docs/support_policy.md).

## Reporting a Vulnerability

Do not open a public issue for a suspected vulnerability.

Use the repository's private
[GitHub security advisory form](https://github.com/Nickm1128/psann/security/advisories/new)
and include:

- affected version and installation method;
- a minimal reproduction or malformed input;
- impact and the trust boundary crossed;
- whether model files, checkpoints, datasets, or credentials are involved;
- any temporary mitigation already tested.

Please avoid attaching real secrets, sensitive training data, or proprietary model
artifacts. Use a synthetic reproducer wherever possible.

## Model and Checkpoint Trust

The native `.psann` artifact validates its fixed bundle layout, JSON contracts,
runtime/plugin requirements, and per-file SHA-256 checksums before reconstructing a
registered model. Model state is loaded through `torch.load(weights_only=True)`.
Use `psann.inspect_artifact` to validate metadata without tensor deserialization.

Checksums detect corruption but do not authenticate the artifact author. Internally
distributed and external artifacts still require an organization-approved signature,
digest, or registry attestation before loading. Do not put credentials, raw training
data, or sensitive identifiers in manifest metadata or model cards. See
`docs/artifacts.md` for the full trust model.

Class-specific estimator checkpoints use Python/Torch whole-object serialization and
emit `LegacyCheckpointWarning`. Loading one can execute code. Generic loading rejects
legacy files unless `trusted_legacy_checkpoint=True`, and the migration tool requires
the same explicit acknowledgement. Only use that path after independently verifying
the file's source and contents.

## Deployment Service

The reference service logs request latency, batch size, error type, device, and
artifact identity; it does not log raw inputs or context. Its `/metadata` endpoint
returns a reduced service-safe view rather than the full manifest metadata.

The reference worker does not provide authentication, authorization, TLS termination,
rate limiting, or request-size enforcement. Deploy it behind an organization-approved
edge and mount the reviewed native artifact read-only. Registry resolvers must return
a local file; registration is explicit and PSANN still validates the resulting native
bundle before loading.

## Explanation Data

SHAP backgrounds, raw feature values, feature/output names, and row-level
attributions may contain or reveal personal, confidential, or proprietary data.
PSANN never selects training data implicitly and never embeds an explanation
background in the native `.psann` artifact. Persisted explainer configuration includes
background values only after the caller marks the summary approved and explicitly
requests inclusion.

Keep explanation work offline or behind separately authenticated, authorized,
rate-limited endpoints. Do not log raw explanation requests or row-level results. The
aggregate drift report intentionally excludes raw rows, but its feature names and
importance statistics still require normal organizational review and retention
controls. See [`docs/explainability.md`](docs/explainability.md).

## Operational Metadata and Supply Chain

Artifact metadata, registry extension metadata, and model cards reject
credential-like keys and values. Operational hooks receive redacted event metadata.
Do not place credentials, authorization headers, private keys, raw request rows, or
row-level explanations in manifests, logs, hooks, benchmark summaries, or promoted
reports. A data fingerprint is an integrity identifier, not anonymization.

Weekly automation audits installed Python dependencies, scans the reference container
for fixed high/critical vulnerabilities, and generates source and image SBOMs.
Release owners must review current evidence rather than treating a prior scan as a
permanent assurance. Retention defaults and the complete secrets boundary are in
[`docs/workplace_operations.md`](docs/workplace_operations.md).

## Response Expectations

Maintainers will acknowledge a complete report when they next review the private
advisory queue, assess severity, coordinate a fix and disclosure window, and credit
the reporter if requested. Exact response times are not guaranteed for this
research-stage project.
