# Workplace Operations and Scale Contract

Status: Active

Applies to: PSANN workplace 1.1 release-candidate line

This document defines the Phase 7 accelerator, bounded-data, performance, security,
and integration contract. It describes what automation certifies; it is not a claim
that PSANN supplies a hosted scheduler, registry, retention service, or control plane.

## Accelerator and dtype policy

Use `psann.accelerator_support_matrix()` for the machine-readable support table and
`psann.runtime_accelerator_evidence()` for a privacy-safe description of the active
runtime. `fallback_policy="error"` is recommended for certification and deployment.
`"warn"` permits only explicit fallbacks that emit a warning and metadata/event
evidence.

| Device | Operation | Dtype and mode | Tier | Evidence |
| --- | --- | --- | --- | --- |
| CPU | Training and inference | float32, no AMP or compile | Stable | Blocking CPU suite and weekly benchmark |
| CUDA | Training | float32; optional `torch.compile` | Stable | Weekly self-hosted CUDA suite |
| CUDA | Training | AMP float16 or bfloat16 | Stable when hardware supports it | Weekly CUDA dtype matrix; unsupported bfloat16 fails under strict policy |
| CUDA | Inference, native artifact load, export, explanation | float32 | Stable | Weekly CUDA lifecycle suite |
| Apple MPS | Training and inference | float32, no AMP or compile | Experimental | Non-blocking macOS observation job |
| XPU and other devices | All | All | Unsupported by the workplace API | Rejected before training/inference |
| Distributed/multi-process | Training | All | Out of scope | No public contract |

Stable inference and export accept float32 only. AMP is a training policy, not an
artifact dtype. CPU/MPS AMP or compile requests either warn and degrade through the
training fallback policy or fail when that policy is `"error"`. Arbitrary
`TorchModuleAdapter` models cannot silently ignore AMP or compile. Device unavailability
also follows the selected fallback policy. AMP and compile are each certified on CUDA,
but their simultaneous use is not certified and is rejected during configuration.
MPS export and explanation are unsupported.

Distributed training was evaluated and deferred: worker/rank ownership, distributed
sampler position, optimizer sharding, exact multi-worker resume, aggregated event
ordering, and single-writer artifact promotion do not yet have stable contracts.
Adding DDP before those contracts would make resume and audit evidence ambiguous.

The scheduled accelerator workflow covers forward/backward, native export and load,
resume, inference, AMP float16/bfloat16, compile, supported derived exports,
gradient explanations, and CUDA memory observations. A local CPU run does not count
as CUDA evidence.

## Bounded-data training

`StreamingSupervisedData` accepts a callable that returns a fresh iterable of bounded
batches. `numpy_shard_stream` provides a concrete path over uncompressed `.npy`
shards using `numpy.load(..., mmap_mode="r")`, so a complete shard does not need to be
loaded into memory.

```python
stream = psann.numpy_shard_stream(
    [
        psann.NumpyShard("data/x-000.npy", "data/y-000.npy"),
        psann.NumpyShard("data/x-001.npy", "data/y-001.npy"),
    ],
    batch_size=4096,
)
run = psann.train_streaming(
    psann.create_model(spec),
    stream,
    config=psann.TrainingConfig(epochs=2, batch_size=4096),
)
```

The first streaming contract intentionally supports registered regression estimators
only. Model weights warm-start between batches, while optimizer state is batch-local.
Classification vocabulary, scheduler/early-stopping semantics, compiled batch loops,
and cursor-aware resume/checkpoints fail explicitly. The batch source is restarted
once per logical epoch, must not be empty, and enforces matching input, target, and
context lengths plus `max_batch_samples`.

Ordinary inference remains bounded by `InferenceConfig.batch_size` and accepts empty,
single-row, large, non-contiguous, and mixed numeric inputs through the fitted schema
contract. Missing/reordered named columns and malformed context fail with actionable
errors.

## Fingerprints and operational metadata

`fingerprint_data(inputs, targets, context)` and `fingerprint_model(model)` produce
versioned SHA-256 identifiers. Data fingerprints stream array bytes into the digest
and do not return or place raw rows in artifacts. Streaming training combines
per-batch fingerprints. Native artifacts always compute their model fingerprint;
caller metadata cannot replace it.

A fingerprint is an integrity/correlation identifier, not anonymization. Small or
guessable datasets may be vulnerable to dictionary comparison. Treat fingerprints as
internal metadata and do not use them as authentication or proof of authorship.

`OperationalEvent` contains event kind, UTC timestamp, run/model identity, and
redacted metadata. `OperationalHooks` accepts caller-supplied experiment-tracker,
registry-publisher, and monitor callables without importing vendor SDKs:

```python
hooks = psann.OperationalHooks(
    experiment_tracker=my_tracker_sink,
    registry_publisher=my_registry_sink,
    monitor=my_monitor_sink,
    error_policy="raise",
)
run = psann.train(model, data, operational_hooks=hooks)
run.export("artifacts/model.psann")
```

Hooks run synchronously. `"raise"` is the auditable default; `"warn"` is an explicit
best-effort mode. Registry publication is triggered only after a native artifact has
been written successfully.

## Retention, redaction, and secrets

`RetentionPolicy` serializes recommended maximum retention and raw-data redaction
choices. Defaults are:

| Surface | Maximum retention |
| --- | --- |
| Training history | 90 days |
| Resume checkpoints | 30 days |
| Row-level explanations/backgrounds | 30 days |
| Reference-service logs | 14 days |

Raw inputs, targets, and context are redacted by default. These values are policy
metadata: the application, object store, registry, logging backend, or workplace
orchestrator must enforce deletion. PSANN does not delete external records.

Artifact metadata, registry extension metadata, and model cards reject credential-like
keys or values. `redact_sensitive` protects operational events as a second boundary.
Never put API keys, authorization headers, passwords, private keys, access tokens,
credential-bearing URLs, raw request rows, or explanation rows in manifests, logs,
benchmark summaries, or promoted reports. Pass credentials through the workplace
secret manager directly to the external integration.

The reference service records aggregate latency, batch size, device, errors, and
artifact identity without raw inputs. It still needs an approved authentication,
authorization, TLS, rate-limit, and request-size edge.

## Performance evidence

`tools/workplace_benchmark.py` measures:

- training samples/second;
- inference p50 and p95 latency;
- peak traced Python memory;
- native artifact load p50 latency;
- explanation wall time when `psann[explain]` is installed.

The checked-in reference is
[`benchmarks/workplace_cpu_baseline.json`](benchmarks/workplace_cpu_baseline.json).
Machine and dependency details are part of the record. Performance variation is noisy,
so regressions outside the per-metric tolerance produce a visible warning by default.
Incorrect prediction parity or explanation additivity is the blocking gate. Maintainers
may opt into `--fail-performance` only for a controlled runner with an intentionally
rebaselined reference.

The weekly performance workflow uploads the full observation for 90 days. Promote
only compact aggregate summaries—never raw model inputs—to version control.

## Vulnerability and SBOM evidence

The former supply-chain workflow is [archived](archive/workflows/README.md) and does
not currently perform or retain these checks. Its historical contract covered:

- an installed-environment `pip-audit` across core, serving, export, and explanation
  dependencies;
- a blocking repository secret scan;
- a non-root reference-service image build;
- a Trivy gate for fixed high/critical image vulnerabilities;
- SPDX JSON SBOM generation for the source/dependency surface, each release
  distribution, and container image;
- 90-day upload of scan/SBOM evidence.

Dependabot continues to observe Python, GitHub Actions, and Docker dependencies
weekly. The active tag-based container publication workflow generates SBOMs for both
package distributions and the image, but no tag exists for the candidate and those
future artifacts are not present evidence. Vulnerability results describe known
databases at scan time and do not replace threat modeling, artifact signatures,
provenance, or incident response.

## Promotion checklist

Before promoting a model:

1. train and resume under a supported accelerator/dtype combination with strict
   fallback policy;
2. retain the data/model fingerprints, configuration, metrics, and accelerator
   evidence without raw rows;
3. export and re-load the native artifact, then run task-specific parity;
4. apply the workplace retention policy to histories, checkpoints, explanations, and
   logs;
5. route optional hooks through approved sinks and secret management;
6. require current dependency/container vulnerability and SBOM evidence for the
   release environment.
