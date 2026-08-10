# Deployment-Grade Inference

Status: Active

Workplace Phase 5 adds a schema-aware deployment runtime over the native `.psann` artifact.
The runtime is stateless by default, bounds accelerator memory through raw-input
batching, carries stable result metadata, and separates mutable streaming state into
explicit sessions.

## Load and predict

```python
import psann

runtime = psann.load_runtime(
    "artifacts/demand.psann",
    config=psann.InferenceConfig(
        batch_size=512,
        device="cpu",
        feature_policy="reorder",
        device_transfer="per_batch",
    ),
)

result = runtime.predict(raw_features)
predictions = result.values
print(result.task, result.output_names, result.model_id, result.run_id)
```

`load_runtime` validates and restricted-loads the native artifact through
`load_model`; it does not add another persistence path. A fitted in-process estimator
can be wrapped with `create_inference_runtime(model, config=...)`.

`InferenceResult` contains:

- `values`: predictions, probabilities, labels, or logits;
- `task` and `output_names`;
- artifact format version, artifact/model identifier, and training-run identifier;
- request metadata: configured batch size, actual chunk count, sample count, device,
  dtype, transfer policy, and output kind.

Classification returns probabilities by default. Set
`classification_output="label"` for fitted labels or pass `return_logits=True` for
raw logits. For multiclass probability output, `InferenceConfig(top_k=k)` retains the
complete matrix in `InferenceResult.values` and adds ranked labels, probabilities,
and class indices in `InferenceResult.top_k`. It rejects other tasks, label/logit
output, and `k` larger than the fitted class count.

Regression returns inverse-target-transformed predictions. The stable
runtime currently accepts `float32`; another dtype fails explicitly rather than
silently changing numerical behavior.

## Batching and device policy

`device_transfer="per_batch"` is the bounded-memory default. Raw inputs and optional
context are sliced together, then schema validation, fitted scaling, layout
transforms, context handling, model execution, task conversion, and inverse target
scaling run through the same estimator path used before export.

`device_transfer="full_batch"` is an explicit throughput-oriented option. It sends
the full request through one prediction call and can require substantially more
host/accelerator memory.

Every call uses `torch.inference_mode()` and eval mode. Runtimes sharing the same
Torch module also share a request lock, so estimator mode/device bookkeeping cannot
race. Repeated and concurrent stateless requests are verified not to change
parameters or persistent buffers.

## Stateful sessions

Ordinary `runtime.predict(...)` never advances a fitted state controller. A model
trained with `stateful=True` exposes an isolated session:

```python
with runtime.create_session(session_id="request-42") as session:
    first = session.step(raw_step_1)
    second = session.step(raw_step_2)
    sequence = session.predict_sequence(raw_sequence, reset_state=True)
```

Each session deep-copies the fitted estimator, advances and commits only its own
state, and optionally applies explicitly requested online parameter updates. Closing
the session releases the copy and makes later calls fail. Non-stateful models reject
session creation.

## Device pools and registry adapters

Single-device behavior is the foundation. When a worker owns multiple devices,
`load_runtime_pool` loads one independent model per device and routes requests
round-robin:

```python
pool = psann.load_runtime_pool(
    "artifacts/demand.psann",
    devices=("cuda:0", "cuda:1"),
    config=psann.InferenceConfig(batch_size=256),
)
result = pool.predict(raw_features)
```

External registries remain optional and explicit. A plugin resolves its URI to a
local artifact; PSANN still performs the normal checksum, schema, version, extension,
and restricted-weight checks:

```python
psann.register_artifact_resolver(
    "acme",
    lambda uri: internal_registry.download_to_cache(uri),
)
runtime = psann.load_registry_runtime("acme://forecasting/demand/42")
```

There is no implicit plugin discovery, network access, credential serialization, or
vendor-specific field in the native artifact.

## Derived exports

Install optional exporters with:

```bash
pip install "psann[export]"
```

Derived formats are never the source of truth. `evaluate_export_capabilities`
captures the fitted tensor module, declares batch dimension `1..1,000,000`, executes
an alternate batch size, and checks numerical parity before a format is advertised:

```python
report = psann.evaluate_export_capabilities(
    runtime.model,
    representative_raw_inputs,
    formats=("torch_export", "onnx"),
    atol=1e-5,
    rtol=1e-4,
)
print(report.advertised_formats)

exported = psann.export_derived(
    runtime.model,
    "artifacts/demand.pt2",
    format="torch_export",
    sample_inputs=representative_raw_inputs,
)
```

`export_derived` refuses an uncertified format. It writes `.pt2` for `torch.export`
or `.onnx` for ONNX plus a sibling
`<filename>.preprocessing.json`. The contract records the raw schema, fitted input
preprocessing, tensor layout/context, task conversion, target inverse transform,
artifact identity, tolerance evidence, and dynamic-batch result. The exported graph
accepts prepared tensors; consumers must implement that generated contract.

The Phase 5 matrix certifies both formats for regression, binary, multiclass, and
multilabel tasks across all eight registered stable backbones. The native `.psann`
bundle remains mandatory even when a derived format is available. A different
dependency/runtime combination must re-run capability evaluation rather than inherit
the claim.

## Reference HTTP service

Install the optional service dependencies:

```bash
pip install "psann[serve]"
python -m psann.serving --artifact artifacts/demand.psann --device cpu --port 8080
```

Endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Process liveness; does not claim a model is loaded. |
| `GET /ready` | Returns 200 only after the artifact loads successfully. |
| `GET /metadata` | Service-safe task, schema, artifact, run, device, and state metadata. |
| `GET /metrics` | Aggregate request, error, sample, and average-latency counters. |
| `POST /predict` | Batched raw-input inference with optional context and batch override. |

Request example:

```json
{
  "inputs": [[1.2, 0.4, 7.0], [0.9, 0.3, 5.0]],
  "batch_size": 128,
  "return_logits": false
}
```

Each handled prediction emits a structured JSON log containing latency, batch size,
status/error type, device, and artifact identity. Raw inputs and context are never
logged. Unknown request fields are rejected.

## Container

The reference image is defined by `deploy/Dockerfile`, runs as a non-root user, uses
the locked Python 3.11 CPU snapshot in `constraints/deployment-py311.txt`, and expects
a read-only artifact at `/artifacts/model.psann`.

```bash
docker build -f deploy/Dockerfile -t psann-serving:0.14.0 .
docker run --rm \
  -p 8080:8080 \
  --mount type=bind,source="$PWD/artifacts/demand.psann",target=/artifacts/model.psann,readonly \
  psann-serving:0.14.0
```

PowerShell uses an absolute mount source:

```powershell
$artifact = (Resolve-Path artifacts/demand.psann).Path
docker run --rm -p 8080:8080 `
  --mount "type=bind,source=$artifact,target=/artifacts/model.psann,readonly" `
  psann-serving:0.14.0
```

The container workflow builds the image, creates and mounts a native smoke artifact,
checks health/readiness/metadata/prediction, and publishes
`ghcr.io/<owner>/<repository>/serve:<tag>` only for version tags.

## Operational boundaries

- The reference service demonstrates a safe worker, not authentication, authorization,
  rate limiting, request queues, model rollout, or a hosted control plane.
- Terminate TLS and enforce request-size/time limits at the platform edge.
- Mount only reviewed native artifacts. Derived exports do not weaken the native
  artifact trust rules.
- Use one process per accelerator unless the workload has independently validated a
  different memory/concurrency policy.
- Do not place credentials, raw training data, or sensitive feature values in artifact
  metadata.
