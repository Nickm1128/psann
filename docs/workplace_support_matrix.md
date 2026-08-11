# Workplace Platform Support Matrix

Status: `1.1.0rc1` release-candidate contract

Last reviewed: 2026-08-11

Owner: Nickm1128

Tracking issue: https://github.com/Nickm1128/psann/issues/2

This matrix records the final intended `workplace-v1` support commitment in the 1.1
release line and the evidence required to promote `1.1.0rc1`. The selected naming
contract is [`release_identity.md`](release_identity.md). A release-candidate
implementation is not a published support claim: promotion requires the
clean-checkout, built-wheel, CPU, CUDA, security, container, and migration evidence
to pass for the exact tagged commit. The workflows that formerly aggregated release
certification and supply-chain evidence were [archived on
2026-08-11](archive/workflows/README.md), so the candidate currently has no active
promotion path.

## Final certification tiers

| Tier | Included capabilities | Evidence |
| --- | --- | --- |
| Stable after promotion | Regression, binary/multiclass/multilabel classification; eight core registered backbones where the common matrix applies; native `.psann`; bounded stateless inference; explicit sessions for supported stateful models; model-agnostic SHAP with explicit backgrounds | [Local six-scenario contract tests](../tests/test_workplace_certification.py) |
| Capability-gated | Gradient/deep SHAP, `torch.export`, ONNX, CUDA AMP/compile, context/state combinations, and derived exports | [Capability and parity tests](../tests/test_deployment_exports.py) |
| Experimental | Apple MPS, GeoSparse, PSANN-LM workplace integration, and registered custom `torch.nn.Module` factories | [Experimental-boundary tests](../tests/test_platform_contracts.py) |
| Out of scope | Distributed workplace training, hosted registry/control plane, arbitrary-module portability, and universal export support | [Accepted task/backbone decision](adr/0002-task-and-backbone-support.md) |

Local Phase 8 CPU evidence covers all six scenarios with warnings promoted to errors.
CUDA remains a blocking promotion job, not a claim that can be substituted by CPU
results. Archived workflows and local runs do not constitute exact-commit promotion
evidence.

## Artifact compatibility provenance

| Line or format | Candidate status | Evidence |
| --- | --- | --- |
| Public `0.12.7` legacy `.pt` | Explicit-trust load and migration supported | [Retained fixture, pinned producer wheel, and numerical parity](compatibility_evidence.md#public-0127-fixture-provenance) |
| Native artifact format `1.0` | Producer support begins with the 1.1 line | [Current artifact contract and certification path](artifacts.md#artifact-versions-and-migration) |
| Manifest format `0.9` | In-memory schema migration supported | [Synthetic schema-test boundary](compatibility_evidence.md#supported-evidence) |
| Internal `0.13`-`0.16` phase labels | Not released producer lines; no compatibility claim | [Repository/public-producer audit result](compatibility_evidence.md#supported-evidence) |

## Status Key

| Status | Meaning | Evidence authority |
| --- | --- | --- |
| Stable | Covered by the frozen contract after all release promotion gates pass | [Support policy](support_policy.md) |
| Current | Available through the current public estimator API and covered by existing tests | [Public API freeze](workplace_public_api.json) |
| Target | Required for the stable workplace milestone but not yet certified | [Release roadmap](backlog/workplace_nn_platform_todo.md) |
| Experimental | Available for research or evaluation without the stable compatibility guarantee | [Support policy](support_policy.md) |
| Capability-gated | Supported only for combinations that pass an explicit parity matrix | [Certification contract](workplace_operations.md) |
| Out of scope | Not promised for the first workplace milestone | [Accepted lifecycle decision](adr/0001-workplace-lifecycle-api.md) |

## Runtime and Dependency Matrix

| Component | `1.1.0rc1` candidate | Workplace commitment | Required evidence |
| --- | --- | --- | --- |
| Python | Declares `>=3.11` | 3.11, 3.12, 3.13 | [Blocking installed-wheel matrix](../.github/workflows/ci.yml) |
| Python 3.9/3.10 | Prior estimator releases only | Legacy estimator line only | [Public 0.12.7 inventory](public_api_0_12_7.json) |
| Python 3.14 | Unsupported by the 1.1 line | Re-evaluate after dependency wheels are available | [Package metadata and CI boundary](../pyproject.toml) |
| NumPy | Declares `>=1.26` | Floor 1.26 plus admitted current 2.x | [Floor/current dependency jobs](../.github/workflows/ci.yml) |
| PyTorch | Declares `>=2.4` | Floor 2.4 plus current stable | [Pinned 2.4 floor job](../.github/workflows/ci.yml) |
| scikit-learn | Optional extra declares `>=1.4,<2` | Floor 1.4 plus current pre-2.0 | [Floor/current dependency jobs](../.github/workflows/ci.yml) |
| SHAP | Optional `psann[explain]`: 0.50-0.51 on Python 3.11 and 0.50-0.52 on Python 3.12+ | Python-minor-compatible floor/current bands | [3.11-3.13 and floor/current explain jobs](../.github/workflows/ci.yml) |
| PSANN-LM | Alpha `1.1` line requires `psann>=1.1.0rc1,<1.2` and reports its own version | Experimental; outside the stable workplace API | [Compatibility tests and Alpha coverage floor](quality_policy.md) |

The workplace Python minimum is 3.11 because SHAP 0.50 and later require Python 3.11
or newer. SHAP 0.50+ requires NumPy 2, so the explanation extra and the conservative
NumPy 1.26 `compat` snapshot are validated as separate environments.

## Operating-System and Accelerator Matrix

| Environment | Target tier | Workplace commitment | Evidence |
| --- | --- | --- | --- |
| Linux x86_64 CPU | Tier 1 | Blocking correctness, package, artifact, inference, service, and SHAP jobs | [Installed-wheel CI](../.github/workflows/ci.yml) |
| Linux x86_64 CUDA | Tier 1 after candidate evidence | Scheduled training, resume, artifact, inference, AMP, and export jobs | [Scheduled accelerator workflow](../.github/workflows/accelerator.yml) |
| Windows x86_64 CPU | Tier 1 | Blocking package, artifact, inference, service, and SHAP smoke on Python 3.11-3.13 | [Windows installed-wheel CI](../.github/workflows/ci.yml) |
| Windows x86_64 CUDA | Unsupported | No 1.1 stable claim; re-evaluate after a retained Windows CUDA job exists | [Current accelerator workflow scope](../.github/workflows/accelerator.yml) |
| macOS arm64 CPU | Experimental observation | No package-support claim from the current MPS-only observation | [Non-blocking MPS observation](../.github/workflows/accelerator.yml) |
| Apple MPS | Experimental | No stable guarantee until training and explanation behavior is reliable | [MPS behavior tests](../tests/gpu/test_workplace_mps.py) |
| Distributed training | Out of scope | Reconsider only after single-device resume and deployment are certified | [Accepted task/backbone decision](adr/0002-task-and-backbone-support.md) |

CPU correctness is required for every stable core capability. Device, dtype, AMP, and
compile combinations that are not listed as supported must fail clearly or use a
documented fallback.

| Device | Stable dtype/mode | Experimental | Unsupported | Evidence |
| --- | --- | --- | --- | --- |
| CPU | float32 training/inference | None | AMP, compile, reduced-precision inference | [Device-policy tests](../tests/test_device_dtype_policy.py) |
| CUDA | float32 lifecycle; fp16/bf16 training AMP; float32 compile | None | Reduced-precision artifact/inference/export/explanation | [CUDA lifecycle suite](../tests/gpu/test_workplace_cuda.py) |
| Apple MPS | None | float32 training/inference | AMP, compile, reduced precision | [MPS observation suite](../tests/gpu/test_workplace_mps.py) |
| XPU/other | None | None | Workplace lifecycle | [Fail-closed device policy](../tests/test_device_dtype_policy.py) |

The machine-readable policy is `accelerator_support_matrix()`. Weekly CUDA evidence
covers forward/backward, save/load, resume, inference, AMP, compile, exports,
explanations, and memory. MPS observation is non-blocking. Distributed training is
out of scope until worker-aware resume, aggregated events, and artifact ownership
have stable contracts.

## Task Matrix

| Task | `1.1.0rc1` API | Workplace commitment | Evidence |
| --- | --- | --- | --- |
| Regression | Certified lifecycle, target scaling, resume, artifact, batching, and SHAP scenario | Stable | [Certification regression scenario](../tests/test_workplace_certification.py) |
| Binary classification | Certified probabilities, threshold, metrics, artifact, service, and SHAP scenario | Stable classifier contract | [Certification binary scenario](../tests/test_workplace_certification.py) |
| Multiclass classification | Certified labels, probability matrix, top-k, artifact parity, and per-class SHAP scenario | Stable classifier contract | [Certification multiclass scenario](../tests/test_workplace_certification.py) |
| Multilabel classification | Current through `PSANNClassifier` with explicit thresholds | Stable classifier contract | [Task/backbone and threshold matrix](../tests/test_workplace_platform.py) |

Task adapters own target validation, losses, metrics, probability conversion, class or
label metadata, and thresholds. A backbone does not become stable for a task merely
because its output shape can represent that task.

## Backbone Matrix

| Registry identifier | Current implementation | Current public maturity | Workplace target | Evidence |
| --- | --- | --- | --- | --- |
| `psann_mlp` | `PSANNRegressor` and dense Torch modules | Registered task/shape API | Stable for applicable tasks | [Common backbone matrix](../tests/test_workplace_platform.py) |
| `respsann_mlp` | `ResPSANNRegressor` | Registered task/shape API | Stable for applicable tasks | [Common backbone matrix](../tests/test_workplace_platform.py) |
| `psann_conv1d` | Convolutional Torch modules | Registered task/shape API | Stable after common matrix | [Common backbone matrix](../tests/test_workplace_platform.py) |
| `psann_conv2d` | Convolutional Torch modules | Registered task/shape API | Stable after common matrix | [Common backbone matrix](../tests/test_workplace_platform.py) |
| `psann_conv3d` | Convolutional Torch modules | Registered task/shape API | Stable after common matrix | [Common backbone matrix](../tests/test_workplace_platform.py) |
| `respsann_conv2d` | `ResConvPSANNRegressor` | Registered task/shape API | Stable for applicable tasks | [Common backbone matrix](../tests/test_workplace_platform.py) |
| `wave_resnet` | `WaveResNetRegressor` | Registered task/shape API | Stable for applicable tasks | [Common backbone matrix](../tests/test_workplace_platform.py) |
| `sgr_psann` | `SGRPSANNRegressor` | Registered task/shape API | Stable for applicable tasks | [Common backbone matrix](../tests/test_workplace_platform.py) |
| GeoSparse family | Experimental modules and estimators | Experimental | Remains experimental until it passes the common matrix | [Experimental API classification](public_api.md#experimental-apis) |
| Arbitrary `torch.nn.Module` | `TorchModuleAdapter` | Explicitly limited | In-process regression training/inference only | [Early task rejection tests](../tests/test_platform_contracts.py) |
| Registered custom `torch.nn.Module` factory | `TorchModuleAdapter` plus registry/plugin identity | Experimental | Regression native-artifact round trip only; no general derived-export or gradient-SHAP guarantee | [Registered-module artifact tests](../tests/test_model_artifacts.py) |

“Applicable tasks” is resolved by the common task, shape, schema, training, artifact,
deployment, and explanation tests. LSM is a preprocessor/expander, not a backbone.

## Activation Matrix

| Activation | Current baseline | Workplace target | Evidence |
| --- | --- | --- | --- |
| `psann` | Current | Stable | [Activation factory matrix](../tests/test_workplace_platform.py) |
| `relu` | Current | Stable | [Activation factory matrix](../tests/test_workplace_platform.py) |
| `tanh` | Current | Stable | [Activation factory matrix](../tests/test_workplace_platform.py) |
| `sigmoid` | Current, including a compatibility alias | Stable canonical identifier | [Activation and alias tests](../tests/test_activation.py) |
| `relu_sigmoid_psann` | Current | Stable | [Activation tests](../tests/test_activation.py) |
| `gelu` | Current in dense, residual, and convolutional PSANN cores | Stable | [Activation factory matrix](../tests/test_workplace_platform.py) |
| `silu` | Current in dense, residual, and convolutional PSANN cores | Stable | [Activation factory matrix](../tests/test_workplace_platform.py) |
| Mixed and GeoSparse-specific activations | Experimental | Experimental until serialized and export-tested | [Experimental GeoSparse tests](../tests/test_mixed_activation.py) |

## Artifact, Inference, Export, and Explanation Matrix

| Capability | `1.1.0rc1` candidate | Workplace guarantee | Evidence |
| --- | --- | --- | --- |
| Estimator save/load | Deprecated whole-object Torch serialization with security warning | Legacy trusted migration path only | [Authentic 0.12.7 migration fixture](compatibility_evidence.md) |
| Native deployment artifact | `.psann` safe bundle, inspection, migrations, and generic `load_model` | Stable `.psann` safe bundle and generic `load_model` | [Artifact matrix](../tests/test_model_artifacts.py) |
| Training resume checkpoint | Checksummed, restricted-load `.psann-train` for core supervised regressors | Stable for documented supervised paths | [Resume matrix](../tests/test_training_resume.py) |
| Native Python inference | `InferenceRuntime` | Stable, schema-validated, batched, stateless by default | [Inference robustness matrix](../tests/test_inference_robustness.py) |
| Stateful inference | Isolated `InferenceSession` | Explicit session object only | [Session isolation tests](../tests/test_deployment_inference.py) |
| `torch.export` | Certified task/backbone combinations | Capability-gated by task/backbone parity | [Derived-export matrix](../tests/test_deployment_exports.py) |
| ONNX | Certified combinations with `psann[export]` | Capability-gated by task/backbone parity | [Derived-export matrix](../tests/test_deployment_exports.py) |
| Reference HTTP service | Optional FastAPI worker and non-root container | Optional reference deployment, not a hosted control plane | [Service and container tests](../tests/test_reference_service.py) |
| SHAP model-agnostic | Raw-input permutation/partition explanations | Stable for registered deployed artifacts and explicit backgrounds | [Explainability matrix](../tests/test_explainability.py) |
| SHAP gradient-based | Frozen differentiable adapter | Capability-gated by preprocessing, context, activation, output, and state checks | [Gradient capability tests](../tests/test_explainability.py) |

The native `.psann` artifact is the only mandatory deployment format. Derived exports
are never the source of truth and are not advertised unless their exact combination
passes numerical and dynamic-shape parity.

The Phase 5 certification uses `atol=1e-5`, `rtol=1e-4`, and an alternate dynamic
batch for all four tasks across the eight registered stable backbones. Claims on a
different Torch/ONNX runtime must be re-evaluated. See
[`deployment.md`](deployment.md).

The Phase 6 explanation matrix covers deployed regression and classification outputs,
multi-output shapes, named features/classes, artifact parity, fixed-seed determinism,
state isolation, built-in scaler/layout parity, and domain groups for spatial and
sequence data. Deep explanations are deliberately narrower than gradient
explanations. See [`explainability.md`](explainability.md).

## Scale, Operations, and Supply Chain

| Capability | Current contract | Workplace guarantee | Evidence |
| --- | --- | --- | --- |
| Large training data | Restartable bounded batches and memory-mapped `.npy` shards | Stable for registered regression; batch-local optimizer state is explicit | [Streaming tests](../tests/test_streaming_platform.py) |
| Fingerprints | SHA-256 data/model identifiers without embedded raw rows | Stable integrity/correlation metadata, not authentication or anonymization | [Operational contract tests](../tests/test_phase7_operations.py) |
| Retention/redaction | Serializable maximum-retention policy and redacted events | Host storage/logging systems enforce deletion | [Operational contract tests](../tests/test_phase7_operations.py) |
| External operations | Optional synchronous tracking, registry, and monitor hooks | No vendor SDK in core; strict hook failures by default | [Operational hook tests](../tests/test_phase7_operations.py) |
| Performance | Versioned throughput/latency/memory/load/explanation observations | Correctness blocks; timing/memory regressions warn by default | [Scheduled performance workflow](../.github/workflows/performance.yml) |
| Dependency/container security | Automation archived; local tools remain available | No current release guarantee; candidate promotion is blocked | [Archived workflow decision](archive/workflows/README.md) |
| SBOM | Historical source and image generation definitions retained | No current candidate or tag evidence | [Archived workflow decision](archive/workflows/README.md) |

See [`workplace_operations.md`](workplace_operations.md) for failure policies,
limitations, secrets handling, and promotion requirements.

## Evidence Required to Change a Status

A capability moves to stable only when:

1. its public contract and failure behavior are documented;
2. floor and current dependency environments pass;
3. built-wheel tests pass on every claimed Tier 1 environment;
4. applicable task, shape, schema, artifact, inference, and explanation tests pass;
5. accelerator claims have recent scheduled evidence;
6. migration and deprecation guidance exists for replaced behavior.

The governing decisions are
[`ADR 0002`](adr/0002-task-and-backbone-support.md),
[`ADR 0003`](adr/0003-compatibility-and-support-policy.md), and
[`ADR 0004`](adr/0004-artifact-and-deployment-contract.md).
