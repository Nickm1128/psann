# ADR 0003: Compatibility and Support Policy

- Status: Accepted
- Date: 2026-07-27
- Decision owner: Nickm1128
- Tracking issue: https://github.com/Nickm1128/psann/issues/2

## Context

The current package declares Python 3.9+, while current SHAP releases support Python
3.11+. A workplace platform also needs a bounded, testable dependency and operating
environment rather than an open-ended best-effort claim.

SHAP 0.49 was the final release supporting Python 3.9 and 3.10; SHAP 0.50 and later
require Python 3.11 or newer:
https://shap.readthedocs.io/en/stable/release_notes.html

## Decision

### Python policy

- The current `0.x` estimator line retains its documented Python 3.9+ compatibility
  until the workplace transition release.
- The stable workplace platform targets Python 3.11, 3.12, and 3.13.
- Python 3.9 and 3.10 are legacy-estimator-only environments and do not receive the
  stable SHAP/platform guarantee.
- Python 3.14 is evaluated after required NumPy, PyTorch, scikit-learn, and SHAP wheels
  are available and the full matrix passes.

The packaging metadata changes only during the implementation phase with migration
notes; this ADR does not itself change installed requirements.

### Dependency policy

The workplace release tests a floor plus a current environment:

| Dependency | Target floor | Test policy |
| --- | --- | --- |
| NumPy | 1.26 | Floor and latest supported 2.x |
| PyTorch | 2.4 | Floor and current stable line |
| scikit-learn | 1.4 | Floor and current pre-2.0 line |
| SHAP | 0.50 | Optional explain extra; floor and current stable line |

Dependency upper bounds are added only for known incompatibilities. A version is not
advertised as supported until the applicable package, task, artifact, and inference
tests pass.

### Platform tiers

- **Tier 1**: Linux x86_64 CPU, Linux x86_64 CUDA, and Windows x86_64 CPU.
- **Tier 2**: Windows CUDA and macOS arm64 CPU.
- **Experimental**: Apple MPS until scheduled training, artifact, inference, and
  explanation tests are reliable.

CPU correctness is required for all stable core capabilities. CUDA support requires a
documented driver/runtime matrix and recent scheduled evidence.

### Support-window policy

- CI tests every stable Python minor.
- The floor environment and a current environment are both blocking.
- Scheduled jobs may cover expensive accelerator and cross-version artifact tests.
- Dropping a Python minor, dependency floor, operating system, or device tier follows
  ADR 0005.

## Consequences

- The stable platform can use current SHAP without maintaining two explanation APIs.
- Existing Python 3.9/3.10 estimator users receive a documented transition path.
- Compatibility claims become tied to a reproducible matrix.
- Raising the workplace PyTorch floor reduces the artifact/export compatibility
  surface while leaving the current estimator line available during transition.

## Rejected alternatives

- Pinning the stable workplace platform to SHAP 0.49 would start the new platform on a
  legacy dependency line.
- Claiming all Python 3 versions supported by any dependency would create an
  untestable matrix.
- Treating every operating system and accelerator as Tier 1 would overstate current
  evidence.
