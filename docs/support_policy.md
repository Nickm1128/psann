# Supported Versions

Status: Active

Last reviewed: 2026-08-11

## Runtime Support

The workplace development line supports Python 3.11, 3.12, and 3.13. Every claimed
Python minor must pass the fast CPU suite from an installed package in CI.

The dependency and platform tiers are maintained in
[`workplace_support_matrix.md`](workplace_support_matrix.md). A version, operating
system, accelerator, task, or export is not supported merely because it can be
installed; it must have the evidence required by that matrix.

Python 3.9 and 3.10 remain relevant only to prior estimator releases. Users requiring
those interpreters should pin the last compatible release and should not expect the
new workplace or SHAP APIs.

## Release Support

The selected release identity is documented in
[`release_identity.md`](release_identity.md). The public stable package is currently
`0.12.7`; `1.1.0rc1` is an unpublished release candidate.

For the `1.1.0rc1` candidate:

- native artifact format `1.0` begins its supported producer history with the 1.1
  line; the internal `0.13` through `0.16` phase labels were never published producer
  releases and are not compatibility claims;
- manifest format `0.9` has an explicit in-memory migration to `1.0`, covered as a
  synthetic schema test rather than historical producer evidence;
- an authentic checkpoint created by the public `0.12.7` wheel is retained with
  pinned provenance and continuously tested for trusted load, migration, and
  numerical parity;
- the frozen `workplace-v1` API is enforced by `docs/workplace_public_api.json`;
- no stable release is tagged until the release-candidate CPU, CUDA, security,
  container, package, migration, and warning gates pass for one commit.

The workflows that formerly aggregated release certification and supply-chain
evidence were [archived on 2026-08-11](archive/workflows/README.md). Therefore no
candidate currently satisfies the final condition above.

PSANN-LM remains Alpha and outside the stable workplace API commitment. Its `1.1`
distribution line requires `psann>=1.1.0rc1,<1.2`, validates that band at import time,
and reports its own installed distribution version. Coordinated candidate builds use
the same package version, but compatibility is governed by the declared range rather
than version aliasing.

Security support follows this explicit window:

- the latest `1.1.x` patch becomes the primary supported line when `1.1.0` reaches
  general availability;
- `0.12.7` remains the supported legacy stable line until 90 days after the `1.1.0`
  publication date;
- releases older than `0.12.7` receive migration guidance but no guaranteed security
  fixes;
- release candidates receive fixes during certification but are not production
  support claims.

Beginning with `1.1`:

- the current major and immediately previous major receive artifact read/migration
  support for the documented window;
- normal public deprecations receive at least two minor releases and 90 days of
  warning;
- removals occur in a major release unless a security boundary requires earlier
  action.

The artifact format remains `1.0`; that protocol version is independent of the
distribution version. Checkpoint and alias details are governed by
[`ADR 0005`](adr/0005-legacy-deprecation-policy.md) and
[`deprecation_policy.md`](deprecation_policy.md).

## Support Evidence

Blocking pull-request evidence:

- canonical Ruff, Black, and mypy checks;
- Python 3.11-3.13 CPU tests;
- 70% minimum core coverage, a 35% PSANN-LM Alpha floor, observational aggregate
  scripts coverage, and a separate 60% release-helper floor as defined in
  [`quality_policy.md`](quality_policy.md);
- repository and notebook hygiene;
- clean built-wheel smoke tests.

Scheduled evidence:

- CUDA forward/backward, resume, artifact, batched inference, fp16/bf16 AMP, compile,
  supported export, explanation, and memory observations;
- experimental MPS float32 observations that do not block releases;
- CPU performance comparisons where correctness blocks and noisy timing regressions
  alert by default;
- dependency/container vulnerability scans and source/image SBOM generation are
  required but currently absent; their former workflow is
  [archived](archive/workflows/README.md).

See [`releasing.md`](releasing.md) for the release checklist,
[`compatibility_evidence.md`](compatibility_evidence.md) for retained migration
fixtures and claim boundaries, and [`workplace_operations.md`](workplace_operations.md)
for the operational evidence contract.
