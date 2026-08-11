# ADR 0005: Legacy Deprecation Policy

- Status: Accepted
- Date: 2026-07-27
- Decision owner: Nickm1128
- Tracking issue: https://github.com/Nickm1128/psann/issues/2

## Context

PSANN must transition from whole-object `.pt` estimator checkpoints and legacy
parameter aliases without stranding existing users or allowing unsafe files to load
implicitly.

## Decision

### Minimum warning period

Normal public deprecations receive:

- at least two published minor releases;
- at least 90 days of documented warning;
- migration documentation and regression tests;
- removal only in a major release.

Security-critical behavior may be disabled earlier, but the release notes must state
the reason and provide the safest available migration path.

### Legacy whole-object checkpoint timeline

- **1.1 release-candidate cycle**: introduce the safe `.psann` artifact, `export`,
  `load_model`, and a migration command. Existing class-specific `.save()` / `.load()`
  continue but emit a security and deprecation warning when using the legacy format.
- **1.1 GA**: safe artifacts are the default workplace persistence path. Generic loading
  rejects legacy checkpoints. Legacy loading requires an explicit
  `trusted_legacy_checkpoint=True` opt-in and remains class-specific.
- **1.x**: retain the explicit trusted migration path and test fixtures, but do not add
  capabilities to the legacy format.
- **2.0**: remove legacy whole-object loading from the core package. A standalone
  migration tool may remain for trusted environments.

The migration tool must never imply that an untrusted legacy checkpoint is safe. It
loads only after explicit trust confirmation and writes a validated new artifact.

### Parameter and structured-configuration timeline

- Canonical estimator names remain those in `docs/deprecation_policy.md`.
- Existing estimator aliases remain accepted with warnings through 1.x.
- New `ModelSpec` and related structured configuration accept canonical names only.
- Artifact migrations normalize legacy names into canonical fields.
- Earliest estimator-alias removal is 2.0 after usage, docs, and serialization audits.

### User-defined callables and custom objects

- Stable artifacts store registered identifiers and JSON-safe parameters, not
  callables.
- Existing local workflows using custom losses/scalers may continue in-process.
- Persistence of a custom component requires explicit registry support.
- Legacy custom-object checkpoints are migrated only in an environment where their
  code is installed and trusted.

## Consequences

- Existing estimator users have a concrete multi-release transition.
- New workplace APIs start with canonical configuration and a safe persistence model.
- The 1.1 release can be workplace-ready without pretending old pickle files are safe.
- Removal is delayed until 2.0, reducing pressure to combine API, task, and persistence
  migration in one release.

## Rejected alternatives

- Removing legacy checkpoints immediately would break current users without a
  migration path.
- Silently loading legacy files in the generic loader would defeat the new trust
  boundary.
- Carrying aliases into new structured specifications would make the new artifact
  schema harder to stabilize.
