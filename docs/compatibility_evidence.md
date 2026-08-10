# Artifact Compatibility Evidence

Status: Active

Last reviewed: 2026-08-10

This document separates authentic producer evidence from synthetic schema tests. A
package-version string written into a current artifact is metadata; it does not prove
that an older release produced or can consume that artifact.

## Supported evidence

| Producer or schema | Evidence | Release claim |
| --- | --- | --- |
| Public PSANN `0.12.7` | Retained legacy checkpoint created by the exact public wheel, pinned wheel and fixture SHA256 values, producer predictions, and current trusted load/migration parity tests | Supported explicit-trust migration into the 1.1 native format |
| Native format `1.0` | Artifacts created and loaded by the 1.1 candidate in source and installed-wheel certification | Supported producer history begins with the 1.1 line |
| Manifest format `0.9` | Current native artifact with a deliberately rewritten format field, then inspection, in-memory migration, rewrite, and prediction checks | Schema migration behavior only; not historical producer evidence |
| Internal labels `0.13`-`0.16` | Repository-history and public-package audit found no immutable released producer sources or artifacts | No historical artifact-support claim; labels identify unpublished development phases only |

## Public 0.12.7 fixture provenance

The retained files are:

- [`psann-0.12.7-regressor.pt`](../tests/fixtures/legacy/psann-0.12.7-regressor.pt),
  SHA256
  `99d49317fce455b3b1c419b3ea02e8846c3a5a8c345c62daefd244c2832e897b`;
- [`psann-0.12.7-regressor.json`](../tests/fixtures/legacy/psann-0.12.7-regressor.json),
  containing the input rows, producer predictions, training configuration, runtime
  versions, and source-wheel provenance;
- [`README.md`](../tests/fixtures/legacy/README.md), documenting reproduction and the
  pickle security boundary.

The producer is the public
[`psann==0.12.7`](https://pypi.org/project/psann/0.12.7/) wheel
`psann-0.12.7-py3-none-any.whl`, SHA256
`43e6bc16a06a27b72e9073d1f80dbac70e07634df4dd01459ab949032997699b`.
The fixture builder verifies this hash before installing or importing the producer.
It then trains through `PSANNRegressor.fit`, saves through the public `save` method,
and verifies producer-side reload parity.

The same verified wheel produced
[`public_api_0_12_7.json`](public_api_0_12_7.json), which retains all 53 public
top-level exports plus the parameter order and required methods for six principal
estimators. The API gate verifies that inventory together with checkpoint migration.

Rebuild from the pinned public source with:

```bash
python tools/generate_legacy_fixture.py
```

For an approved offline wheel, use `--wheel PATH`; the same SHA256 check remains
mandatory.

## Candidate verification

Run the focused compatibility matrix with:

```bash
python -m pytest -q tests/test_model_artifacts.py \
  -k "synthetic_0_9 or synthetic_producer_version or public_0_12_7"
```

The public fixture test first verifies its retained digest. It proves that generic
loading rejects the pickle without explicit trust, trusted loading reproduces the
producer predictions, migration writes a valid native artifact, and the migrated
artifact preserves those predictions. The release-certification workflow runs the
same checks against the installed candidate wheel.

## Trust boundary

The `.pt` fixture is a Python pickle and may execute code while loading. Its use in
tests is an explicit trust decision based on pinned provenance and a verified digest.
This does not make arbitrary `.pt` files safe. Unknown checkpoints must remain
rejected or quarantined; native `.psann` artifacts are the deployment default.
