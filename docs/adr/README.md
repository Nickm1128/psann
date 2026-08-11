# Architecture Decision Records

This directory contains accepted architecture and product decisions for the stable
PSANN workplace neural-network platform.

ADRs describe contracts and constraints. The phased implementation backlog lives in
`docs/backlog/workplace_nn_platform_todo.md`, and current supported public symbols
remain documented in `docs/public_api.md`.

| ADR | Status | Decision |
| --- | --- | --- |
| `0001-workplace-lifecycle-api.md` | Accepted | High-level create, train, export, load, and explain API |
| `0002-task-and-backbone-support.md` | Accepted | Stable tasks, backbones, activations, and extension tiers |
| `0003-compatibility-and-support-policy.md` | Accepted | Python, dependency, operating-system, and accelerator policy |
| `0004-artifact-and-deployment-contract.md` | Accepted | Safe native artifact, training checkpoint, export, and serving contract |
| `0005-legacy-deprecation-policy.md` | Accepted | Transition from legacy checkpoints and compatibility aliases |

## ADR lifecycle

- **Proposed**: open for review and not authoritative.
- **Accepted**: authoritative for implementation and release decisions.
- **Superseded**: retained for history and linked to its replacement.
- **Rejected**: considered but not adopted.

Changing an accepted contract requires a new ADR that links to and supersedes the old
record. Editing wording for clarity is allowed when it does not change the decision.
