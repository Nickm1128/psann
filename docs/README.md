# Documentation Index

Use this guide to locate the right document and to see which notes are still active versus historical context. Status values:

- **Active**: maintained reference material that matches the current codebase.
- **Backlog**: planning notes that feed directly into upcoming issues or refactors.
- **Archive**: historical context retained for posterity; do not treat as up to date.

| Document | Status | Notes |
| --- | --- | --- |
| `API.md` | Active | Public estimator surface reference, kept in sync with `src/psann/sklearn.py`. |
| `migration.md` | Active | Upgrade guidance and behavioural changes between releases. |
| `CONTRIBUTING.md` | Active | Contribution workflow, coding standards, and review expectations. |
| `examples/README.md` | Active | CPU runtimes and usage notes for scripts in `examples/`. |
| `benchmarks/hisso_variants.md` | Active | Benchmark description; accompanying JSON captures reproducible config. |
| `benchmarks/README.md` | Active | Data provenance, size, and regeneration instructions for HISSO benchmarks. |
| `PSANN_Results_Compendium.md` | Active | Curated experiment results and interpretation tips. |
| `diagnostics.md` | Active | Quick reference for feature quality diagnostics; see revision history for recent notation fixes. |
| `wave_resnet.md` | Active | Background and design rationale for the WaveResNet backbone. |
| `extras_removal_inventory.md` | Backlog | Source-of-truth inventory for removing the legacy extras stack (linked to `backlog/extras-removal.md`). |
| `backlog/docs-site-generator.md` | Backlog | Notes on evaluating MkDocs/Sphinx once the HISSO refactor settles. |
| `lsm_robustness_todo.md` | Archive | Historical HISSO/extras backlog; superseded by the extras removal plan. |
| `phase1_audit.md` | Archive | Snapshot of the naming audit prior to the cleanup work. Preserve for traceability. |
| `archive/codex_instructions.md` | Archive | Legacy Colab instructions; kept for historical context only. |

## Maintenance Notes

- The extras deprecation effort is tracked in `backlog/extras-removal.md`; update both files together when status changes.
- If a document moves to Archive status, add a banner at the top explaining why and where to find current guidance.
- When adding new documentation, link it here and mark the status so downstream readers know whether it is normative guidance or planning material.
