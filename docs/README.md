# Documentation Index

This directory contains maintained product and engineering documentation. Planning,
private workflow material, and historical task lists are intentionally not published.

| Document | Purpose |
| --- | --- |
| `PROJECT_MAP.md` | Supported surfaces, installation model, and repository map. |
| `REPO_STRUCTURE.md` | Repository and output conventions. |
| `API.md` and `public_api.md` | Supported estimator and package APIs. |
| `architecture.md` and `architecture_contract.md` | Current implementation map and approved Phase 3 architecture contract. |
| `migration.md` | Compatibility and upgrade guidance for the current 0.12.4 source line. |
| `CONTRIBUTING.md` | Contribution workflow and local quality checks. |
| `deprecation_policy.md` | Canonical parameter names and alias policy. |
| `lm.md` | PSANN-LM usage, training, and benchmark guidance. |
| `PSANN_Results_Compendium.md` | Curated experiment results and interpretation notes. |
| `benchmarks/` | Reproducible benchmark inputs, promotion guidance, and checked-in summaries. |
| `examples/README.md` | Runnable example notes. |

## Logging directories

- Use `runs/hisso/` for local HISSO logs and `/content/hisso_logs/` on hosted notebooks.
- The HISSO logging CLI accepts `--output-dir`; run
  `python -m psann.scripts.hisso_log_run -h` for its current options.

## Documentation maintenance

- Keep public documents focused on supported behavior and reproducible guidance.
- Link new maintained documentation from this index.
- Use the issue tracker for proposed work and follow-up tasks rather than committing
  project plans or session notes to the repository.
