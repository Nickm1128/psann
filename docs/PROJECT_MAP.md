# Repository map

| Path | Purpose |
| --- | --- |
| `src/psann/estimators/` | Canonical regression task facade |
| `src/psann/architectures/` | Immutable core policies, validation, registry, shared components |
| `src/psann/preprocessing/` | LSM/custom preprocessing policies, construction, persistence |
| `src/psann/episodic/` | HISSO strategy and task orchestration |
| `src/psann/_sklearn/` | Internal estimator fit/inference/persistence implementation |
| `src/psann/scripts/` | Packaged core command-line utilities |
| `psannlm/` | Separate LM distribution and canonical CLI |
| `examples/`, `notebooks/`, `configs/` | Maintained consumers; see the consumer manifest |
| `scripts/`, `benchmarks/` | Source-checkout research and benchmarking tools |
| `tests/` | Runtime, compatibility, consumer, and distribution contracts |
| `docs/` | Task guides, references, and labeled historical reports |

[consumer_manifest.json](consumer_manifest.json) declares maintained public consumers, prerequisites, build boundaries, and workflow coverage. Generated data, checkpoints, caches, logs, and benchmark outputs are not package source. [Contributing](CONTRIBUTING.md) describes validation.
