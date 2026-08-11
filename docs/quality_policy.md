# Scoped Quality and Coverage Policy

Status: Active

Last reviewed: 2026-08-10

PSANN publishes separate coverage reports because the stable core, experimental
language-model distribution, and operational scripts have different support
contracts. `python tools/run_coverage.py` runs the fast suite once and enforces these
branch-aware floors:

| Scope | Blocking floor | Rationale and next target |
| --- | ---: | --- |
| `src/psann` | 70% | Stable package floor; increases require sustained cross-platform results rather than excluding difficult code. |
| `psannlm` | 35% | PSANN-LM remains Alpha. The floor prevents regression below the currently exercised model/config/tokenizer/trainer paths; promotion to Beta requires at least 50% plus direct CLI, SFT, streaming-data, and trainer failure-path coverage. |
| all `scripts` | Observational | The directory mixes hardware launchers, external-data benchmarks, and one-off operational CLIs whose aggregate execution is not a meaningful package-quality gate. The XML report remains visible for prioritization. |
| `scripts/release.py` | 60% | Release-critical exception to the aggregate scripts policy. Focused tests cover version parsing, dry runs, clean-tree/version/changelog/tag/PyPI/LM compatibility preflights, artifact identity, upload confirmation, and local gate sequencing. |

The floors are release gates, not quality ceilings. A line may remain uncovered only
because its behavior requires unavailable hardware or external systems, not because
coverage configuration silently omits it. New release-critical scripts must receive
a focused report and blocking threshold before they participate in publication.

PSANN-LM intentionally remains classified Alpha while its user-facing CLI, SFT,
dataset/streaming, and trainer failure paths are below the Beta target. Its metadata,
documentation, and support matrix must not imply workplace-stable support before that
target and the installed-wheel compatibility checks pass together.
