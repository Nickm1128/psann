# PSANN Cleanup Master To-Do

## Working Instructions for Future Sessions
- Begin each session by skimming this document; pick one unchecked item and note progress when you stop.
- Keep edits focused. Land enabling refactors separately from feature work and avoid bundling unrelated changes.
- Prefer shared utilities over copy/paste. If logic is duplicated, pause and extract a helper.
- When touching training or inference logic, run targeted tests (`python -m pytest tests/test_hisso_primary.py`, etc.) before and after changes.
- Keep edits ASCII unless a file already uses Unicode.
- Update the "Progress log" bullet under any task you touched with today's date, what changed, and upcoming blockers.

---

## Milestone 1 - Harden the HISSO Primary Pipeline
- [x] Codify reward/context configuration
  - [x] Expose a single `HISSOOptions` helper that normalises reward fn, context extractor, noise, and transforms.
  - [x] Ensure `hisso_evaluate_reward` mirrors training transforms (softmax/tanh/etc.) without relying on implicit hooks.
- [x] Complete estimator parity
  - [x] Confirm `ResPSANNRegressor` and `ResConvPSANNRegressor` route through the shared hooks for both supervised and HISSO fits.
  - [x] Add regression coverage for preserve-shape + HISSO (per-element off/on) to guard the conv hooks.
- [ ] Performance & stability sweep
  - [ ] Benchmark dense vs. conv HISSO on representative datasets; capture CPU/GPU wall time and reward trends.
  - [ ] Stress-test noise injection, warm starts, and short series (episode length > series length) to verify graceful degradation.
- [ ] Progress log:
  - 2025-10-14 - Introduced `HISSOOptions` to normalise reward/context/noise/transform handling, refactored evaluation helpers, extended estimator hooks (ResPSANN/ResConv) to share the options, and added regressions for preserve-shape HISSO flows (including explicit per-element guards). Next: move on to the performance & stability sweep benchmarking dense vs. conv HISSO.
  - 2025-10-19 - Restored `_make_flatten_fit_hooks` and retired predictive-extras demos; HISSO primary helpers now power all surviving examples.
  - 2025-10-14 (late session) - Landed the dense vs. conv HISSO benchmarking harness (CPU/GPU summaries) and stress tests for noise injection, warm starts, and short-series handling; patched the warm-start optimizer hook and verified `tests/test_hisso_primary.py`. Next: collect real dataset baselines and surface the benchmark findings in docs/CI.
  - 2025-10-14 (evening) - Extended the benchmarking harness with dataset/output selectors, ingested an AAPL open/close portfolio slice (`benchmarks/hisso_portfolio_prices.csv`), captured `docs/benchmarks/hisso_variants_portfolio_cpu.json`, and documented the baseline in `docs/benchmarks/hisso_variants.md`. Next: add GPU runs + CI wiring so regressions surface automatically.
  - 2025-10-14 (night) - Added `scripts.compare_hisso_benchmarks.py` and `.github/workflows/hisso-benchmark.yml` so CI replays the CPU portfolio sweep and flags drift against the stored snapshot; docs updated to mention the automation. Next: stage GPU baselines once hardware is available.
  - 2025-10-15 - Added unit coverage for the benchmark comparator and clarified that GPU baselines remain pending until hardware access is available; sticking with CPU runs in CI for now.

## Milestone 2 - Formalise Stateful & Streaming Behaviour
- [x] Design a `StateConfig` that captures rho/beta/init/detach semantics for PSANN blocks and downstream wrappers.
- [x] Unify `predict_sequence`, `predict_sequence_online`, and HISSO inference so they share state management and teacher forcing options.
- [x] Add bounded-state safeguards (norm clipping or scheduling) and surface warnings when state norms explode/vanish.
- [x] Build regression datasets (drift, shocks, regime switches) to validate both free-running and teacher-forced paths.
- [x] Document the recommended staged workflow (fit -> warm start -> streaming evaluation) once implementation settles.
- [ ] Progress log:
  - 2025-10-16 - Introduced `StateConfig` throughout the stack, unified `predict_sequence`/`predict_sequence_online`/HISSO rollouts on a shared streaming backend, added state saturation/collapse warnings, landed drift/shock synthetic datasets with regression tests, and refreshed docs to outline the fit -> warm start -> streaming workflow.

## Milestone 3 - Documentation & Packaging Refresh
- [ ] Rewrite README and high-level docs to reflect the primary-output-only API:
  - [ ] Remove references to predictive extras, growth schedules, and deprecated CLI switches.
  - [ ] Highlight the streamlined HISSO flow with updated code snippets.
- [ ] Update `docs/examples/` to showcase the current example set (21/26/27/etc.) and delete links to retired notebooks/scripts.
- [ ] Prepare migration guidance (`docs/migration.md`) that explains the extras removal, HISSO changes, and estimator signature updates.
- [ ] Audit packaging metadata (`pyproject.toml`, classifiers, optional extras) and ensure `psann.__init__` exports match the lean surface.
- [ ] Progress log:
  - 2025-10-14 - Began milestone 3 by rewriting the README/API docs for the primary-output pipeline, refreshing `docs/examples/` and `docs/migration.md`, renaming the HISSO allocation examples, exporting `HISSOOptions` at the top level, and updating packaging metadata/excludes. Next: continue scrubbing remaining docs (e.g., `TECHNICAL_DETAILS.md`, extras inventories) so no stale extras references linger.

## Milestone 4 - Quality, Tooling, and Release Prep
- [ ] Expand automated tests
  - [ ] Add HISSO smoke tests that cover CPU/GPU selection, warm starts, and reward evaluation without extras.
  - [ ] Ensure LSM + HISSO combinations (train/eval) are covered with and without `lsm_train`.
- [ ] Introduce lightweight benchmarks or CI checks to catch performance regressions in training loops.
- [ ] Run a full lint/type pass and capture remaining warnings for follow-up tickets.
- [ ] Draft release notes summarising breaking changes, new helpers, and removal of extras-centric modules.

---

## Backlog / Nice-To-Haves
- Explore a lightweight replacement for predictive extras (e.g., multi-target regression guidance) once the primary pipeline ships.
- Investigate exporting trained models (TorchScript/ONNX) now that the surface is leaner.
- Consider providing a small synthetic dataset + notebook demonstrating end-to-end HISSO usage for onboarding.
- Evaluate whether the stateful backlog should become its own milestone post-release.

---

## Completed (Historical Reference)
- Modularised estimator `fit` paths via `_fit_utils`, unifying argument normalisation, scaler prep, and shared training loops.
- Extracted WaveResNet/WaveEncoder components and stabilised LSM expanders for both dense and conv pipelines.
- Added gradient clipping, AMP toggles, and scheduler support to the training loop, with matching tests.
- Introduced reward registry helpers and hardened HISSO warm-start handling.
- Removed the extras framework, predictive-extras trainers, and dependent demos; HISSO now targets primary outputs only.

