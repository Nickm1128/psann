# Migration Notes

Guidance for upgrading projects to the refactored training surface introduced after 0.9.19. Follow this checklist when updating downstream code so documentation and implementation stay aligned.

## What changed

- **0.10.5 housekeeping (2025-10-19)**  
  - Development tooling extras now install coverage and build so CI can publish coverage reports and validate wheels.  
  - HISSO integration tests are marked `slow` to unblock quick `pytest -m "not slow"` iterations while the refactor settles.  
  - GitHub Actions runs Ruff/Black across the full tree, captures coverage on Python 3.11, and uploads built artifacts.  
  - Benchmark data provenance moved into `benchmarks/README.md` with a helper downloader; legacy Colab instructions live under `docs/archive/`.

- **Primary-output pipeline** - predictive extras and growth schedules were removed. Constructors ignore legacy `extras_*` arguments and emit warnings so downstream projects can detect stale configuration.
- **Shared fit helpers** - all estimators route through `normalise_fit_args`, `prepare_inputs_and_scaler`, `build_model_from_hooks`, and `run_supervised_training`. Custom loops should import these helpers instead of copying logic from `PSANNRegressor.fit`.
- **HISSO options** - episodic runs resolve reward, transform, noise, and warm-start settings via `HISSOOptions`. The public API still uses familiar keyword arguments (`hisso_window`, `hisso_reward_fn`, etc.), but the resolved options are now stored for evaluation helpers.
- **Neutral terminology** - episodic configs standardise on `transition_penalty`. The legacy aliases (`transition_cost`, `trans_cost`) remain temporarily and trigger deprecation warnings.

## Helper replacement table

| Previous touchpoint                             | Updated helper / destination                         | Notes |
|-------------------------------------------------|------------------------------------------------------|-------|
| Manual dtype/validation handling inside `fit`   | `normalise_fit_args`                                 | Converts validation triplets to float32. |
| Ad-hoc scaler prep + flattening                 | `prepare_inputs_and_scaler`                          | Returns `PreparedInputState` with train tensors + metadata. |
| Variant-specific model construction             | `build_model_from_hooks` + `FitVariantHooks`         | Supply hooks instead of overriding the full `fit`. |
| Episodic adapters per estimator                 | `build_hisso_training_plan` via hooks                | Ensures conv/dense variants share the same HISSO flow. |
| Direct reward-function wiring                   | `register_reward_strategy` / `get_reward_strategy`   | Bundles reward functions and secondary metrics. |

## Primary-output workflow

- Remove extra heads: drop `extras`, `extras_growth`, and `extras_*` kwargs from estimator construction and calls to `fit`. Any remaining references will raise warnings so you can locate stale code.
- HISSO warm starts use `hisso_supervised={"y": targets}`. Provide `hisso_window`, `hisso_reward_fn`, and optional `hisso_context_extractor` as before; the helpers consolidate them into `HISSOOptions` and persist the configuration for evaluation utilities.
- If you previously inspected `_extras_cache_` or other extras-specific attributes, switch to the HISSO helpers (`hisso_infer_series`, `hisso_evaluate_reward`) or the shared prepared-input metadata.

## Upgrade checklist

1. Replace manual preprocessing with `normalise_fit_args` and `prepare_inputs_and_scaler` so shapes and scalers match estimator behaviour.
2. Remove any `extras_*` constructor arguments or `extras_targets` usage. Confirm your tests no longer expect extras outputs.
3. Swap bespoke reward wiring for registry lookups (`register_reward_strategy`, `get_reward_strategy`) and keep configs on the neutral naming (`transition_penalty`).
4. Update notebooks and scripts to mention the curated example set (`examples/21`, `26`, `27`, etc.) instead of the retired predictive extras demos.
5. Log progress in `CLEANUP_TODO.md` whenever you touch docs or code that affects the migration effort.

## Outstanding TODOs

- Stage GPU benchmark baselines once shared hardware becomes available so CI can compare CPU and CUDA runs side by side.
- Expand regression coverage around HISSO evaluation helpers to exercise both supervised warm starts and reward-only episodes.
- Publish CI guidance tied to the contributor workflow (ruff and pytest gates) after the documentation refresh completes.

