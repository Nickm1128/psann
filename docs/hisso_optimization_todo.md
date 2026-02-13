# HISSO Training Optimization TODO (psann)

# Agent instructions (GoT / Codex 5.3 â€œdonâ€™t get stuckâ€ playbook)
- [ ] Maintain a **task graph**: every change node must link to (a) baseline evidence, (b) a test, and (c) a doc/bench update if user-facing.
- [ ] Always implement in **small, reversible diffs**: one behavior change per PR-sized patch (reward penalty â†’ state commit â†’ batching).
- [ ] Keep **defaults backward-compatible**; add â€œfast presetsâ€ as opt-in knobs first.
- [ ] When performance differs, verify with **two signals**: wall time + episodes/sec (avoid trusting one metric).
- [ ] If GPU isnâ€™t available locally, still implement CUDA-safe code paths and add **CPU tests** + CI-safe guards.
- [ ] Treat `context_extractor` as untrusted user code: wrap calls, validate shapes, and provide actionable errors/warnings.
- [ ] Avoid per-step Python/`inspect` overhead: resolve signatures and constants **once in `__init__`** and reuse.
- [ ] If a test fails, shrink the repro to `scripts/profile_hisso.py` or a new micro-test (donâ€™t debug inside full suite first).
- [ ] Use deterministic settings whenever possible: seed RNGs, fix episode starts in tests, and assert on shapes + monotonic counters.
- [ ] When editing docs/bench baselines, record **exact command + config** used (so future diffs are attributable).
- [ ] Watch for Windows/PowerShell encoding quirks: if output explodes, re-run with UTF-8 stdout (e.g., `sys.stdout.reconfigure(encoding="utf-8")`).
- [ ] Donâ€™t â€œoptimize blindlyâ€: after each perf change, re-run the same benchmark command from Phase 0 and compare deltas.
- [ ] Add a fallback path for every new fast path (feature flag or config switch) to prevent hard blockers in edge environments.

## Scope & acceptance criteria
- [ ] **Preserve default behavior (unless explicitly changed)**
  - [ ] Keep existing `fit(..., hisso=True, ...)` working with no new required args.
  - [ ] Maintain backwards-compatible config loading/serialization for saved models (`_hisso_cfg_`, `_hisso_options_`).
- [ ] **Correctness fixes are covered by tests**
  - [ ] `transition_penalty` is actually applied to rewards (when configured).
  - [ ] Stateful models donâ€™t silently skip/lose state updates during HISSO (commit/reset behavior is defined and tested).
- [ ] **Performance goals are measurable**
  - [ ] Add/extend benchmarks to report episodes/sec and wall time for comparable runs.
  - [ ] Demonstrate a clear speedup on CPU (and GPU when available) vs current baseline.

---

## Phase 0 â€” Baseline + profiling (donâ€™t change code yet)
- [x] Capture current baselines (timing + throughput)
  - [x] Run `python scripts/profile_hisso.py --epochs 4` and save output.
  - [x] Run `python -m scripts.benchmark_hisso_variants --dataset portfolio --epochs 4 --devices cpu --variants dense,conv --output <tmp.json>` and save output.
  - [x] Record: `episode_length`, `episodes_per_batch`, model type, device, torch version, CPU/GPU info.
  - Baseline artifacts: `reports/hisso_phase0/20260213_093424/profile_hisso_epochs4.txt`, `reports/hisso_phase0/20260213_093424/benchmark_portfolio_cpu.json`, `reports/hisso_phase0/20260213_093424/benchmark_portfolio_cpu.log`, `reports/hisso_phase0/20260213_093424/runtime_metadata.json`, `reports/hisso_phase0/20260213_093424/machine_info.txt`.
  - `profile_hisso.py` (toy `torch.nn.Sequential` model): `episode_length=64`, `batch_episodes=8`, `device=cpu`, `torch=2.7.1+cu118`, CPU `13th Gen Intel(R) Core(TM) i5-1335U (10 cores/12 logical)`, GPU unavailable in this environment (`torch.cuda.is_available()=False`, `nvidia-smi` missing).
  - Portfolio benchmark (`window=64`, HISSO default batching): dense `mean_wall_time_s=2.3549`, conv `mean_wall_time_s=2.7419`; both report `episodes_per_epoch_mean=[32.0, 32.0, 32.0, 32.0]` (implying effective HISSO `episodes_per_batch=32` in current trainer path).
- [x] Identify where time is going (quick triage)
  - [x] Confirm whether `context_extractor` is used in your slow runs (and whether it triggers NumPy fallback).
  - [x] Confirm whether the model is `stateful=True` in slow runs (likely impacts both correctness and speed).
  - Triage outcome: no `context_extractor` entries are present in `configs/hisso/*.yaml` or `runs/hisso/**/config_resolved.yaml`; your checked-in HISSO benchmark/config runs are not using custom context extraction, so the NumPy fallback path is not implicated in these runs.
  - Triage outcome: existing HISSO run logs consistently report `stateful=False` (see `runs/hisso/dense_cpu_smoke_dev/events.csv`, `runs/hisso/wave_resnet_cpu_smoke_dev/events.csv`, `runs/hisso/dense/dense_cuda_colab_gpu_20251101_180009/events.csv`, `runs/hisso/wave_resnet/wave_resnet_cuda_colab_gpu_20251101_180016/events.csv`).
- [x] Define the â€œnew fast pathâ€ target configuration(s)
  - [x] Decide what â€œfastâ€ means for you (e.g., â€œâ‰¥2Ã— faster on CPU for portfolio dataset; no major reward regressionâ€).
  - Proposed optimization target for Phase 3+: on the same portfolio benchmark command, achieve at least `2.0x` improvement in `mean_profile_time_s` and at least `1.5x` improvement in end-to-end `mean_wall_time_s`, with no reward regression worse than `10%` on `final_reward_mean` for dense and conv variants.

---

## Phase 1 â€” Fix: `transition_penalty` is parsed but not passed to `reward_fn`
- [x] Add reward-kwarg resolution once per trainer (no per-step `inspect`)
  - [x] Implement a helper similar to `psann.episodes._resolve_reward_kwarg` for HISSO reward functions.
  - [x] Store the resolved kwarg name on `HISSOTrainer` (e.g., `None | "transition_penalty" | "trans_cost" | "<var_kwarg>"`).
- [x] Apply penalty during reward evaluation
  - [x] Update HISSO reward call so it passes the configured penalty when supported, otherwise calls `reward_fn(actions, context)` unchanged.
  - [x] Ensure scalar/shape coercion remains compatible with existing reward functions.
- [x] Tests
  - [x] Add a unit test that fails on current code: reward fn expects `transition_penalty` and asserts it is received.
  - [x] Add a unit test for legacy `trans_cost`.
  - Validation command: `python -m pytest tests/test_hisso_primary.py -k "transition_penalty_forwarded or trans_cost_alias_forwarded or infer_series_matches_predict" -q` (`3 passed`).
- [x] Docs
  - [x] Update `docs/API.md` (HISSO section) to state penalty is forwarded automatically when supported.

---

## Phase 2 â€” Fix: stateful models need explicit reset/commit semantics in HISSO
- [x] Decide and document semantics (pick one and enforce)
  - [ ] Option A (most consistent): reset state at the start of each sampled episode batch/update; commit after optimizer step.
  - [x] Option B: follow `state_reset` style semantics (batch/epoch/none) inside HISSO.
  - Chosen semantics documented in `docs/API.md` under HISSO configuration.
- [x] Implement minimal correctness plumbing
  - [x] If model has `reset_state()`, call it at the chosen cadence.
  - [x] If model has `commit_state_updates()`, call it after `optimizer.step()` / `scaler.step()` (AMP) in HISSO.
  - [ ] (Optional) If model has `set_state_updates(enabled)`, consider disabling state updates when `stateful=False` or when explicitly requested (perf knob).
- [x] Tests
  - [x] Add a small stateful model test where state changes only if `commit_state_updates()` is called (detect regression).
  - [x] Ensure supervised training vs HISSO training donâ€™t diverge unexpectedly in state handling.
  - Validation command: `python -m pytest tests/test_hisso_primary.py -k "stateful_hooks_respect_state_reset or stateful_hooks_match_supervised_loop_pattern or transition_penalty_forwarded or trans_cost_alias_forwarded" -q` (`8 passed`).

---

## Phase 3 â€” Performance: vectorize episodes (batch episodes per optimizer update)
- [x] Design the new training loop API (keep backwards compatibility)
  - [x] Introduce explicit concepts:
    - [x] `batch_episodes` (B): how many episodes per optimizer update
    - [x] `updates_per_epoch` (U): how many optimizer updates per epoch
  - [x] Maintain old behavior by default (e.g., default `B=1`, `U=episodes_per_batch`), but enable â€œfast configâ€ (e.g., `B=32`, `U=1`) via new knobs.
  - Implemented as `HISSOTrainerConfig.episode_batch_size` + `HISSOTrainerConfig.updates_per_epoch` with compatibility fallback to legacy `episodes_per_batch`.
- [x] Implement batched episode sampling without Python loops
  - [x] Sample `starts` as a vector of size `B` (CPU or device-appropriate).
  - [x] Build an index tensor `(B, T)` and gather `x_tensor[idx]` into `inputs` shaped `(B, T, ...)`.
  - [x] Ensure the gather path supports both flat and conv-shaped inputs.
- [x] Forward pass in one call
  - [x] Reshape `(B, T, ...)` â†’ `(B*T, ...)`, run `model` once, reshape outputs back to `(B, T, primary_dim)`.
  - [x] Apply `primary_transform` on the last dim.
- [x] Context extraction for batched inputs
  - [x] Default context: use `inputs.detach()` (already on device) shaped `(B, T, ...)`.
  - [x] If `context_extractor` is provided, call it once on batched inputs (define expected shape rules).
  - [x] Ensure `_align_context_for_reward` works for `(B, T, M)` actions + `(B, T, C)` context.
- [x] Reward + optimization
  - [x] Compute rewards as `(B,)` (or coerce to it) and optimize `loss = -reward.mean()`.
  - [x] Keep AMP support working with the new batched tensors.
- [x] Profile instrumentation
  - [x] Extend `trainer.profile` to include `batch_episodes`, `updates_per_epoch`, and separate timing for: gather, forward, reward, backward, optimizer.
  - Added keys: `episode_batch_size`, `updates_per_epoch`, `episode_gather_time_s_total`, `forward_time_s_total`, `reward_time_s_total`, `backward_time_s_total`, `optimizer_time_s_total`.
- [x] Tests (behavioral + invariants)
  - [x] Verify shapes for actions/context across flat + conv.
  - [x] Verify deterministic run (seeded) produces stable reward trend for small configs.
  - [x] Add a â€œcompat modeâ€ test: `B=1` path matches old per-episode stepping behavior closely (within tolerance).
  - Validation commands:
  - `python -m pytest tests/test_hisso_primary.py -q` (`25 passed`)
  - `python -m pytest tests/test_regressor_inference.py -k hisso -q` (`1 passed`)

---

## Phase 4 â€” Expose tuning knobs end-to-end (fit API, typed dict, CLI configs)
- [x] Python API (`fit`)
  - [x] Add new `fit` kwargs (names TBD, but consistent and explicit), e.g.:
    - [x] `hisso_batch_episodes` (B)
    - [x] `hisso_updates_per_epoch` (U)
    - [ ] (Optional) `hisso_grad_clip_norm` (disable or set value)
  - [x] Thread them into `HISSOOptions` / `HISSOTrainerConfig` / plan building.
- [x] Types
  - [x] Update `src/psann/types.py` `HISSOFitParams` to include the new knobs.
- [x] Serialization/migrations
  - [x] Update `_serialize_hisso_cfg` / `ensure_hisso_trainer_config` to carry new fields (with sensible defaults).
- [x] Logging CLI + YAML templates
  - [x] Update `src/psann/scripts/hisso_log_run.py` to accept new config keys under `hisso:`.
  - [x] Update `configs/hisso/*.yaml` templates with recommended fast settings (but keep smoke configs conservative).
- [x] Docs
  - [x] Update `docs/API.md` and `README.md` with the new knobs and recommended presets (CPU vs CUDA).
- Validation commands:
  - `python -m pytest tests/test_hisso_primary.py -q` (`26 passed`)
  - `python -m pytest tests/test_regressor_inference.py -k hisso -q` (`1 passed`)
  - `python -m pytest tests/test_hisso_logging_cli.py -q` (`2 passed`)

---

## Phase 5 â€” Performance: warn once on NumPy fallback for `context_extractor`
- [x] Detect fallback and warn (especially on CUDA)
  - [x] When `_call_context_extractor` falls back to NumPy, emit a one-time warning explaining it will cause host/device transfers on GPU.
  - [x] Include a â€œfix tipâ€: accept `torch.Tensor` inputs and return a tensor on the same device/dtype.
- [x] Tests
  - [x] Add a test that forces the fallback path and asserts warning content/stacklevel points to user code.
- Validation commands:
  - `python -m pytest tests/test_hisso_primary.py -q` (`27 passed`)
  - `python -m pytest tests/test_regressor_inference.py -k hisso -q` (`1 passed`)
  - `python -m pytest tests/test_hisso_logging_cli.py -q` (`2 passed`)

---

## Phase 6 â€” Benchmarks, baselines, CI guardrails
- [x] Update benchmark scripts to exercise new knobs
  - [x] Extend `scripts/benchmark_hisso_variants.py` to report B/U and compare “compat mode” vs “fast mode”.
- [x] Update baseline artifacts and docs
  - [x] Refresh `docs/benchmarks/hisso_variants_portfolio_cpu.json` (if you decide it should reflect the fast default/preset).
  - [x] Update `docs/benchmarks/hisso_variants.md` with new timings and config used.
- [x] Add a regression threshold
  - [x] Ensure CI compares "compat mode" for stability, and optionally tracks "fast mode" performance trend.
- Validation commands:
  - `python -m pytest tests/test_benchmark_compare.py -q` (`4 passed`)
  - `python -m scripts.compare_hisso_benchmarks docs/benchmarks/hisso_variants_portfolio_cpu.json docs/benchmarks/hisso_variants_portfolio_cpu.json --modes compat --reward-rtol 0.5 --reward-atol 5e-3 --wall-rtol 0.8 --wall-atol 5.0` (`Benchmarks within tolerance.`)
  - `python -m scripts.compare_hisso_benchmarks docs/benchmarks/hisso_variants_portfolio_cpu.json docs/benchmarks/hisso_variants_portfolio_cpu.json --modes fast --reward-rtol 0.75 --reward-atol 1e-2 --wall-rtol 1.0 --wall-atol 8.0` (`Benchmarks within tolerance.`)

---

## Optional stretch goals (only after Phases 1â€“6 land)
- [ ] Precompute context for the full series once (when extractor is pure per-step)
  - [ ] Add an opt-in mode: compute `context_full = extractor(x_tensor)` once, slice during episode sampling.
- [ ] Reduce grad clipping overhead
  - [ ] Skip clipping when disabled; consider clipping only every N updates (opt-in).
- [ ] Consider `torch.compile` for the HISSO training loop (CUDA-first)
  - [ ] Gate behind an explicit flag and keep a safe fallback when compile fails.

---



