# LSM Robustness TODO

## Extras Stability Backlog

- [x] Keep the HISSO extras expansion path dormant until long sequences are available so short rollouts stay on the base learner.
  - Gate the HISSO auto-expansion path in `PSANNRegressor.load` and `fit` (src/psann/sklearn.py:1882, src/psann/sklearn.py:640) so extras widen only when `hisso=True` and the observed window exceeds the episode length.
  - Add guards around HISSO warm-start caches to ensure `_extras_cache_` stays empty until a full episode rolls over (src/psann/hisso.py:150).
  - Extend regression coverage to exercise `hisso=True` + `preserve_shape=True` expansions once sequences cross the threshold (tests/test_supervised_extras.py).

- [x] Expose a staged extras growth policy (e.g. `extras_warm_start_epochs`) that freezes the extras head until the base loss plateaus, then expands and unfreezes it.
  - Promote an `extras_warm_start_epochs` setter that hydrates `ExtrasGrowthConfig.warm_start_epochs` (src/psann/extras.py:44) while keeping legacy `extras_growth` dict parsing backwards compatible.
  - Gate the extras head parameters in `PSANNRegressor.fit` until the base loss plateaus, wiring the toggle around the first `run_training_loop` invocation (src/psann/sklearn.py:1250, src/psann/training.py:60) and logging when the head unfreezes.
  - Backfill regression tests that freeze extras weights for the warm-start window and confirm gradients flow only after the plateau trigger (tests/test_supervised_extras.py).

- [x] Harden backend training loops to reinitialise optimiser state, schedulers, and gradient scalers whenever extras dimensions change so resumed fits remain stable.
  - After each `expand_extras_head` call, rebuild optimisers via `_make_optimizer` (src/psann/sklearn.py:471) and reset any cached schedulers before re-entering `run_training_loop`.
  - Drop and recreate the AMP scaler (or introduce `_amp_scaler`) when extras expand so mixed-precision runs do not reference stale parameter shapes.
  - Add resume-fit coverage where extras expand mid-training and verify loss stays finite versus diverging due to stale optimiser state (tests/test_supervised_extras.py).

- [x] Add invariants around extras tensor packing/unpacking in loaders, predictors, and checkpoint restores to prevent silent shape drift.
  - Code explicit shape assertions in `_prepare_supervised_extras_targets` (src/psann/sklearn.py:521), HISSO warm-start ingestion (src/psann/hisso.py:150), and predictive extras adapters (src/psann/augmented.py:113).
  - Validate checkpoint metadata before mutating modules so `extras_growth` widths align with persisted tensors (src/psann/sklearn.py:1882) and emit targeted errors when they do not.
  - Add negative-path tests that corrupt extras tensors/checkpoints and assert deterministic `ValueError` messages instead of silent cache reuse (tests/test_supervised_extras.py).

- [x] Add regression coverage for base-to-expanded extras fine-tuning, including failure cases where new extras columns destabilise base predictions.
  - `tests/test_supervised_extras.py:405` trains against `tests/data/extras_regression/stable_finetune.npz` to confirm primary predictions stay within tolerance after `expand_extras_head` resumes training.
  - `tests/test_supervised_extras.py:426` uses `tests/data/extras_regression/correlated_extras.npz` to surface controlled warnings and keep drift bounded when extras correlate with targets.
  - `tests/test_supervised_extras.py:452` exercises the HISSO warm-start path (`hisso=True`) so extras caches and predictive extras adapters remain aligned through expansion.

- [x] Document the staged extras workflow alongside optimiser reset caveats in docs/examples and API reference.
  - Add a walkthrough detailing warm-start freezes, plateau detection, and optimiser resets in `docs/API.md` and `docs/examples/README.md`.
  - Call out mixed-precision and scheduler reset caveats introduced above, linking to the new regression tests for context.
  - Provide upgrade notes that map legacy `extras` kwargs onto the staged workflow and highlight the new warnings surfaced during expansion.

- [x] Run a compatibility sweep to confirm legacy configs and checkpoints either map cleanly to the unified API or emit actionable warnings.
  - Enumerate historical knobs (`extras`, `extras_growth`, warm-start shortcuts) and feed them through `PSANNRegressor.set_params` to verify they hydrate `ExtrasGrowthConfig` correctly.
  - Load representative legacy checkpoints (tests/fixtures or archived runs) to ensure `auto_expand_on_load` either upgrades modules cleanly or surfaces migration warnings at `src/psann/sklearn.py:1882`.
  - Ensure the compatibility warnings point to remediation steps in the docs above, and capture their text in doctests so regressions fail loudly.

## HISSO Performance Backlog

- [x] Move episodic batch sampling onto torch tensors so contiguous windows are gathered without numpy round-trips, keeping data resident on GPU when available.
  - Preload the full training series to the target device and use `torch.unfold`/`as_strided` to draw (episodes_per_batch ��� episode_length) windows each epoch.
  - Reuse the episode tensor across epochs while still supporting input noise injection.

- [x] Keep the HISSO extras cache on-device and update it with vectorised tensor ops while regenerating Gaussian noise each epoch.
  - Store `extras_cache` as a torch buffer so supervision targets avoid host/device copies, and refresh the leading slice with `torch.randn_like` rather than reallocation.
  - Replace the per-episode python loop that writes back extras trajectories with batched slice assignments.

- [x] Reuse `PredictiveExtrasTrainer` and optimizer state across successive HISSO runs instead of rebuilding them for every invocation.
  - Allow callers to supply a preconstructed trainer or expose a context manager that preserves optimizer moments across fits/hyperparameter sweeps.

- [x] Replace the manual extras-initial-state gradient tweak with a formal parameter update so autograd stays on-device and step sizes are schedule-driven.
  - Treat the initial extras vector as a learnable buffer (or fold it into the model state) so the optimizer manages updates and scaling.

- [x] Add profiling coverage that records HISSO wall-clock time on CPU vs GPU after the optimisations above and guards against regressions.
  - Capture representative configs (e.g., 64-length episodes on Colab T4) in benchmark scripts and surface the results in CI logs or docs.
## Stateful Stability Backlog

- [ ] Formalise the current state update (rho/beta/max_abs/init/detach) into a documented dynamical system so we can reason about stability and convergence.
- [ ] Unify the estimator, sequence predictors, and online trainers around a `StateConfig` object instead of ad-hoc dicts, ensuring construction, cloning, and checkpoint reloads stay consistent.
- [ ] Rework `predict_sequence` / `predict_sequence_online` to share a single backend that handles detach semantics, teacher forcing, and gradient flow guarantees.
- [ ] Introduce bounded-state safeguards (normalisation/clipping schedules, optional learnable gain) and prove they keep the update contract satisfied under typical streaming lr values.
- [ ] Add diagnostics to surface exploding/vanishing state norms during fit and streaming evaluation, tying alerts back to config suggestions.
- [ ] Build regression suites with synthetic drift, shock, and regime-switch datasets to validate free vs. teacher-forced rollouts across resets.
- [ ] Document the recommended staged training workflow for stateful models, calling out when to enable streaming optimisers and how to checkpoint/reset safely.
- [ ] Run a compatibility pass over existing HISSO/stateful notebooks to flag behaviours that break once the new formalism lands.

## Completed

- [x] Generalise `expand_extras_head` for conv/preserve_shape/per_element regressors and add regression coverage so extras expansion stays stable across layouts (src/psann/extras.py:352, tests/test_supervised_extras.py:183, tests/test_supervised_extras.py:209).
- [x] Finalise a unified `extras_growth` API (constructor arg plus setter) and map existing extras flags onto it so estimators, helpers, and configs present a single surface area (src/psann/extras.py:35, src/psann/sklearn.py:148, tests/test_supervised_extras.py:125).
- [x] Ship an `expand_extras_head` helper that clones a fitted PSANNRegressor into a wider extras head while preserving trunk weights and optimiser schedules (src/psann/extras.py:239, src/psann/__init__.py:12, tests/test_supervised_extras.py:164).
- [x] Detect extras width mismatches when loading checkpoints and auto-trigger head expansion rather than erroring out (src/psann/sklearn.py:1876, tests/test_supervised_extras.py:179).

- [x] Align alias handling with estimator conventions to match warning-based shims instead of hard errors (src/psann/lsm.py:68, :147, :432, :499; reference behaviour in src/psann/sklearn.py:154, :156).
- [x] Wire the declared \batch_size parameter into the optimiser loop or remove it for parity with the revised regressors (src/psann/lsm.py:130).
- [x] Extend the expanders to accept both NumPy arrays and Torch tensors (and round-trip tensors when provided) to stay compatible with the new regressor fit/validate pathways (src/psann/lsm.py:225, :352, :365, :559, :627).
- [x] Add a thin \forward (plus \to/\train/eval) wrapper that delegates to the fitted LSM so expanders can plug directly into lsm= (src/psann/lsm.py:106, _resolve_lsm_module in src/psann/sklearn.py:415).
- [x] Provide a score_reconstruction helper for the conv expander to mirror the dense path and support diagnostics promised in the docs (src/psann/lsm.py:469).
- [x] Add regression coverage that exercises PSANNRegressor with both LSMExpander and LSMConv2dExpander under lsm_train=True/False and preserve_shape modes to keep _fit integration stable (\tests/).
- [x] Refresh API docs and examples once behaviour changes land so the LSM sections reflect canonical parameter naming, tensor support, and conv diagnostics (docs/API.md, docs/examples/README.md).





