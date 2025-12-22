# Research Findings and Next Steps

This document summarizes what we learned from the current sklearn-style regressors and the latest ablation sweeps, plus a concrete set of follow-ups to investigate.

## What We Built (Relevant Pieces)

- **`ResPSANNRegressor`**: residual PSANN MLP baseline (strong on tabular).
- **`SGRPSANNRegressor`**: sequence-first regressor with:
  - **per-channel phase shifts** (`PhaseSineParam`) and
  - **lightweight spectral gating** (`SpectralGate1D`) over the inferred sequence axis,
  - pooling via `pool="last"|"mean"`.
  - Designed for inputs shaped `(N, T, F)`; does not support `preserve_shape=True`.
- **`WaveResNetRegressor`**: sine residual backbone with optional context modulation (FiLM + phase shift), plus:
  - **new optional spectral gating** for sequence-shaped inputs via `use_spectral_gate=True`.
  - Gating is applied over `(N, T, F)` (dense) and 1D conv-stem tokens (preserve-shape 1D only).

## Experiments Run (Latest Reference Sweep)

Benchmark harness: `scripts/benchmark_regressor_ablations.py`

Run directory: `reports/ablations/20251222_104953_wrn_spectral_compare/manifest.json:1`

- Device: CPU
- Seeds: 0, 1
- Epochs: 25
- Batch size: 64
- LR: 1e-3
- Datasets:
  - `tabular_sine`, `tabular_shifted`, `classification_clusters`, `ts_periodic`, `ts_regime_switch`
- Models:
  - ResPSANN: `res_base`, `res_drop_path`, `res_no_norm`
  - WaveResNet: `wrn_base`, `wrn_no_phase`, `wrn_no_film`
  - WaveResNet + spectral gate: `wrn_spec_gate_rfft`, `wrn_spec_gate_feats`
  - SGR-PSANN: `sgr_base`, `sgr_no_gate`, `sgr_fourier_feats`, `sgr_no_phase`

Raw results:
- `reports/ablations/20251222_104953_wrn_spectral_compare/results.jsonl:1`
- `reports/ablations/20251222_104953_wrn_spectral_compare/summary.csv:1`

## Key Findings

### 1) Tabular regression: `ResPSANNRegressor` is the best baseline in this sweep

- **`tabular_shifted`** (sMAPE, lower is better): `res_no_norm` wins (≈0.437), while WaveResNet is ≈0.759 and SGR is ≈0.845.
- **`tabular_sine`** (sMAPE): `res_base` wins (≈0.689), while WaveResNet is ≈1.012 and SGR is ≈1.320.

Interpretation:
- The residual MLP inductive bias is a strong fit for these tabular targets under the shared optimizer/epochs settings.
- SGR’s sequence machinery is wasted (or actively unhelpful) on flat/tabular tasks.
- WaveResNet’s sine-residual backbone appears under-tuned for these particular tabular targets at the chosen hyperparameters (could be improved with model-/task-specific tuning).

### 2) Sequence regression: `SGRPSANNRegressor` variants are consistently strongest

- **`ts_periodic`**: best is `sgr_no_phase` (sMAPE ≈0.157), then `sgr_base` (≈0.159). ResPSANN is competitive (≈0.167). WaveResNet is clearly behind (≈0.315).
- **`ts_regime_switch`**: best is `sgr_fourier_feats` (sMAPE ≈0.033), with other SGR variants and ResPSANN close behind (≈0.034–0.035). WaveResNet is behind (≈0.063).

Interpretation:
- The **spectral controller** in SGR (gate + phase) is a strong inductive bias for these synthetic time series, especially under distribution shift/regime dynamics.
- On CPU, SGR is **slower per run** than ResPSANN/WaveResNet at these sizes (FFT/Fourier-feature work + sequence ops), but accuracy gains can justify it.

### 3) WaveResNet FiLM/phase ablations did not matter in these runs (expected)

In this benchmark harness we did not pass explicit `context` to WaveResNet. As a result:
- `wrn_base`, `wrn_no_phase`, and `wrn_no_film` are effectively the same model behaviorally on these datasets.

Implication:
- To evaluate FiLM/phase knobs, we should add **context-aware datasets** (or pass context during `.fit`) where the context pathway is exercised.

### 4) Weighted spectral gating added to WaveResNet: small win on nonstationary TS, neutral-to-negative on periodic TS

The new WaveResNet spectral-gate variants only activate on sequence-shaped inputs `(N, T, F)`; on tabular/classification they match `wrn_base` exactly.

On the two time series datasets:
- **`ts_periodic`**: `wrn_spec_gate_rfft` is worse (sMAPE ≈0.354 vs ≈0.315 base); `wrn_spec_gate_feats` is roughly neutral (≈0.312).
- **`ts_regime_switch`**: both improve over base; `wrn_spec_gate_feats` improves the most (sMAPE ≈0.055 vs ≈0.063 base).

Cost:
- Parameter increase is tiny (+~100–136 params in these tasks), but CPU wall-time increases (~+1–2s/run here).

Interpretation / hypothesis:
- Applying a gate directly on **raw per-step features** can be disruptive when initialized “half-on” (`gate_init=0` → sigmoid ≈0.5). Periodic tasks are sensitive to phase/amplitude; an overly-active gate can degrade performance.
- The regime-switch dataset benefits from a weak frequency-selection prior, but the current placement/initialization likely isn’t optimal yet.

## What We Should Look Into Next (Prioritized)

### A) Make WaveResNet spectral gating “start off” and tune its strength

Concrete experiments:
- Try `gate_init=-4` (sigmoid ≈0.018) so the spectral branch starts near zero.
- Sweep `gate_strength` in `{0.05, 0.1, 0.3, 1.0}` and optionally schedule it upward.
- Sweep `k_fft` in `{32, 64, 128}` (latency vs. selectivity) and compare `gate_type="rfft"` vs `"fourier_features"`.

Expected outcome:
- Reduce regressions on `ts_periodic` while preserving the gains on `ts_regime_switch`.

### B) Move WaveResNet spectral gating deeper (operate on embeddings, not raw inputs)

Instead of gating `x` at the input token level, gate:
- after the stem projection (feature space closer to what WaveResNet actually uses), or
- between WaveResNet blocks (multi-stage spectral steering).

Rationale:
- The gate is trying to be a controller; it should act on a representation with enough capacity/structure to be controllable.

### C) Exercise and measure the WaveResNet context path (FiLM + phase shift)

Add a dataset variant where:
- `context` is explicitly passed to `.fit`/`.predict`, and targets depend on context (regime label, exogenous driver, etc.).

This will let us meaningfully evaluate:
- `use_film` on/off
- `use_phase_shift` on/off
- whether context improves robustness under shift.

### D) “Best of all models” in one estimator: router/ensemble strategy

Two pragmatic approaches:
- **Auto-estimator (routing by shape/task)**:
  - If input is `(N, T, F)` and `T>1`, default to SGR (or SGR + tuned WaveResNet gate).
  - If input is tabular `(N, F)`, default to ResPSANN.
- **Small ensemble / stacking**:
  - Train `{ResPSANN, SGR, WaveResNet(+gate)}` and combine predictions via learned linear blender on a validation split.

This targets what we observed empirically:
- ResPSANN is strong on tabular.
- SGR is strong on sequence/time-series.
- WaveResNet is currently weaker in these sweeps but has unique knobs (context, progressive depth, w0 schedules) and may contribute complementary error patterns.

### E) Add frequency interpretability and “mask diagnostics”

For spectral-gated models (SGR + WaveResNet gate), log:
- learned `sigmoid(mask)` statistics (mean/max per channel),
- optional heatmaps over frequency bins for a few runs,
- correlation of mask peaks with known cycles (24h, 7d, etc. for periodic tasks).

This turns the gate into a measurable hypothesis rather than a black box.

### F) Benchmark hardening and fairness

Improvements to increase confidence:
- More seeds (e.g., 5) and confidence intervals.
- Longer training for WaveResNet (it may need more epochs or different LR to show its best).
- Match model capacity across cores (params/MACs), not just hidden_units.
- Add time-series datasets with:
  - varying window lengths,
  - nonstationary variance,
  - phase shifts,
  - multi-step prediction heads (instead of single-step next value).

## Quick Actions (Suggested Immediate Runs)

1. Rerun WaveResNet spectral gate with conservative init/strength:
   - `gate_init=-4`, `gate_strength=0.1`, `k_fft=64`, `gate_type ∈ {rfft, fourier_features}`
2. Add a context-aware dataset and rerun `wrn_*` ablations with context passed.
3. If the above works, consider a simple `AutoPSANNRegressor` router to capture “tabular vs. sequence” best defaults in one estimator.

