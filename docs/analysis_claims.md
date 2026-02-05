# Rationale Notes and Claims-to-Test

Sources reviewed: `TECHNICAL_DETAILS.md`, `docs/architecture.md`, `docs/wave_resnet.md`, `src/psann/activations.py`, `src/psann/layers/sine_residual.py`, `src/psann/models/wave_resnet.py`, `src/psann/nn_geo_sparse.py`, `src/psann/params.py`.

## Rationale Notes (short)
- SineParam activation exposes learnable amplitude, frequency, and decay per feature; this acts like an adaptive Fourier basis with damping that can represent periodic structure while remaining trainable. (`TECHNICAL_DETAILS.md`, `src/psann/activations.py`)
- Residual PSANN blocks use a learnable residual scale alpha, intended to stabilize deep stacks by keeping early Jacobians near identity. (`TECHNICAL_DETAILS.md`, `src/psann/nn.py`)
- WaveResNet residual sine blocks apply optional phase shift + FiLM context modulation to steer constructive/destructive interference; RMSNorm can stabilize deeper stacks without erasing phase. (`docs/wave_resnet.md`, `src/psann/layers/sine_residual.py`)
- GeoSparse uses structured sparse connectivity for parameter efficiency, with analytic param counts available for parity control. (`src/psann/nn_geo_sparse.py`, `src/psann/params.py`)
- The estimator pipeline standardizes scaling, context construction, and supervised/HISSO training hooks so variants share a consistent fit path. (`docs/architecture.md`, `TECHNICAL_DETAILS.md`)

## Claims-to-Test (tied to analysis sections)
1) H1 Generalization: SineParam-based PSANN/ResPSANN should match or exceed dense baselines on smooth seasonal forecasting (Jena) under compute parity.
   - Test: Jena multi-seed runs with matched params/time; compare RMSE/MAE/SMAPE/R2.
2) H2 Information usage: In multivariate forecasting, PSANN should rely more on history than exogenous covariates, consistent with adaptive frequency features.
   - Test: feature group ablations or permutation tests on Beijing/Jena; quantify delta-R2.
3) H3 Spectral/geometry: PSANN should exhibit lower Jacobian participation ratio (PR) on periodic data vs MLP, indicating fewer active modes.
   - Test: PR/NTK snapshots on Jena or synthetic periodic tasks across training.
4) H4 Robustness: PSANN/ResPSANN should be stable under drift/shock/missingness due to damped sine activations and residual scaling.
   - Test: drift/shock/regime synthetic suite with multi-seed volatility and NaN/grad-norm monitoring.
5) H5 Inductive bias limits: Attention-only or fully dense spines should underperform small conv spines on long-memory sequences (HAR/EAF), implying minimal temporal bias is necessary.
   - Test: HAR raw vs conv-spine variants with matched params; compare accuracy/F1 and overfit gaps.
6) Efficiency (GeoSparse): Mixed-activation GeoSparse should approximate dense performance with fewer params and different wall-clock profile.
   - Test: `benchmark_geo_sparse_vs_dense.py --task mixed --sparse-activation mixed` + sweep; compare metrics vs param/time.
