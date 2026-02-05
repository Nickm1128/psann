**PSANN Results Compendium**

- Purpose: One-stop reference of datasets, methods, configurations, and non-visual results collected so far to accelerate paper writing and reproducibility.
- Scope: Compiles light-probe runs, prior outputs under `outputs/`, the experiment plan, and environment details.

**Environment**
- Python: 3.11.9 (Windows x64)
- Torch: 2.7.1+cu118
- NumPy: 1.26.4
- PSANN: 0.10.3
- Device: CPU (auto selection in scripts supports CUDA if available)

**Datasets**
- Jena Climate 2009–2016
  - Source CSV: resolved under `datasets/Jena Climate 2009-2016/jena_climate_2009_2016.csv` (auto-downloaded if absent).
  - Windowing: context 72 steps (12 hours at 10-minute cadence), horizon 36 steps (6 hours).
  - Shapes (train/val/test after windowing):
    - X: (12022, 72, 14), (2575, 72, 14), (2575, 72, 14)
    - y: (12022, 36), (2575, 36), (2575, 36)
- Beijing Multi-Site Air Quality
  - Station file: `datasets/Beijing Air Quality/PRSA_Data_Guanyuan_20130301-20170228.csv`
  - Windowing: context 24 hours, horizon 6 hours.
  - Shapes (train/val/test after windowing):
    - X: (1996, 24, 16), (427, 24, 16), (427, 24, 16)
    - y: (1996, 6), (427, 6), (427, 6)
- Industrial Data from the Electric Arc Furnace (EAF)
  - CSV: `datasets/Industrial Data from the Electric Arc Furnace/eaf_temp.csv`
  - Lite loader: context 16, horizon 1; selects top heats by length (falls back when no heat reaches 120 rows).
  - Shapes (train/val/test after windowing):
    - X: (97, 16, 2), (20, 16, 2), (20, 16, 2)
    - y: (97, 1), (20, 1), (20, 1)
- Additional (planned, not executed here): HAR Smartphone, Rossmann Store Sales.

### Archived datasets (extras)
- Predictive “extras” datasets, heads, and growth-schedule mixes used in earlier demos are archived. New results and sweeps cover the primary-output pipeline only.
- See `docs/backlog/extras-removal.md` and `docs/README.md` for migration notes and historical references.

**Methods**
- PSANN Conv Spine (`scripts/run_light_probes.py:PSANNConvSpine`)
  - `PSANNConv1dNet` temporal backbone with small strided Conv1d; global temporal aggregator (last/mean); linear head to horizon.
- MLP Regressor (`scripts/run_light_probes.py:MLPRegressor`)
  - Flattened input → [Linear, ReLU]xdepth → Linear(out_dim).
- Training
  - Optimizer: Adam(lr=1e-3); loss: MSE.
  - Batch size: 256 (default in light runner); epochs per CLI.
  - Seeds: configurable list (e.g., 7, 8).
  - Compute parity: parameter counts within each method family kept in a small range; equal epochs and batching per task under fixed wall-time budgets in notebooks; script focuses on matched epochs.
- Torch Dynamo Compatibility
  - A minimal shim installs no-op `torch._dynamo.disable` and `torch._dynamo.graph_break` for Torch builds where these are not present.

**Light-Probe Results (Script)**
- Command
  - `python scripts/run_light_probes.py --epochs 20 --seeds 7 8`
  - Writes `colab_results_light/metrics.csv`
- Raw CSV (verbatim)
```
(task,model,seed,params,epochs,val_loss,steps,train_size,rmse,mae,r2)
jena_light,psann_conv,7,19380,20,0.10169322788715363,940,12022,0.2593061029911041,0.19540846347808838,0.7128593117148211
jena_light,mlp,7,71076,20,0.09296286851167679,940,12022,0.3329276740550995,0.24463370442390442,0.5268910944378138
jena_light,psann_conv,8,19380,20,0.09003312140703201,940,12022,0.23122230172157288,0.16664689779281616,0.7717180970148231
jena_light,mlp,8,71076,20,0.13780806958675385,940,12022,0.4748856723308563,0.3576308488845825,0.036958438603300295
beijing_light,psann_conv,7,30662,20,0.31544986367225647,160,1996,0.4303075671195984,0.3199010491371155,0.5641837429897502
beijing_light,mlp,7,46854,20,0.8222692608833313,160,1996,0.6205906867980957,0.44732412695884705,0.09351441841996426
beijing_light,psann_conv,8,30662,20,0.306238055229187,160,1996,0.43655648827552795,0.3085808753967285,0.5514335925320024
beijing_light,mlp,8,46854,20,0.7648752927780151,160,1996,0.690960705280304,0.5018033385276794,-0.12369183987220524
eaf_temp_lite,psann_conv,7,4609,20,1.127816081047058,20,97,1.509945273399353,0.745273768901825,-0.10227529533011404
eaf_temp_lite,mlp,7,3985,20,1.287597894668579,20,97,1.462648630142212,0.663995087146759,-0.03430273561524788
eaf_temp_lite,psann_conv,8,4609,20,1.1749560832977295,20,97,1.501381754875183,0.7333859205245972,-0.0898078500418491
eaf_temp_lite,mlp,8,3985,20,1.2586772441864014,20,97,1.4548237323760986,0.729034960269928,-0.023265637331702615
```
- Aggregated (mean±std across seeds)
  - jena_light
    - psann_conv: rmse 0.2453±0.0199, mae 0.1810±0.0199, r2 0.7423±0.0419
    - mlp: rmse 0.4039±0.1004, mae 0.3011±0.0815, r2 0.2819±0.3465
  - beijing_light
    - psann_conv: rmse 0.4334±0.0044, mae 0.3142±0.0057, r2 0.5578±0.0090
    - mlp: rmse 0.6558±0.0498, mae 0.4746±0.0536, r2 -0.0151±0.2176
  - eaf_temp_lite
    - psann_conv: rmse 1.5057±0.0061, mae 0.7393±0.0060, r2 -0.0960±0.0088
    - mlp: rmse 1.4587±0.0055, mae 0.6965±0.046, r2 -0.0288±0.0065

Notes
- EAF lite split is tiny and noisy; negative R2 indicates limited predictability at this granularity; the loader already falls back to top heats when none meet the 120-row minimum.
- For Jena and Beijing, PSANN+Conv spine consistently outperforms MLP under the same epoch budget.

**GPU sweep status (Colab first, runpod fallback)**
- Tooling: primary path remains the Colab notebook `notebooks/HISSO_Logging_GPU_Run.ipynb`, which installs the released wheel, stages configs/data under `/content/hisso_*`, and runs `python -m psann.scripts.hisso_log_run --config <yaml> --output-dir <dir> --run-name <tag> --device cuda:0`. Same invocation ports to runpod for longer CUDA windows.
- CPU baselines (2025-11-01, seed 42):
  - Dense: `runs/hisso/dense_cpu_smoke_dev/prep_20251101/metrics.json` - duration 31.1 s, reward_mean -0.111, throughput 777 eps/s, Sharpe -5.9e3.
  - WaveResNet: `runs/hisso/wave_resnet_cpu_smoke_dev/prep_20251101/metrics.json` - duration 3.45 s, reward_mean -0.127, throughput 229 eps/s, Sharpe -2.21.
- Colab CUDA runs (2025-11-01; dense seed 7, wave seed 11; mixed precision float16):
  - Dense: `runs/hisso/dense/dense_cuda_colab_gpu_20251101_180009/metrics.json` - duration 2.68 s, throughput 203 eps/s, train/val/test loss 0.245 / 0.304 / 0.231, best_epoch 4 of 8, reward_mean -0.111 (std 4.7e-08), transition_penalty 0.0, turnover 1.15e-4. Wall time shrinks 11.6x relative to the CPU prep despite launch overhead on the tiny synthetic dataset.
  - WaveResNet: `runs/hisso/wave_resnet/wave_resnet_cuda_colab_gpu_20251101_180016/metrics.json` - duration 3.34 s, throughput 161 eps/s, train/val/test loss 1.435 / 1.402 / 1.569, best_epoch 10 of 10, reward_mean -0.182 (std 0.068), transition_penalty 0.01, turnover 2.65. Reward variance widens alongside the higher turnover flagged in the CPU baseline.
  - Observations: dense throughput trails the CPU smoke because the workload is launch bound, but both CUDA runs cut wall time materially while keeping reward means aligned with their CPU counterparts.
- Capture: artifacts synced locally (`metrics.json`, `events.csv`, `config_resolved.yaml`, `checkpoints/best.pt`). The notebook now logs device, throughput, reward, and loss summaries; CUDA memory was not exposed in Colab, so add `torch.cuda.max_memory_allocated()` probes during the runpod sweep if contention becomes a concern.
- Runpod CUDA run (2025-11-02; NVIDIA L4; mixed precision float16):
  - WaveResNet (config `configs/hisso/wave_resnet_small.yaml`): `runs/hisso/wave_resnet_cuda_runpod_20251102_212855/` — duration ~18.37 s, throughput ~113.07 eps/s; best_epoch 56; train/val/test loss 0.722 / 0.864 / 0.835; reward_mean −0.114 (std 0.0103); turnover 3.18; Sharpe −1.87; AMP float16.
  - WaveResNet (config `configs/hisso/wave_resnet_small.yaml`): `runs/hisso/wave_resnet_cuda_runpod_20251102_153117/` — duration 19.41 s over 1920 episodes (~107.3 eps/s), best_epoch 17, train/val/test loss 0.621 / 0.755 / 0.670, reward_mean -0.114 (std 0.010), turnover 2.69.

### GPU Sweep Summary Table

Short caption: Runpod L4 WaveResNet-small CUDA runs; see notebooks/HISSO_Logging_GPU_Run.ipynb#runpod-cuda-metrics-2025-11-02 for the source notebook cell with full Runpod metrics and context.

| Run ID | Device | Duration (s) | Throughput (eps/s) | Best Epoch | Train/Val/Test | Reward Mean (±std) | Turnover | AMP |
| --- | --- | ---:| ---:| ---:| --- | --- | ---:| --- |
| 212855 | Runpod L4 | 18.37 | 113.07 | 56 | 0.722 / 0.864 / 0.835 | −0.114 (±0.0103) | 3.18 | float16 |
| 153117 | Runpod L4 | 19.41 | 107.3 | 17 | 0.621 / 0.755 / 0.670 | −0.114 (±0.0100) | 2.69 | float16 |
  - Notes: results reflect a longer episode budget than the Colab smoke; AMP remained stable. Instrument memory via `torch.cuda.max_memory_allocated()` if running concurrent jobs on the pod.
- Next CUDA steps:
  - Run HISSO regression suite under CUDA once the runpod slot is available (pytest tests/test_hisso_primary.py::test_hisso_fit_sets_trainer_state -k cuda plus nightly selection).
  - Tune WaveResNet episodes/penalty if the negative reward mean persists on richer datasets; capture findings in this compendium and the README GPU appendix.
**Prior Outputs (Local)**
- Predictions (NPZ arrays) and metrics bundle: `outputs/colab_results (1)/`
  - Prediction files by task + model, e.g.,
    - `Jena_tdegc_72ctx_36h_ResPSANN_conv_spine_predictions.npz`
    - `Beijing_PM25_24h_ctx_6h_horizon_LSTM_baseline_predictions.npz`
    - `HAR_raw_sequence_TCN_baseline_predictions.npz`
  - Aggregate metrics CSV: `outputs/colab_results (1)/experiment_metrics.csv` (train/val/test blocks with RMSE/MAE/SMAPE/R2 and wall-time/params).
- Synthetic probes: `outputs/psann_synth_results (1)/`
  - `synthetic_experiment_metrics.csv` (multi-dataset synthetic parity results)
  - `synthetic_spectral_results.json` (Jacobian/PR snapshots per model on synthetic seasonal proxy)

**Experiment Plan**
- Source: `plan.txt`
- Verbatim content
```
# ResPSANN Under Compute Parity — Adapted Experiment Plan (Datasets: EAF, Beijing Air, Jena Climate, HAR, Rossmann)

## Scope & Changes

This revision aligns the original plan to the datasets described in the companion data brief. We anchor flagship robustness work on the Industrial Electric Arc Furnace (EAF) tables, use Beijing + Jena for mid‑scale multivariate forecasting and seasonality probes, deploy HAR for classification/representation tests, and include Rossmann for structured business forecasting. Synthetic families remain for stress testing but are de‑emphasized in this pass.

## Datasets & Targets

### 1) Industrial Data from the Electric Arc Furnace (EAF)

**Targets**

* Temperature forecasting: next‑step and short horizon TEMP.
* Oxidation forecasting: VALO2_PPM regression; optionally detection when measured (VALO2_PPM>0).
* Final chemical composition after tapping: multi‑output regression on available chemistry columns (through VALNI).

**Notes**

* Eleven linked CSVs spanning ~2015‑01‑01 – 2018‑07‑30; join on `HEATID`.
* Very large high‑frequency logs for gas/oxygen/carbon; temperature table ~85k rows.
* Decimal commas in numeric fields and timestamps; some duplicate TEMP rows; transformer durations string‑encoded.
* Carbon/gas usage counters accumulate and reset around heat boundaries; final composition file stops at VALNI, so downstream features expecting e.g., VALV/VALTI must be revised.

### 2) Beijing Multi‑Site Air‑Quality

**Targets**

* PM2.5 (primary), optionally PM10/NO2; 1h–6h ahead.

**Notes**

* Hourly data across 12 stations (2013‑03‑01 – 2017‑02‑28); station‑segregated files.
* Hundreds of NA gaps per station; require imputation or masking. Ideal for train/held‑out station generalization.

### 3) Jena Climate 2009–2016

**Targets**

* 6h–24h ahead temperature; optionally multivariate (humidity, pressure).

**Notes**

* 420k ten‑minute records (2009‑01‑01 – 2017‑01‑01) with standard decimals; day‑first timestamps.
* Clean seasonal structure suitable for spectral diagnostics and distribution‑shift splits.

### 4) Human Activity Recognition (HAR) — Smartphones

**Targets**

* 6‑class activity classification (Walking, Upstairs, Downstairs, Sitting, Standing, Laying).

**Notes**

* Two input options: engineered 561‑feature windows (official split), or raw 50‑Hz sequences (128x9) from Inertial Signals.
* Respect provided train/test splits by subject to avoid leakage.

### 5) Rossmann Store Sales

**Targets**

* Next‑day sales per store; optional multi‑horizon.

**Notes**

* ~1.0M training rows (2013‑01‑01 – 2015‑07‑31) + test period (2015‑08‑01 – 2015‑09‑17). Join with store metadata; encode holidays; reconcile missing `Open`.

## Preprocessing & Feature Engineering

### EAF

* Locale normalization; integrity de‑dupe; heat segmentation; per‑heat features; lag/EMA features; target variants.

### Beijing

* Station‑wise normalization; missingness handling; calendar features.

### Jena

* Windowing; temporal splits; seasonal encodings.

### HAR

* Engineered vs raw pipelines.

### Rossmann

* Joins/encodings; temporal CV.

## Splits & Validation

* EAF heat‑aware; Beijing cross‑station; Jena year‑based; HAR official; Rossmann calendar‑based; ≥5 seeds; paired tests.

## Models, Baselines & Compute Parity

* ResPSANN (primary) + tiny temporal spine; baselines (MLP/TCN/LSTM/Transformer‑lite); matched wall‑time/params.

## Experiments by Hypothesis

* H1 Generalization; H2 Information Usage (PSD/SHAP); H3 Spectral; H4 Robustness; H5 Limits & Tiny Spines.

## Metrics & Reporting

* Forecasting: RMSE/MAE/R² (+sMAPE/MASE). Classification: Acc/F1/ECE. Resources: wall‑time/params.

## Execution Order

1) EAF loaders → 2) EAF sweep → 3) Beijing station‑gen → 4) Jena geometry → 5) HAR → 6) Rossmann → 7) Aggregate.

## Artifacts & Reproducibility

* Versioned scripts, saved splits/seeds/configs, figure scripts, environment snapshot & wall‑clock calibration.
```

**Key Files**
- Light-probe script: `scripts/run_light_probes.py`
- Light-probe metrics: `colab_results_light/metrics.csv`
- Prior predictions/metrics: `outputs/colab_results (1)/`
- Synthetic results: `outputs/psann_synth_results (1)/`
- Plan: `plan.txt`

**Repro Steps**
- Prepare datasets
  - Place `datasets.zip` in project root or ensure `datasets/` contains Jena, Beijing, EAF (paths as above). The runner will extract and normalize paths.
- Run light probes
  - `python scripts/run_light_probes.py --epochs 20 --seeds 7 8`
  - Outputs to `colab_results_light/metrics.csv`
- Optional: record PR snapshots
  - Add `--pr-snapshots` to the command to write `colab_results_light/jacobian_pr.csv` (for Jena/psann_conv).

**Notes & Next Work**
- EAF lite setting is intentionally small; full EAF tasks (TEMP/O2 multi-horizon, final composition) remain for the compute-parity sweep with richer spines and feature engineering per the plan.
- Beijing results strongly favor PSANN+Conv under the current config; cross-station generalization and missingness stress tests should be surfaced next.
- Jena spectral diagnostics (Jacobian/NTK, PR over epochs) can be recorded via `--pr-snapshots` or the diagnostics cells in the research notebook.

**Instrumented Run Commands (recommended)**
- One-command full suite (light probes + synthetic ablations + GeoSparse benchmarks/sweep/micro):
  - `python scripts/run_full_suite.py --device cuda --git-commit`
- Light probes (real data lite):
  - `python scripts/run_light_probes.py --epochs 20 --seeds 0 1 2 3 4 --match-params --results-dir reports/light_probes/<stamp>`
  - Outputs: `metrics.csv`, `summary.csv`, `history.jsonl`, `env.json`, `manifest.json`.
- Synthetic ablations (drift/shock/regime + tabular + moons):
  - `python scripts/benchmark_regressor_ablations.py --datasets tabular_sine,tabular_shifted,classification_clusters,context_rotating_moons,ts_periodic,ts_regime_switch,ts_drift,ts_shock --seeds 0,1,2,3,4 --out reports/ablations/<stamp>`
  - Outputs: `results.jsonl`, `summary.csv`, `seed_summary.csv`, `env.json`, `manifest.json`.
- GeoSparse mixed-activation benchmark:
  - `python scripts/benchmark_geo_sparse_vs_dense.py --task mixed --sparse-activation mixed --activation-config <json> --out reports/geo_sparse/<stamp>`
  - Outputs: `results.json`, `summary.csv`, `manifest.json` (includes val metrics and timings).
- GeoSparse sweep:
  - `python scripts/geo_sparse_sweep.py --task mixed --activations relu,psann,mixed --seeds 0,1,2 --out reports/geo_sparse_sweep/<stamp>`
- GeoSparse microbench:
  - `python scripts/benchmark_geo_sparse_micro.py --out reports/geo_sparse_micro/<stamp>`

**Local Artifact Inventory (2026-02-05)**
- Light probes: results are embedded above; `colab_results_light/metrics.csv` is not present locally (rerun `scripts/run_light_probes.py` to regenerate).
- Synthetic ablations: `reports/ablations/20260205_110015/` (5 seeds; `results.jsonl`, `summary.csv`, `seed_summary.csv`, `env.json`, `manifest.json`).
- GeoSparse mixed benchmark: `reports/geo_sparse/20260205_131142/` (task=mixed, activation=mixed, shape=12x12 k=8; dense_respsann mismatch <1%; `results.json`, `summary.csv`, `manifest.json`).
- Prior GeoSparse mixed benchmark: `reports/geo_sparse/20260205_121543/` (shape=8x8; dense_respsann mismatch ~2.8%).
- GeoSparse sweep: `reports/geo_sparse_sweep/20260205_121654/` (24 runs, seed=0; `summary.csv`, `summary_by_model.json`).
- GeoSparse microbench: `reports/geo_sparse_micro/20260205_122042/` (layer/block timing; `summary.csv`).
- Legacy GeoSparse sweep: `reports/geo_sparse_sweep/local_smoke2/summary.csv` and `summary_by_model.json` exist (small CPU smoke); `reports/geo_sparse_sweep/local_smoke/summary.csv` contains import errors from an older run.
- HISSO runs: `runs/hisso/*/metrics.json` + `events.csv` include dense and WaveResNet CPU/CUDA smoke runs (see `docs/PSANN_Results_Compendium.md` GPU sweep section).
- GPU environment reports: `outputs/gpu_tests/*/env.json` and `SUMMARY.txt` are present.
- Notebooks: `notebooks/PSANN_Parity_and_Probes.ipynb` contains prior real/synthetic pipelines (HAR/Rossmann loaders), and `notebooks/geosparse_crypto_direction.ipynb` uses the external `psann_crypto_trading` repo + DB; no exported metrics live in this repo.
- Historical references: `benchmarks/psann_results_assessment.md` references `tmp_outputs/colab_results (1)` and `tmp_outputs/psann_synth_results (1)` which are not present in this checkout.

**Comparison Matrix (Target)**
| Dataset | Task | Models (required) | Seeds | Metrics | Status | Post |
| --- | --- | --- | --- | --- | --- | --- |
| Jena Climate (72 ctx / 36 h) | Forecasting | PSANN, ResPSANN, WaveResNet, MLP/TCN/LSTM | >=5 | MSE/RMSE/MAE/SMAPE/R2 + time/params | Light-probe (2 seeds) | Post: Real-data forecasting |
| Beijing Air (24 ctx / 6 h) | Forecasting | PSANN, ResPSANN, WaveResNet, MLP/TCN/LSTM | >=5 | MSE/RMSE/MAE/SMAPE/R2 + time/params | Light-probe (2 seeds) | Post: Real-data forecasting |
| EAF TEMP (lite) | Forecasting | PSANN, ResPSANN, WaveResNet, MLP | >=5 | MSE/RMSE/MAE/SMAPE/R2 + time/params | Light-probe (2 seeds) | Post: Real-data forecasting |
| HAR engineered / raw | Classification | PSANN/ResPSANN + baselines | >=5 | Acc/F1 (+regression metrics for parity) | Loader in notebook | Post: Real-data forecasting |
| Rossmann sales | Forecasting | PSANN/ResPSANN + baselines | >=5 | MSE/RMSE/MAE/SMAPE/R2 | Loader in notebook | Post: Real-data forecasting |
| Synthetic drift / shock / regime | Forecasting | PSANN, ResPSANN, WaveResNet | >=5 | MSE/RMSE/MAE/SMAPE/R2 + stability | CPU sweep complete (`reports/ablations/20260205_110015/`) | Post: Synthetic robustness |
| Tabular mixed / shifted | Regression | PSANN, ResPSANN, WaveResNet | >=5 | MSE/RMSE/MAE/SMAPE/R2 + stability | CPU sweep complete (`reports/ablations/20260205_110015/`) | Post: Synthetic robustness |
| Context rotating moons | Classification | PSANN, ResPSANN, WaveResNet | >=5 | Acc/F1 (+regression metrics for parity) | CPU sweep complete (`reports/ablations/20260205_110015/`) | Post: Synthetic robustness |
| GeoSparse mixed activation | Regression | GeoSparse + dense baselines | >=5 | MSE/RMSE/MAE/SMAPE/R2 + time/params | CPU mixed bench + sweep + micro (`reports/geo_sparse*`) | Post: GeoSparse mixed activation |
| Crypto direction (external) | Classification | GeoSparse mixed + dense | >=3 | Accuracy / ROC + time/params | Notebook (external repo) | Post: GeoSparse mixed activation |

**Post Mapping (Draft)**
- Real-data forecasting post: Jena/Beijing/EAF + HAR/Rossmann runs (compute parity, multi-seed).
- Synthetic robustness post: drift/shock/regime + tabular mixed/shifted + rotating-moons context tests, plus stability stats.
- GeoSparse mixed-activation post: `benchmark_geo_sparse_vs_dense.py` (task=mixed, sparse_activation=mixed), `geo_sparse_sweep.py`, and microbench results.
- Architecture/diagnostics post: SineParam/WaveResNet rationale + Jacobian/NTK/PR probes.
