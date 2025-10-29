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

**Methods**
- PSANN Conv Spine (`scripts/run_light_probes.py:PSANNConvSpine`)
  - `PSANNConv1dNet` temporal backbone with small strided Conv1d; global temporal aggregator (last/mean); linear head to horizon.
- MLP Regressor (`scripts/run_light_probes.py:MLPRegressor`)
  - Flattened input → [Linear, ReLU]×depth → Linear(out_dim).
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

**Pending GPU sweep (runpod)**
- Tooling: use `python -m psann.scripts.hisso_log_run --config <yaml> --output-dir runs/hisso --run-name <tag> --device cuda:0 --seed 42` (see `configs/hisso/` templates). The WaveResNet config expects `datasets/wave_resnet_small.npz`; stage it locally or override `--config` with the remote path before launching on runpod.
- Capture: copy `metrics.json`, `events.csv`, `checkpoints/best.pt`, and `config_resolved.yaml` back into `runs/hisso/<tag>/` locally after each remote run.
- `configs/hisso/dense_cpu_smoke.yaml` now produces price matrices aligned with the primary dimension; the run should succeed once rerun.
- Experiments queued
  - HISSO WaveResNet small (`configs/hisso/wave_resnet_small.yaml`) on synthetic portfolio data.
- Dense CPU baseline rerun after staging the GPU node.
- Reporting: update this compendium with throughput, reward curves, and memory figures once artifacts land; mirror highlights in `README.md` and notebooks (`PSANN_WaveResNet_Context_Demo.ipynb`, `PSANN_SineParam_Comparison.ipynb`) so guidance matches the CLI entry point.

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

* Two input options: engineered 561‑feature windows (official split), or raw 50‑Hz sequences (128×9) from Inertial Signals.
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
