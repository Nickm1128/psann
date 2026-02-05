# GeoSparse vs ReLU Baselines — Colab-Reproducible Notebook (psann==0.12.2) TODO

Goal: produce **one Jupyter notebook** that installs `psann==0.12.2` from PyPI and runs a clean, reproducible benchmark comparing **GeoSparse** vs **standard ReLU models** on:
- **3 synthetic datasets** (diverse task structure),
- **3 real datasets** (diverse domains / feature regimes),
while ensuring **parameter-matched models per dataset** and reporting **wall-clock training time**.

Conventions:
- Each checklist item is actionable: `- [ ]`.
- Indentation indicates subtasks.
- “Definition of done” is included where helpful.

---

## 0) Pin Scope and Success Criteria

- [x] Define what “performance” means in this notebook
  - [x] Metrics (regression-only v1): MSE, MAE, R2 (classification metrics deferred)
  - [x] Time metrics: train wall time; optionally steps/sec and samples/sec
  - [x] Report distribution across seeds (mean/std) with per-seed rows retained

- [x] Choose whether this notebook supports both regression and classification
  - [x] Decision: regression-only v1 with 6 regression datasets (3 synthetic + 3 real)
  - [x] Rationale: keeps loss/metric comparisons clean; classification can be added later

- [x] Define a strict fairness contract (written at top of notebook)
  - [x] Same optimizer family + schedule (AdamW, fixed lr schedule), same batch size
  - [x] Same number of optimization steps (preferred) across all datasets
  - [x] Same preprocessing policy (StandardScaler on X; optional y-scaling for regression)
  - [x] Same parameter count within tolerance (<= 1% relative difference)

Definition of done:
- Notebook runs end-to-end in Colab and produces a final results table + plots in < ~20 minutes in "quick mode" (1 seed, reduced steps), and documents the expected runtime for the full 3-seed run.

---

## 1) Colab-Ready Environment & Installation Cell

- [x] Create notebook at `notebooks/geosparse_vs_relu_benchmarks.ipynb`
  - [x] First report cell: hardware + versions report (after install)
    - [x] Print `psann.__version__`, `torch.__version__`, `numpy`, `sklearn`
    - [x] Detect GPU and print device name

- [x] Add a single install cell that uses PyPI (no repo checkout required)
  - [x] `pip install -q psann==0.12.2`
  - [x] Install minimal deps needed for datasets/plots: `scikit-learn`, `pandas`, `matplotlib`

- [x] Validate imports use the installed package (not local files)
  - [x] Print `psann.__file__` path (should point to site-packages)

---

## 2) Dataset Suite Selection (3 Synthetic + 3 Real)

- [x] Decide dataset list and freeze it in a single cell (easy to edit)

- [x] Synthetic datasets (pick 3 that stress different inductive biases)
  - [x] `syn_sparse_linear`: make_regression with n_features=200, n_informative=10, noise=5.0
  - [x] `syn_friedman1`: make_friedman1 with n_features=20, noise=0.5
  - [x] `syn_piecewise_sine`: regime-switch sine with noise (n_features=10)
  - [x] Finalized in notebook with explicit generation parameters and seed

- [x] Real datasets (pick 3 with diverse feature regimes and sizes)
  - [x] `real_california_housing`: fetch_california_housing (regression; mid-dimensional)
  - [x] `real_diabetes`: load_diabetes (regression; small, classic baseline)
  - [x] `real_linnerud`: load_linnerud (regression; uses single target column)
  - [x] Finalized with sources and rationale in the notebook markdown

- [x] Implement dataset loader functions with a shared signature
  - [x] `load_dataset(name) -> (X, y, task_type, feature_names, target_name)`
  - [x] `X`/`y` are cast to `float32` in the loader for consistency

---

## 3) Preprocessing & Splits (Including Optional Target Scaling)

- [x] Define consistent train/val/test splitting
  - [x] Default: `train_test_split` with fixed `test_size` and seed
  - [x] Validation split implemented from the temp set

- [x] Standardize input scaling policy
  - [x] `StandardScaler` on `X` (fit on train only)
  - [x] No leakage into val/test (train-only fit)

- [x] Optional: add target scaling for regression tasks
  - [x] `y` scaler included with inverse-transform support
  - [x] Documented in preprocessing markdown (metrics on unscaled targets)

- [x] Confirm dataset tensors are shaped consistently for both model families
  - [x] Shape conventions documented in a markdown cell

---

## 4) Models Under Test (GeoSparse vs ReLU Baselines)

- [x] Identify the exact public API to construct GeoSparse models in `psann==0.12.2`
  - [x] `from psann import GeoSparseRegressor, PSANNRegressor`
  - [x] Estimators support CPU/GPU training and sklearn-style `fit/predict`

- [x] Implement ReLU baselines using similarly “native” PSANN components (preferred)
  - [x] Dense MLP ReLU baseline via `PSANNRegressor(activation_type='relu')`
  - [x] (Optional) Sparse ReLU baseline deferred for now

- [x] Ensure both model families support:
  - [x] Same input/output dimensions (handled via common dataset tensors)
  - [x] Same batch size and optimizer settings (shared defaults in notebook)
  - [x] Deterministic seeding controls (`random_state` in estimators)

---

## 5) Parameter-Matching (Critical)

- [x] Define the parameter budget policy
  - [x] Choose GeoSparse config first, then match dense baseline width to GeoSparse param count

- [x] Implement a robust parameter-count function
  - [x] `count_params(model) -> int` using actual PyTorch modules (includes PSANN params)
  - [x] Count both GeoSparseNet and PSANNNet for accurate comparison

- [x] Implement a matching routine with tolerance
  - [x] Search width candidates for closest match to target params
  - [x] Warn if `rel_mismatch > tol` (default tol=1%)
  - [x] Documented in notebook markdown

- [x] Confirm “apples-to-apples” with GeoSparse extra params (if applicable)
  - [x] GeoSparse counts include PSANN sine parameters via module-based counting

---

## 6) Training Loop + Accurate Wall-Clock Timing

- [x] Implement a unified training harness function
  - [x] Inputs: model factory, dataset tensors, batch size, seed, target steps
  - [x] Outputs: model, epochs, steps, wall time, steps/sec, samples/sec

- [x] Timing best practices (especially on GPU)
  - [x] Warm-up run on a throwaway model to avoid first-iteration overhead
  - [x] `torch.cuda.synchronize()` before/after timing regions when on CUDA
  - [x] Timing focuses on full fit; transfer breakdown optional for later

- [x] Define training budget per run
  - [x] Target fixed number of optimizer steps, compute epochs from dataset size
  - [x] Record actual total steps and samples seen per run

- [x] Add progress prints (Colab-friendly)
  - [x] Progress callback prints loss every N steps (epoch-based approximation)
  - [x] End-of-run summary handled by training harness return values

---

## 7) Evaluation & Reporting

- [x] Implement evaluation function for each task type
  - [x] Regression: MSE/MAE/R² computed on unscaled `y` (inverse-transform if scaled)
  - [x] Classification deferred for regression-only v1

- [x] Create a single results table (pandas DataFrame)
  - [x] Columns: dataset, model, params, train_time_s, metrics, seed (expected)
  - [x] Aggregate view: mean/std per dataset+model

- [x] Add plots that make comparisons obvious
  - [x] Metric vs model (bar chart) per dataset
  - [x] Train time vs model (bar chart) per dataset
  - [x] Pareto plot left as optional follow-up

---

## 8) Reproducibility & Stability Checks

- [x] Run each config with multiple seeds (recommended: 3)
  - [x] Seeding function sets numpy, torch, and python random
  - [x] `SEEDS` list and `QUICK_MODE` toggles are defined

- [x] Add a “sanity check” cell
  - [x] Param-match sanity check per dataset
  - [x] Train time presence check in `results` rows
  - [x] Metric directionality noted in evaluation section

- [x] Add an optional “CPU mode” section (for users without GPU)
  - [x] `QUICK_MODE` reduces seeds and steps; works on CPU-only runtime

---

## 9) Notebook Polish (Hand-off Quality)

- [x] Make the notebook self-contained and readable
  - [x] Intro cell added with goal + fairness contract
  - [x] Clear section headers and short narrative text inserted before major sections
  - [x] Cells segmented by concept (install, data, preprocessing, models, timing, eval)

- [x] Add a “Reproduce in Colab” note
  - [x] Runtime settings + GPU guidance noted (includes Python version note)
  - [x] Expected runtime range (quick vs full)

- [x] Add a final “Conclusions” section
  - [x] Placeholder bullets for summary and next tests

---

## 10) Optional Extensions (Only If Time Allows)

- [ ] Add a distilled/soft-target variant (if appropriate for task type)
- [ ] Add sensitivity analysis (vary param budget / steps)
- [ ] Add robustness checks (noise, feature scaling variants)
- [ ] Export results as CSV/JSON for easy external plotting
