# Sparse 3D / Geometric-Connectivity Module - TODO

## Agent operating instructions (must follow)

1. **Start with repo reconnaissance before coding**: use `rg` to find existing patterns (training loops, parameter counting, activation configs) and reuse them instead of inventing new APIs.
2. **Obey all `AGENTS.md` scope rules**: before editing any file, confirm which `AGENTS.md` applies; if instructions conflict, prefer the most local file.
3. **Keep diffs surgical**: avoid refactors/renames unless required to land the module + benchmark; add new files over reshaping existing ones.
4. **Prototype -> validate -> optimize**: first land a minimal correct sparse layer + forward/backward + benchmark harness; only then pursue GPU/compile optimizations.
5. **No Python loops in hot paths**: forward/backward for the sparse layer must be vectorized (gather/matmul/scatter) and compatible with `torch.compile`.
6. **Precompute connectivity once**: generate and store connectivity indices as `register_buffer(...)` tensors; never regenerate per batch/step.
7. **Make benchmarks fair and reproducible**: same dataset split, seed, optimizer, dtype, batch size, epochs, and stopping rules across models; log all of it.
8. **Match parameter counts explicitly**: implement both (a) analytic parameter formulas and (b) empirical `sum(p.numel())` checks; use an automated search to match dense MLP params within a tight tolerance.
9. **Measure time correctly**: record wall-clock (end-to-end) and GPU step-time (via `torch.cuda.Event`) separately; include warm-up (and clearly separate/annotate `torch.compile` compile time).
10. **Prefer existing project utilities**: reuse `psann.utils.seed_all`, existing dataset builders, and logging conventions in `scripts/` where possible.
11. **Add tests for new primitives**: include at least shape + determinism + gradient tests under `tests/` before adding optional optimizations.
12. **Design for Grace Blackwell by default**: support bf16/TF32, contiguous tensors, static shapes, and `torch.compile`-friendly code; avoid sparse formats that do not accelerate on modern GPUs.
13. **Log environment for every run**: GPU name, driver/CUDA, PyTorch version, `torch.compile` backend/mode, dtype, and effective batch size.
14. **Avoid dependency creep**: do not add new libraries unless they clearly unlock performance (e.g., Triton); if added, gate them behind optional extras.
15. **Document assumptions + formulas**: write down the exact connectivity rule, parameter scaling equations, and shape conventions so results are interpretable.

---

## Concept (current working definition)

Build a **geometric-connectivity neural network** where each neuron connects to a *fixed-size subset* of neurons in the next layer (fan-out or fan-in = `k`, e.g. `k=8`). Neurons can be arranged in a **2D layer geometry** (e.g., `H x W` square grid), stacked through depth (a "3D" network in the sense of a stack of 2D layers). Add **ResNet-style residual blocks** to make the network deep and stable.

Key goals:
- **Parameter scaling**: per-layer parameters scale ~`O(n * k)` instead of dense ~`O(n^2)` (for same-width layers).
- **Pluggable activations**: support ReLU and existing PSANN/WaveResNet-style sine activations/configs.
- **Benchmarking**: compare vs traditional dense ReLU MLPs with **matched parameter counts** and **recorded wall-clock training time**.
- **Grace Blackwell optimization path**: vectorized kernels, mixed precision, compile-friendly design, and optional kernel specialization.

---

## Phase 0 output: definitions + API decisions (draft v0)

- **Connectivity convention**: use **fan-in k** per output neuron. Connectivity stored as `in_index_per_out` with shape `(n_out, k)` where values are input indices in `[0, n_in)`. Optional edge list form: `src_index` and `dst_index` of length `E = n_out * k` for scatter-based implementations.
- **Forward mapping**: gather `x[:, in_index_per_out] -> (B, n_out, k)`, multiply by per-edge weights `(n_out, k)`, sum over `k`, then add optional bias `(n_out,)`.
- **Geometry**: layer shape specified as `(H, W)` with `n = H * W`. Flatten order is row-major: `idx = row * W + col`. v0 uses the same `(H, W)` for all hidden layers; future work can add resampling between shapes.
- **Connectivity patterns (v0)**:
  - local neighborhood: offsets list or Chebyshev radius `r` with candidate offsets in `[-r, r]`.
  - random within radius: sample `k` inputs from the candidate set with deterministic RNG (seeded by base seed + layer index).
  - hash-based deterministic mapping: PRNG seeded by `(base_seed, layer_index, out_index)` to pick `k` inputs; still precompute indices once.
  - edge handling via `wrap_mode` in `{ "clamp", "wrap" }` with `clamp` default.
- **Residual block**: pre-norm by default. Block = `x + alpha * f(x)` with `alpha` learnable and initialized near 0. `f(x)` is `sparse_linear -> activation -> sparse_linear`, optional DropPath, optional RMSNorm/LayerNorm.
- **Activation interface**: `activation_type` in `{ "relu", "tanh", "psann", "phase_psann" }` plus `activation_config` (same keys as `ActivationConfig`). For `"psann"` use `SineParam`; for `"phase_psann"` use `PhaseSineParam`; for ReLU/Tanh ignore `activation_config`.
- **Loss interpretation**: "loss based on residuals" means standard supervised loss with residual connections inside the model. No auxiliary residual loss in v0.
- **Parameter formulas**:
  - sparse layer: `params = n_out * k + (bias ? n_out : 0)`
  - dense layer: `params = n_in * n_out + (bias ? n_out : 0)`
  - for same-shape layers `n = H * W`, sparse scales as `O(n * k)` vs dense `O(n^2)`
- **API sketch (minimal)**:
  - `GeoSparseConfig(shape=(H, W), depth, k, connectivity="local|random|hash", radius_or_offsets, wrap_mode, activation_type, activation_config, norm, drop_path, residual_alpha_init, bias, seed)`
  - `GeoSparseNet(input_dim, output_dim, cfg)` with validation that `input_dim == H * W` for geometric layers.

---

## Phase 1 output: repo reconnaissance (notes)

- **Activations**:
  - `src/psann/activations.py` provides `SineParam` and `PhaseSineParam` (supports `learnable`, `decay_mode`, `bounds`).
  - `src/psann/types.py` defines `ActivationConfig` (TypedDict) used across models.
  - `src/psann/nn.py` already supports `activation_type` in `{psann, relu, tanh}` and includes `ResidualPSANNBlock`, `RMSNorm`, and `DropPath`.
  - `src/psann/models/wave_resnet.py` uses `ActivationConfig` and `_build_activation_module` patterns with `SineResidualBlock` in `src/psann/layers/sine_residual.py`.
- **Training loop + seeding**:
  - Shared loop lives in `src/psann/training.py` (`TrainingLoopConfig`, `run_training_loop`).
  - sklearn-style estimators call that loop via `src/psann/estimators/_fit_utils.py`.
  - Reproducibility helper: `seed_all` in `src/psann/utils/__init__.py`.
- **Synthetic data builders**:
  - `src/psann/utils/synthetic.py` has `make_regime_switch_ts`, `make_drift_series`, `make_shock_series`.
  - `scripts/benchmark_regressor_ablations.py` defines multiple tabular/sequence datasets and a standard train/test split flow.
- **Benchmarking/logging conventions**:
  - `scripts/benchmark_regressor_ablations.py` writes `manifest.json`, `results.jsonl`, and `summary.csv` under `reports/ablations/<timestamp>_regressor_ablations`.
  - `scripts/gpu_env_report.py` captures GPU/torch details into `outputs/gpu_tests/<timestamp>/env.json` with a `SUMMARY.txt`.
- **Parameter counting**:
  - `scripts/count_psannlm_params.py` uses `sum(p.numel())` (total and trainable), a useful pattern to reuse.
- **Recommended module placement**:
  - Primitive layer + connectivity: `src/psann/layers/geo_sparse.py`.
  - Network + residual blocks: new `src/psann/nn_geo_sparse.py` (to avoid bloating `src/psann/nn.py`), or add a small wrapper in `src/psann/nn.py` if preferred.
  - Benchmark script: `scripts/benchmark_geo_sparse_vs_dense.py`, following the `reports/` layout above.
- Optional estimator: add `GeoSparseRegressor` in `src/psann/sklearn.py` only if we need sklearn-style pipelines.

---

## Phase 2 output: connectivity generator + tests

- Added connectivity builder and edge-list helper in `src/psann/layers/geo_sparse.py`.
- Implemented determinism, index range, and edge-count tests in `tests/test_geo_sparse.py`.
- Kept the generator deterministic with explicit seeds and precomputation-only loops.

---

## Phase 3 output: sparse layer primitive + tests

- Implemented `GeoSparseLinear` with gather/scatter modes in `src/psann/layers/geo_sparse.py`.
- Added forward + gradient equivalence tests and scatter/gather parity in `tests/test_geo_sparse.py`.
- Exported new layer/helpers in `src/psann/layers/__init__.py`.

---

## Phase 4 output: residual network stack

- Added `GeoSparseResidualBlock` and `GeoSparseNet` in `src/psann/nn_geo_sparse.py`.
- Implemented pre-norm residual blocks with DropPath and learnable residual scale `alpha`.
- Added forward shape tests for the new network in `tests/test_geo_sparse.py`.

---

## Phase 5 output: activation plug-in adapter

- Added an activation-config adapter to accept both `ActivationConfig` and LM-style keys (amp/freq/damp init, bounds, trainable).
- Kept `psann`, `phase_psann`, `relu`, and `tanh` activation choices in `src/psann/nn_geo_sparse.py`.
- Added tests to confirm alias keys and phase parameters are honored in `tests/test_geo_sparse.py`.

---

## Phase 6 output: parameter counting utilities

- Added analytic parameter formulas and `count_params` helper in `src/psann/params.py`.
- Exposed param helpers at the package level via `src/psann/__init__.py`.
- Added tests for formulas and dense-width matching in `tests/test_geo_sparse.py`.

---

## Phase 7 output: benchmark script (matched params + timing)

- Added `scripts/benchmark_geo_sparse_vs_dense.py` to compare GeoSparseNet vs dense ReLU MLP.
- Implemented parameter matching, wall-clock timing, GPU step-time metrics, and optional `torch.compile`/AMP controls.
- Logged manifest/results/summary outputs under `reports/geo_sparse/<timestamp>/`.

---

## Additional: sklearn-style estimator wrapper

- Added `GeoSparseRegressor` in `src/psann/sklearn.py` with GeoSparseNet backbone.
- Exported the estimator via `src/psann/__init__.py`.
- Added a smoke test in `tests/test_geo_sparse_regressor.py`.

---

## Phase 8 output: Grace Blackwell optimization pass

- Added `scripts/benchmark_geo_sparse_micro.py` to microbenchmark GeoSparseLinear and GeoSparseResidualBlock (forward + backward).
- Ensured sparse connectivity buffers are contiguous for better GPU memory access in `src/psann/layers/geo_sparse.py`.
- Included compile/AMP/TF32 controls and environment logging in the microbenchmark outputs.

---

## Phase 11 output: benchmark script refresh

- Updated `scripts/benchmark_geo_sparse_vs_dense.py` to log richer environment metadata and param mismatch ratios.
- Added a parameter mismatch tolerance check and tracked trainable parameter counts for both models.
- Captured effective AMP dtype in the manifest for reproducibility.

---

## Phase 9 output: documentation + examples

- Added `docs/geo_sparse.md` covering connectivity rules, parameter scaling, and usage examples.
- Documented benchmark and microbenchmark commands with expected output artifacts.

---

## Phase 10 output: experiment plan + sweep tooling

- Added `scripts/geo_sparse_sweep.py` to run parameterized sweeps and aggregate results.
- Documented sweep usage and outputs in `docs/geo_sparse.md`.
- Generates summary CSV, model-aggregate JSON, and optional plots under `reports/geo_sparse_sweep/<timestamp>/`.

---

## TODO (chronological)

### 0) Align on definitions + API surface (1-2 hours)
- Write a short design note (in this file or `docs/`) that fixes:
  - "fan-out" vs "fan-in" convention and how it maps to forward compute.
  - Layer geometry spec: `(H, W)` (optionally `(H, W, C)` later) and flattening order.
  - Connectivity pattern options:
    - local neighborhood (grid offsets)
    - random within radius
    - deterministic hash-based mapping (for reproducibility)
  - Residual structure: block = `x + f(x)` (pre-norm or post-norm), optional scaling (`alpha`) like `src/psann/nn.py:ResidualPSANNBlock`.
  - Activation interface: how to select from `relu`, `tanh`, `psann` (`SineParam`), `phase_psann` (`PhaseSineParam`), and any existing WaveResNet/ResPSANN config knobs.

### 1) Repo reconnaissance (30-60 min)
- Locate and reuse patterns:
  - Activations: `src/psann/activations.py`, `src/psann/nn.py`
  - Training + seeding: `src/psann/training.py`, `src/psann/utils.py` (`seed_all`)
  - Existing benchmark conventions: `scripts/benchmark_regressor_ablations.py`, `bench_psann_lm.py`
  - Param counting helpers: `scripts/count_psannlm_params.py` (and any shared utility)
- Decide new module placement (recommendation):
  - Primitive layer(s): `src/psann/layers/geo_sparse.py`
  - Model wrapper(s): `src/psann/nn_geo_sparse.py` or add to `src/psann/nn.py` if consistent
  - Optional sklearn-style estimator: `src/psann/sklearn.py` (only if it clearly helps benchmarking)

### 2) Implement connectivity generator (CPU, deterministic) + tests
- Implement a pure function (or small class) that outputs:
  - `src_index` and `dst_index` edge lists (shape `(E,)`) **or**
  - `in_index_per_out` (shape `(n_out, k_in)`) for gather-based matmul
- Must support:
  - fixed seed determinism
  - `k` hyperparameter
  - geometry-aware mapping (grid offsets) with edge handling (wrap/clamp)
  - "same-shape" layers first; optional resampling between different `(H, W)` later
- Tests (under `tests/`):
  - determinism for a fixed seed
  - correct index ranges / no out-of-bounds
  - expected `E = n_out * k_in` (or equivalent) invariants

### 3) Implement the sparse layer primitive (correctness-first)
- Implement a `torch.nn.Module` sparse linear layer with fixed connectivity:
  - Parameters: per-edge weights (+ optional per-output bias)
  - Buffers: connectivity indices
  - Forward options to evaluate:
    1) **Gather + dot**: for each output neuron, gather `k_in` inputs and compute dot with weights.
    2) **Edge list scatter-add**: compute per-edge contribution then `index_add_` into outputs.
- Requirements:
  - Supports batched inputs `(B, n_in)` (and optionally `(B, H, W)` via reshape helpers).
  - Autograd works without custom backward.
  - Minimal allocations in forward (preallocate or reuse when feasible).
- Tests:
  - shape tests
  - gradient check on tiny sizes (finite-diff style or compare to dense when `k_in = n_in`)

### 4) Build the deep residual network on top (ResNet-style)
- Create a block + network that mirrors the stability tricks already used in `src/psann/nn.py`:
  - optional RMSNorm/LayerNorm
  - optional DropPath
  - learnable residual scale `alpha` (init near 0)
- Decide how "residual loss" is interpreted:
  - default: standard loss on final output, residual connections inside model
  - optional: deep supervision (aux losses) or explicit residual-prediction head (only if justified)
- Ensure activation plug-in points exist at least:
  - within block MLP (sparse layer -> activation -> sparse layer)
  - optional final head

### 5) Activation plug-in system (reuse existing configs)
- Support:
  - `relu`, `tanh` (standard baselines)
  - `psann` via `SineParam` with `ActivationConfig`
  - `phase_psann` via `PhaseSineParam`
- If WaveResNet/ResPSANN "activation configs" differ, design an adapter layer so the new module can accept the same knobs without copy/paste.

### 6) Parameter counting utilities (analytic + empirical)
- Implement:
  - `count_params(model) -> int` (empirical)
  - analytic parameter formulas for:
    - sparse layer (`E + bias`)
    - sparse residual block and full network
    - dense ReLU MLP baseline (for given widths/layers)
- Add a small script or function to print:
  - `n_params_sparse`, `n_params_dense`, absolute and relative mismatch

### 7) Benchmark script: sparse-vs-dense (matched params + timing)
- Add a new script (suggestion): `scripts/benchmark_geo_sparse_vs_dense.py`
- Must do:
  - choose a dataset (start with synthetic tabular regression like existing scripts; later add image-ish 2D grids)
  - build sparse model from `(H, W, depth, k, activation, residual_cfg, ...)`
  - build dense baseline MLP (ReLU) with *automatically chosen width* to match parameter count:
    - brute-force search over width (and optionally layers) to minimize mismatch
    - enforce tolerance (e.g. <= 1% mismatch) or report best-effort with clear warning
  - train both with the same loop (same optimizer/hparams) and record:
    - wall-clock (start -> end)
    - per-step/epoch timing (CPU and GPU timers)
    - final metrics (loss, maybe R2 for regression)
    - throughput (samples/sec)
    - peak GPU memory (optional)
  - write results to `CSV` + `JSON` in a predictable location (`reports/` or `outputs/`)

### 8) Grace Blackwell optimization pass (after correctness + benchmark)
- Make the sparse layer fast on GPU:
  - avoid dynamic shapes; keep `k` fixed, indices fixed, and tensor shapes static
  - enable bf16 autocast; allow TF32 for matmul paths
  - test `torch.compile` (inductor) for speedups; separate compile time from steady-state timing
  - microbenchmark forward/backward of the sparse layer and block
- Optional (only if profiling shows it matters):
  - implement a Triton kernel for gather-dot or scatter-add
  - consider CUDA graphs for fixed-shape training steps

### 9) Add documentation + examples
- Doc page explaining:
  - connectivity patterns and geometry
  - parameter scaling vs dense MLP
  - how to run the benchmark script
- Minimal example training invocation and expected outputs.

### 10) Experiment plan (what to run once implemented)
- Sweep:
  - `k in {4, 8, 16, 32}`
  - depth (shallow vs deep with residuals)
  - geometry sizes `(H, W)` and whether depth changes shape
  - activation: ReLU vs PSANN sine activations
  - compile on/off, bf16 on/off
- Record:
  - param count, wall-clock, throughput, final metric, stability notes (divergence/NaNs)
- Produce one summary table + plot(s) under `reports/`.
