# HISSO Remote Logging Script Specification

## Goals
- Provide a single entry point for running HISSO fine-tuning or evaluation loops on remote GPU nodes.
- Emit structured summaries that can be transported back into the local workspace for analysis and documentation.
- Keep runtime dependencies minimal so the script can run inside the existing virtual environment without extra tooling.

## Usage Overview
```bash
python -m psann.scripts.hisso_log_run \
  --config configs/hisso/wave_resnet_small.yaml \
  --output-dir /path/to/logs \
  --device cuda:0 \
  --run-name wave_small_2025_10_28
```

### Recommended Logging Directories

- Local shells: use `runs/hisso/` as the project-relative root. This keeps run artifacts versionable and easy to diff.
- Colab/Runpod: use `/content/hisso_logs/` so artifacts persist in the notebook workspace or can be zipped for download.
- The CLI intentionally requires an explicit `--output-dir`; no implicit defaults are applied.

## Inputs
- `--config`: Path to a YAML/JSON configuration describing:
  - Estimator factory (class name + kwargs).
  - HISSO options (window, reward, primary transform, supervised warm-start params).
  - Dataset loader module/function or serialized NumPy file paths.
  - Training/evaluation hooks (epochs, batch size, patience, learning rate schedule).
- `--output-dir`: Directory where logs, checkpoints, and metrics artifacts are stored.
- `--run-name`: Optional identifier for distinguishing multiple runs.
- `--device`: Torch device string (`cpu`, `cuda:0`, etc.).
- `--seed`: Optional integer for reproducibility; defaults to 42.
- `--keep-checkpoints`: Flag to retain intermediate checkpoints; otherwise only the best checkpoint is persisted.

## Outputs
- `metrics.json`: Structured summary with fields
  - `train_loss`, `val_loss`, `best_epoch`.
  - HISSO-specific aggregates: `reward_mean`, `reward_std`, `episodes`, `transition_penalty`.
  - Device/runtime info: `device`, `duration_seconds`, `throughput_eps_per_sec`.
  - Optional evaluation metrics from `psann.metrics.portfolio_metrics`.
- `checkpoints/`: Directory containing
  - Best estimator serialized via `PSANNRegressor.save`.
  - Optional warm-start checkpoint when HISSO supervised pretraining runs.
- `events.csv`: Append-only, human-readable log summarizing epochs, losses, rewards, LR schedule, and dataloader timing.
- `config_resolved.yaml`: Snapshot of the resolved configuration (including defaulted values) for traceability.

## Logging Mechanics
- Use Python logging module with two handlers:
  - Console handler (INFO level) for progress in remote terminal.
  - File handler (DEBUG level) writing to `events.csv`.
- Structured metrics aggregated into an in-memory dict and flushed to `metrics.json` on completion.
- Capture exceptions; write stack trace to `metrics.json::error` and exit with code 1.

## Integration Points
- Reuse `prepare_inputs_and_scaler` and `maybe_run_hisso` from `src/psann/estimators/_fit_utils.py` to minimise duplication.
- For dataset loading, support:
  - Numpy `.npz` files with keys `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`.
  - Python import path to a function returning the above arrays.
- Wrap the training invocation in `torch.cuda.amp.autocast` and gradient scaling when the config sets `mixed_precision=true`.
- Record dataloader shuffle decisions and stateful constraints in `events.csv` (helps Outstanding #7).

## AMP Behavior & Stability

Guidance (approved):
- Enable mixed precision via `mixed_precision=true` in the config on CUDA jobs; keep AMP `float16` with `GradScaler` enabled.
- Prefer normalized inputs and stable reward shaping; keep the scaler to shield rare sharp gradients.
- For debugging numerics, set `mixed_precision=false` temporarily; re-enable once stable.

- Hardware/stack: NVIDIA L4 (Runpod), CUDA 12.1, Torch 2.8.0+cu128; `autocast(enabled=True, dtype=torch.float16)` with `GradScaler` enabled for training steps when `mixed_precision=true`.
- Stability: No `inf`/`NaN` observed in losses or rewards on WaveResNet HISSO runs. No overflow/underflow warnings emitted by `GradScaler` in logs for the latest runs.
- Throughput and wall time (WaveResNet, small config): ~107–113 episodes/sec; ~18–19 s total for 1920 episodes on L4.
- Best epoch and metrics (example, 2025‑11‑02 run): `best_epoch=56` with train/val/test ≈ `0.722 / 0.864 / 0.835`; reward_mean ≈ `−0.114 ± 0.0103`; turnover ≈ `3.18`; AMP `float16` active.
- Scaler behavior: Dynamic loss scaling remained stable (no scale reductions logged under steady conditions). Retain scaler for safety around rare sharp gradients; expect steady scales on this workload.
- Caveats:
  - Prefer normalized inputs and stable reward shaping to avoid out‑of‑range magnitudes under fp16.
  - When extending to larger spines or longer episodes, keep `GradScaler` enabled and probe memory via `torch.cuda.max_memory_allocated()` if running multiple jobs per GPU.
  - For debugging numerics, temporarily set `mixed_precision=false` to compare traces; then re‑enable AMP once stable.

## Next Steps
1. Implement CLI under `psann/scripts/hisso_log_run.py` following this spec.
2. Add configuration templates in `configs/hisso/`.
3. Extend tests with a lightweight integration smoke (mark as slow) to validate metric emission on CPU.
