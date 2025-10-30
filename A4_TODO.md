# A4 TODO

> Progress under the assumption that no CUDA/GPU resources are available right now; prioritize CPU-only tasks until access changes.

## Test Run (2025-10-30)
- [x] `python -m pytest` &rarr; 144 passed, 1 skipped (`tests/test_training_loop.py::test_training_loop_early_stopping_runs_on_cuda`)
- [x] Re-ran after installing targeted warning filters (`tests/conftest.py`) &rarr; 144 passed, 1 skipped, 0 emitted warnings.

The only skipped test requires a CUDA-capable device; the rest of the suite completes successfully on CPU.

## Recent CPU Progress
- [x] Suppressed intentional alias/deprecation compatibility warnings via pytest configuration (`tests/conftest.py`).
- [x] Drafted the CPU-first HISSO logging CLI walkthrough notebook with Colab pointer and TODO placeholders (`notebooks/HISSO_Logging_CLI_Walkthrough.ipynb` + `notebooks/README.md`).
- [x] Added a GPU-ready Colab automation notebook that installs the PyPI wheel and executes the logging CLI on CUDA (`notebooks/HISSO_Logging_GPU_Run.ipynb`) and linked it from the docs (`notebooks/README.md`, `README.md:265`).
- [x] Made the GPU notebook runnable in non-CUDA environments: optional dependency installation, automatic device fallback, and timezone-aware run names (`notebooks/HISSO_Logging_GPU_Run.ipynb`).
- [x] Declared `pyyaml` as a core dependency (`pyproject.toml`) and updated the GPU notebook bootstrap cell to install it and use timezone-aware timestamps, avoiding YAML import errors and `utcnow` deprecation warnings when running on Colab.
- [x] Re-ran HISSO logging CLI CPU smoke baselines:
  - `configs/hisso/dense_cpu_smoke.yaml` → `runs/hisso/dense_cpu_smoke_dev/dense_cpu_smoke_check` (`train_loss=0.122`, `val_loss=0.116`, reward_mean≈-0.111, duration≈1.48s).
  - `configs/hisso/wave_resnet_cpu_smoke.yaml` → `runs/hisso/wave_resnet_cpu_smoke_dev/wave_resnet_cpu_smoke_check` (`train_loss=0.583`, `val_loss=0.682`, reward_mean≈-0.127, duration≈3.09s). Sharpe returns NaN due to zero variance; captured in metrics for follow-up.
- [x] Hardened portfolio metrics against underflow: clamped return denominators, sanitized NaNs/Infs, and filtered Sharpe inputs (`src/psann/metrics.py`). Added regression coverage (`tests/test_metrics_rewards.py::test_portfolio_metrics_handles_underflow_without_nan`) and re-ran the wave-resnet CPU smoke (`runs/hisso/wave_resnet_cpu_smoke_dev/wave_resnet_cpu_smoke_check2` now reports `sharpe=-2.21` with no runtime warnings).

## Outstanding Items
- [ ] Secure a CUDA environment and rerun the skipped GPU early-stopping test to validate the training loop on accelerators.
- [ ] Run `notebooks/HISSO_Logging_GPU_Run.ipynb` against the published wheel to capture CUDA metrics/logs, then fold the outputs (and screenshots) back into the CPU walkthrough notebook.

## Notes
- Warning filters in `tests/conftest.py` suppress expected alias/deprecation chatter so test logs stay clean; drop those filters if we need to audit the raw warnings again.
- Next PyPI release must include the updated dependency list so `pyyaml` is available when the logging CLI consumes YAML configs (the Colab GPU notebook already installs it explicitly); set `PSANN_PACKAGE_SPEC` in the notebook to target the release or git commit that bundles `psann.scripts`.
- `notebooks/HISSO_Logging_GPU_Run.ipynb` can now run locally by setting `INSTALL_DEPENDENCIES=False` and letting the notebook auto-pick CPU when CUDA is absent; remember to flip the toggle and FORCE_DEVICE back to `auto/cuda` before running in Colab.
- CPU smoke artifacts live under `runs/hisso/dense_cpu_smoke_dev/dense_cpu_smoke_check` and `runs/hisso/wave_resnet_cpu_smoke_dev/` (`dense_cpu_smoke_check`, `wave_resnet_cpu_smoke_check2`); metrics JSON includes throughput stats ready to be merged into the notebook once GPU results arrive.
