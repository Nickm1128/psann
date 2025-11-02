# A4 TODO

> Progress under the assumption that no CUDA/GPU resources are available right now; prioritize CPU-only tasks until access changes.

## Test Run (2025-10-31)
- [x] `python -m pytest` (133s wall; CLI timed out at 139s after emitting the summary) &rarr; 145 passed, 1 skipped (`tests/test_training_loop.py::test_training_loop_early_stopping_runs_on_cuda`).
- [x] Existing warning filters in `tests/conftest.py` kept the log clean; no unexpected warnings surfaced.

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
- [x] Exercised `notebooks/HISSO_Logging_GPU_Run.ipynb` on CPU (auto-fallback path); both dense and wave-resnet runs completed under `C:\content\hisso_runs\...`, confirming the CLI wiring and summariser cell function without CUDA. Captured wall-times (1.74s / 2.72s) and reward/Sharpe outputs from the generated `metrics.json` files for later comparison with GPU runs.
- [x] Documented HISSO context alignment rules and the CPU-safe CUDA capture guard in `TECHNICAL_DETAILS.md` / `docs/API.md` so non-CUDA executions understand the fallback semantics.
- [x] Split linear-probe vs baseline diagnostics in `encode_and_probe`; added explicit `probe_accuracy`, `baseline_accuracy`, `baseline_metrics`, and `accuracy_source` fields plus release-note coverage (`src/psann/utils/linear_probe.py`, `tests/test_linear_probe.py`, `docs/migration.md`).
- [x] Reviewed warning stacklevels and ensured HISSO input-noise fallback warns from the caller site (`src/psann/hisso.py`) with regression coverage (`tests/test_warning_stacklevels.py`).

## Next GPU Session Targets
- [ ] Stage a CUDA-capable environment (Colab GPU runtime or local workstation). Set `INSTALL_DEPENDENCIES=True`, supply `PSANN_PACKAGE_SPEC` for the release build, and confirm `torch.cuda.is_available()`/device string before running anything else.
- [ ] Re-run the skipped accelerator test: `python -m pytest tests/test_training_loop.py::test_training_loop_early_stopping_runs_on_cuda` (or the full suite) and capture the timing plus confirmation that early stopping passes on GPU.
- [ ] Execute `notebooks/HISSO_Logging_GPU_Run.ipynb` end-to-end on CUDA: install the wheel, emit dense + WaveResNet runs, and export the `hisso_runs` artefacts (metrics JSON, events CSVs, configs).
- [ ] Back-port the GPU metrics/screenshots into `notebooks/HISSO_Logging_CLI_Walkthrough.ipynb`, `notebooks/README.md`, and README GPU sections once the GPU notebook finishes.
- [ ] Validate `_guard_cuda_capture` on real hardware (ensure no warnings; remove/adjust the guard if unnecessary) and capture fresh HISSO profiling numbers for README/docs.

## Notes
- Warning filters in `tests/conftest.py` suppress expected alias/deprecation chatter so test logs stay clean; drop those filters if we need to audit the raw warnings again.
- Next PyPI release must include the updated dependency list so `pyyaml` is available when the logging CLI consumes YAML configs (the Colab GPU notebook already installs it explicitly); set `PSANN_PACKAGE_SPEC` in the notebook to target the release or git commit that bundles `psann.scripts`.
- `notebooks/HISSO_Logging_GPU_Run.ipynb` can now run locally by setting `INSTALL_DEPENDENCIES=False` and letting the notebook auto-pick CPU when CUDA is absent; remember to flip the toggle and FORCE_DEVICE back to `auto/cuda` before running in Colab.
- CPU smoke artifacts live under `runs/hisso/dense_cpu_smoke_dev/dense_cpu_smoke_check` and `runs/hisso/wave_resnet_cpu_smoke_dev/` (`dense_cpu_smoke_check`, `wave_resnet_cpu_smoke_check2`); metrics JSON includes throughput stats ready to be merged into the notebook once GPU results arrive.
