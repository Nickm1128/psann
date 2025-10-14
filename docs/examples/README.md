# PSANN Examples

All examples live in the top-level `examples/` directory and can be executed with `python examples/<number>_<name>.py`. Create a virtual environment, install the project in editable mode, and run the scripts from the project root so relative paths resolve correctly.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -e .[viz]
python examples/21_psann_config_benchmark.py
```

The curated set below replaces the legacy predictive-extras walkthroughs. Any references to extras heads, growth schedules, or deprecated CLI flags have been removed from the documentation and code comments.

## Core supervised and streaming workflows

- **01_basic_regression.py** - minimal PSANNRegressor fit on a synthetic one-dimensional regression task.
- **05_conv_preserve_shape_regression.py** - convolution-preserving variant that keeps spatial structure instead of flattening.
- **07_recurrent_forecasting.py** - stateful PSANN with a teacher-forced rollout for sequence prediction.
- **12_online_streaming_updates.py** - demonstrates streaming updates with `step(..., update_params=True)` and `stream_lr` enabled.
- **13_online_vs_freerun_plot.py** - compares online adaptation against free-running predictions on the same series.
- **14_psann_with_vs_without_lsm.py** - contrasts a baseline PSANN with a frozen LSM expander preprocessing stage.

These scripts share the refactored helper pipeline (`normalise_fit_args`, `prepare_inputs_and_scaler`) and stick to primary targets only.

## HISSO policies and episodic evaluation

- **15_episode_training_portfolio.py** - baseline dense HISSO policy on synthetic price data with supervised warm start.
- **16_episode_training_psann_vs_lsm.py** - compares dense PSANN and LSM-augmented variants under the same reward bundle.
- **17_episode_training_conv_psann.py** - convolutional HISSO example that preserves spatial structure while optimising rewards.
- **26_hisso_unsupervised_allocation.py** - unsupervised HISSO training on synthetic prices that relies exclusively on the reward registry.
- **27_hisso_lsm_allocation.py** - attaches an LSM expander before HISSO to illustrate hybrid preprocessing + episodic optimisation.

Each HISSO script uses the neutral terminology (`transition_penalty`) and the shared helpers introduced with `HISSOOptions`. After training, utilities such as `psann.hisso.hisso_infer_series` and `psann.hisso.hisso_evaluate_reward` reuse the stored configuration.

## Benchmarks and profiling

- **21_psann_config_benchmark.py** - sweeps PSANN configurations and records results to CSV/JSON; serves as the basis for automated benchmark comparison.
- **scripts/compare_hisso_benchmarks.py** - CLI helper that diff-checks new HISSO runs against the stored baselines.
- **docs/benchmarks/hisso_variants.md** - human-readable summary of the CPU portfolio sweep captured in CI.

Run the benchmark scripts from the project root; outputs land under `docs/benchmarks/` and are checked in to catch regressions.

## Language modelling and diagnostics

- **23_psann_lm_demo.py** - small PSANN language model illustrating token embedding, training, and text generation.
- **WaveResNet diagnostics (docs/wave_resnet.md)** - covers Jacobian/NTK utilities, participation ratio, and mutual information probes for backbone analysis.

## Retired content

Notebooks and scripts that targeted predictive extras or growth schedules were removed during the documentation refresh. If you need historic behaviour, consult the v0.9.18 tag in the repository history; new development should stay on the primary-output pipeline documented here.

