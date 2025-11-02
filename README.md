# PSANN - Parameterized Sine-Activated Neural Networks

PSANN packages sine-activated Torch models behind a sklearn-style estimator surface. The stack combines:
- learnable sine activations with SIREN-friendly initialisation,
- optional learned sparse (LSM) expanders and scalers,
- persistent state controllers for streaming inference, and
- Horizon-Informed Sampling Strategy Optimisation (HISSO) for episodic training.

The current line targets **primary outputs only** so there are no predictive extras, secondary heads, or legacy growth schedules to maintain.

Quick links:
- API reference: `docs/API.md`
- Scenario walkthroughs: `docs/examples/README.md`
- Migration notes: `docs/migration.md`
- Results compendium: `docs/PSANN_Results_Compendium.md`
- Contributor guide: `docs/CONTRIBUTING.md`
- Technical design notes: `TECHNICAL_DETAILS.md`

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux
pip install --upgrade pip
pip install -e .                # editable install from source
```

Optional extras in `pyproject.toml`:
- `psann[sklearn]`: adds scikit-learn conveniences for estimator mixins and metrics.
- `psann[viz]`: plotting helpers used in benchmarks and notebooks.
- `psann[dev]`: pytest, ruff, black, coverage, build, pre-commit tooling.

Need pre-pinned builds (e.g. on Windows or air-gapped envs)? Use the compatibility extra:

```bash
pip install -e .[compat]
```

The `compat` extra pins NumPy, SciPy, scikit-learn, and PyTorch to the newest widely available wheels while keeping `pyproject.toml` as the single source of truth.

## Running Tests

Install the development extras in editable mode so the test suite imports the packaged code without manual `sys.path` tweaks:

```bash
pip install -e .[dev]
python -m pytest
```

HISSO integration suites are marked as `slow`; skip them during quick iterations with:

```bash
python -m pytest -m "not slow"
```

The suite exercises the supported supervised, streaming, and HISSO flows. GPU-specific checks are skipped automatically when CUDA is unavailable.

Common linting commands:

```bash
python -m ruff check src tests scripts examples
python -m black --check src tests scripts examples
```

Set up local hooks (formatting, linting, notebook output stripping) with `pre-commit`:

```bash
pre-commit install
pre-commit run --all-files  # optional one-time sweep
```

## Quick Start

### Supervised regression

```python
import numpy as np
from psann import PSANNRegressor

rs = np.random.RandomState(42)
X = np.linspace(-4, 4, 1000, dtype=np.float32).reshape(-1, 1)
y = 0.8 * np.exp(-0.25 * np.abs(X)) * np.sin(3.5 * X)

model = PSANNRegressor(
    hidden_layers=2,
    hidden_units=64,
    epochs=200,
    lr=1e-3,
    early_stopping=True,
    patience=20,
    random_state=42,
)
model.fit(X, y, verbose=1)
print("R^2:", model.score(X, y))
```

Behind the scenes the estimator normalises arguments via `normalise_fit_args` and prepares data/scalers through `psann.estimators._fit_utils.prepare_inputs_and_scaler`, so dense, residual, and convolutional variants share the same fit surface.

**Parameter aliases.** The constructor still accepts legacy names such as `hidden_width` and `hidden_channels`, but they are treated as deprecated aliases. Whether you pass them to `__init__` or later through `set_params`, the estimator maps them back to the canonical `hidden_units` / `conv_channels` entries and warns when both names disagree.

**Device & dtype.** The estimators operate internally in float32. Supplying `np.float32` arrays (as shown above) avoids extra copies. For GPU training, pass `device="cuda"` (or a specific `torch.device`) when constructing the estimator *before* calling `fit`; the helper will keep HISSO loops and inference on the pinned device.

### Residual regression with `ResPSANNRegressor`

```python
import numpy as np
from psann import ResPSANNRegressor

rng = np.random.default_rng(1234)
X = rng.uniform(-2.0, 2.0, size=(512, 4)).astype(np.float32)
y = (np.sin(X[:, :1]) + 0.25 * X[:, 1:2]).astype(np.float32)

est = ResPSANNRegressor(
    hidden_layers=4,
    hidden_units=48,
    lr=1e-3,
    epochs=120,
    early_stopping=True,
    patience=15,
    random_state=1234,
)
est.fit(X, y, verbose=0)
print("Residual R^2:", est.score(X, y))
```

`ResPSANNRegressor` keeps the same `.fit`/`.predict` interface but routes training through the residual backbone with DropPath, RMSNorm, and optional HISSO hooks enabled.

### Convolutional regression with `ResConvPSANNRegressor`

```python
import numpy as np
from psann import ResConvPSANNRegressor

rng = np.random.default_rng(321)
X = rng.normal(size=(64, 12, 3)).astype(np.float32)  # (N, length, channels)
y = X.mean(axis=(1, 2), keepdims=True).astype(np.float32)

conv_est = ResConvPSANNRegressor(
    hidden_layers=3,
    hidden_units=32,
    conv_channels=24,
    conv_kernel_size=3,
    epochs=60,
    batch_size=16,
    data_format="channels_last",
    random_state=321,
)
conv_est.fit(X, y, verbose=0)
print("Conv R^2:", conv_est.score(X, y))
```

Passing `data_format="channels_last"` lets you keep inputs as `(N, length, channels)` arrays; the estimator handles the channel-first conversion internally while respecting `float32` inputs and the shared alias policy.

### Wave-based regression with `WaveResNetRegressor`

```python
import numpy as np
from psann import WaveResNetRegressor

X = np.linspace(0, 2 * np.pi, 400, dtype=np.float32).reshape(-1, 1)
context = np.stack(
    [np.sin(X[:, 0]), np.cos(X[:, 0])],
    axis=1,
).astype(np.float32)
y = (np.sin(3 * X) + 0.1 * np.cos(5 * X)).astype(np.float32)

wave = WaveResNetRegressor(
    hidden_layers=4,
    hidden_units=64,
    epochs=150,
    lr=3e-4,
    w0=30.0,
    w0_warmup_epochs=20,
    progressive_depth_initial=2,
    progressive_depth_interval=25,
    random_state=7,
)
wave.fit(X, y, context=context, verbose=0)
print("WaveResNet R^2:", wave.score(X, y, context=context))
```

`WaveResNetRegressor` applies SIREN-style initialisation with optional `w0` warmup and progressive depth expansion. Providing explicit `float32` context arrays keeps inference aligned with the estimator's cached `context_dim`.

### Episodic HISSO with `HISSOOptions`

```python
import numpy as np
from psann import PSANNRegressor, get_reward_strategy, HISSOOptions

rng = np.random.default_rng(7)
X = rng.normal(size=(512, 4)).astype(np.float32)
targets = np.sin(X.sum(axis=1, keepdims=True)).astype(np.float32)

model = PSANNRegressor(hidden_layers=2, hidden_units=48, epochs=40, batch_size=64)
model.fit(X, targets, verbose=1)  # supervised warm start

finance = get_reward_strategy("finance")
options = HISSOOptions.from_kwargs(
    window=64,
    reward_fn=finance.reward_fn,
    context_extractor=finance.context_extractor,
    primary_transform="softmax",
    transition_penalty=0.05,
    input_noise=0.0,
    supervised={"y": targets},
)

model.fit(
    X,
    y=None,
    hisso=True,
    hisso_window=options.episode_length,
    hisso_reward_fn=options.reward_fn,
    hisso_context_extractor=options.context_extractor,
    hisso_primary_transform=options.primary_transform,
    hisso_transition_penalty=options.transition_penalty,
    hisso_supervised=options.supervised,
    verbose=1,
)
```

`HISSOOptions` keeps reward, context, noise, and transformation choices in one place. The estimator records the resolved options after fitting so helpers such as `psann.hisso_infer_series` and `psann.hisso_evaluate_reward` can reuse them.

### Context builders

Setting `context_builder="cosine"` (or supplying a callable) instructs the estimator to synthesise auxiliary context features during `fit`, `predict`, and sequence roll-outs. Builder parameter dictionaries are deep-copied, so mutating your original config after `set_params` will not affect the estimator. Calling `set_params(context_builder=None)` clears the cached builder and resets the inferred `context_dim`, letting you switch back to explicit context arrays cleanly.

### Custom data preparation

```python
from psann import PSANNRegressor
from psann.estimators._fit_utils import normalise_fit_args, prepare_inputs_and_scaler

est = PSANNRegressor(hidden_layers=1, hidden_units=16, scaler="standard")
fit_args = normalise_fit_args(est, X_train, y_train, hisso=False, verbose=0, lr_max=None, lr_min=None)
prepared, primary_dim, _ = prepare_inputs_and_scaler(est, fit_args)
# prepared.train_inputs / prepared.train_targets feed straight into custom loops
```

This keeps bespoke research loops aligned with the estimator's preprocessing contract without relying on deprecated extras heads.

## Core components

- **Sine activations** (`psann.SineParam`) expose learnable amplitude, frequency, and decay with optional bounds and SIREN-friendly initialisation.
- **LSM expanders** (`psann.LSM`, `psann.LSMExpander`, `psann.LSMConv2d`, `psann.LSMConv2dExpander`) provide sparse learned feature maps; `build_preprocessor` wires dict specs or modules into estimators with optional pretraining and separate learning rates.
- **State controllers** (`psann.StateController`) keep per-feature persistent gains for streaming/online workflows. Configurable via `StateConfig`.
- **Shared fit helpers** (`psann.estimators._fit_utils`) normalise arguments, materialise scalers, route through residual and convolutional builders, and orchestrate HISSO plans.
- **Wave backbones** (`psann.WaveResNet`, `psann.WaveEncoder`, `psann.WaveRNNCell`, `psann.scan_regimes`) surface the standalone components for experiments and spectral diagnostics outside the sklearn wrappers.
- **HISSO** (`psann.HISSOOptions`, `psann.hisso_infer_series`, `psann.hisso_evaluate_reward`) offers declarative reward configuration, supervised warm starts, episode construction, and inference helpers that reuse the cached configuration.
- **Utilities** (`psann.jacobian_spectrum`, `psann.ntk_eigens`, `psann.participation_ratio`, `psann.mutual_info_proxy`, `psann.encode_and_probe`, `psann.fit_linear_probe`, `psann.make_context_rotating_moons`, `psann.make_drift_series`, `psann.make_shock_series`, `psann.make_regime_switch_ts`) cover diagnostics and synthetic regimes.
- **Token helpers** (`SimpleWordTokenizer`, `SineTokenEmbedder`) remain for experiments that need sine embeddings, but no language-model trainer ships in this release.

## HISSO at a glance

1. Call `HISSOOptions.from_kwargs(...)` (or supply equivalent kwargs to `fit`) to resolve episode length, reward function, primary transform, transition penalty, context extractor, and optional noise.
2. Provide `hisso_supervised` to run a warm-start supervised phase before episodic optimisation.
3. `PSANNRegressor.fit(..., hisso=True, ...)` builds the episodic trainer using the shared fit pipeline.
4. After training, `hisso_infer_series(estimator, series)` and `hisso_evaluate_reward(estimator, series, targets=None)` reuse the cached configuration to score new data.

The project ships CPU benchmark baselines (`docs/benchmarks/`) and CI scripts (`scripts/benchmark_hisso_variants.py`, `scripts/compare_hisso_benchmarks.py`) to catch HISSO regressions.

### HISSO logging CLI

Use `python -m psann.scripts.hisso_log_run` to run HISSO sessions on remote nodes and collect reproducible artefacts. The command accepts JSON/YAML configs (see `configs/hisso/` templates) and emits:
- `metrics.json` with loss/reward/throughput summaries and optional portfolio metrics.
- `events.csv` containing append-only epoch logs and runtime notes (device, shuffle policy, AMP state).
- `checkpoints/` with the best estimator snapshot (and `latest.pt` when `--keep-checkpoints` is passed).
- `config_resolved.yaml` mirroring the resolved estimator/device settings for traceability.

Example:
```bash
python -m psann.scripts.hisso_log_run \
  --config configs/hisso/dense_cpu_smoke.yaml \
  --output-dir runs/hisso \
  --run-name dense_cpu_debug \
  --device cpu \
  --seed 7
```
When `device` points to a CUDA target and the config enables `mixed_precision`, the trainer switches to AMP + GradScaler automatically.

Looking for a guided walkthrough? Use the paired notebooks (both ship with Colab badges):
- `notebooks/HISSO_Logging_CLI_Walkthrough.ipynb` – CPU-first dry run with TODO placeholders.
- `notebooks/HISSO_Logging_GPU_Run.ipynb` – installs a published PyPI release (or a Git commit when specified via PSANN_PACKAGE_SPEC) and executes the logging CLI on CUDA to collect final metrics and checkpoints.

### Convolutional stems

`PSANNRegressor.with_conv_stem(...)` and `ResPSANNRegressor.with_conv_stem(...)` return estimators wired into the convolutional training path without instantiating the legacy `*ConvPSANNRegressor` wrappers. The helpers enable `preserve_shape`, switch training to channel-first tensors, and honour `conv_channels`, `conv_kernel_size`, and `per_element` flags. Example:
```python
est = PSANNRegressor.with_conv_stem(
    hidden_layers=2,
    hidden_units=32,
    conv_channels=16,
    conv_kernel_size=3,
    epochs=20,
    batch_size=32,
    random_state=42,
)
est.fit(images, targets)
```
Residual variants reuse the same call while producing `ResidualPSANNConv2dNet` cores when 2D inputs are supplied.

### Stateful dataloaders

When `stateful=True`, the training dataloader preserves sequence order. PSANN disables shuffling whenever `state_reset` is `"epoch"` or `"none"` so stateful models consume contiguous batches; keep the default `state_reset="batch"` to retain randomised mini-batches.

## Docs and examples

- Examples live in `examples/`; see `docs/examples/README.md` for the curated list (supervised, streaming, HISSO, benchmarks, diagnostics).
- Detailed internals are captured in `TECHNICAL_DETAILS.md`.
- Reward registry usage and custom strategy registration are described in `docs/API.md` under the HISSO section.

## Current status and roadmap

- Predictive extras and growth schedules are gone; legacy `extras_*` arguments are accepted but ignored with warnings for backward compatibility.
- Terminology has converged on `transition_penalty` within HISSO; the `trans_cost` alias still functions but will be removed in a later release.
- CPU benchmarks run in CI; initial CUDA smoke runs now live in `runs/hisso/dense/` and `runs/hisso/wave_resnet/`: dense (seed 7) duration 2.68 s, throughput 203 eps/s, train/val/test loss 0.245/0.304/0.231, reward_mean -0.111; WaveResNet (seed 11) duration 3.34 s, throughput 161 eps/s, train/val/test loss 1.435/1.402/1.569, reward_mean -0.182 (std 0.068). Colab did not expose CUDA memory metrics, so instrument `torch.cuda.max_memory_allocated()` during the runpod sweep before promoting the numbers to release notes.
- Upcoming work highlighted in `REPO_CLEANUP_TODO.md` includes broader reward coverage, lint/type sweeps, and release tooling improvements.

### Reproducibility

The notebook **PSANN_Parity_and_Probes.ipynb** (now under `notebooks/`) reproduces all key results under compute parity.  
- **Release:** [v1.0.0](https://github.com/Nickm1128/psann/releases/tag/v1.0.0)  
- **DOI:** [doi.org/10.5281/zenodo.17391523](https://doi.org/doi.org/10.5281/zenodo.17391523)  
- **Permalink:** [GitHub](https://github.com/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)  
- **Render:** [nbviewer](https://nbviewer.org/github/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)  
- **Run:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nickm1128/psann/blob/v1.0.0/notebooks/PSANN_Parity_and_Probes.ipynb)

Experiments used **Python 3.9**, dependencies pinned in `pyproject.toml` (install `[compat]` for constrained environments).
- When using the Colab notebook, set PSANN_PACKAGE_SPEC to the desired release (e.g., 'psann==0.10.10') or a git URL so the wheel includes 'psann.scripts'.
